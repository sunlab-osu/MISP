""" Decoder for the SQL generation problem."""

from collections import namedtuple, defaultdict
import numpy as np

import torch
import torch.nn.functional as F
from . import torch_utils

from .token_predictor import PredictionInput, PredictionInputWithSchema
import EditSQL.data_util.snippets as snippet_handler
from . import embedder
from EditSQL.data_util.vocabulary import EOS_TOK, UNK_TOK

from MISP_SQL.utils import SELECT_AGG_v2, WHERE_COL, WHERE_OP, WHERE_ROOT_TERM, GROUP_COL, HAV_AGG_v2, \
    HAV_OP_v2, HAV_ROOT_TERM_v2, ORDER_AGG_v2, ORDER_DESC_ASC, ORDER_LIMIT, IUEN_v2, OUTSIDE, END_NESTED, \
    helper_find_closest_bw
from MISP_SQL.utils import Hypothesis as BaseHypothesis

from EditSQL.eval_scripts.evaluation import AGG_OPS, ORDER_OPS
NEW_WHERE_OPS = ('=','>','<','>=','<=','!=','like','not in','in','between', 'not like')
NEW_SQL_OPS = ('none','intersect', 'union', 'except')


class Hypothesis(BaseHypothesis):
    def __init__(self, dec_prefix, decoder_states, decoder_input):
        BaseHypothesis.__init__(self, dec_prefix)

        self.sql = [] # the sql tokens
        self.keyword = None # the current keyword in {select, order_by, having, where, group_by}

        self.nested_keywords = [] # a list of keywords
        # to extend feedback to later decisions
        self.avoid_items, self.confirmed_items = [], [] # a list of dicts of {semantic_tag: avoid/confirmed item list}

        # decoder info
        self.decoder_states = decoder_states
        self.decoder_input = decoder_input

    @staticmethod
    def print_hypotheses(hypotheses):
        for hyp in hypotheses:
            print("logprob: {}, sql: {}\ntag_seq: {}\ndec_seq: {}".format(
                hyp.logprob, hyp.sql, hyp.tag_seq, hyp.dec_seq))

def flatten_distribution(distribution_map, probabilities):
    """ Flattens a probability distribution given a map of "unique" values.
        All values in distribution_map with the same value should get the sum
        of the probabilities.

        Arguments:
            distribution_map (list of str): List of values to get the probability for.
            probabilities (np.ndarray): Probabilities corresponding to the values in
                distribution_map.

        Returns:
            list, np.ndarray of the same size where probabilities for duplicates
                in distribution_map are given the sum of the probabilities in probabilities.
    """
    assert len(distribution_map) == len(probabilities)
    if len(distribution_map) != len(set(distribution_map)):
        idx_first_dup = 0
        seen_set = set()
        for i, tok in enumerate(distribution_map):
            if tok in seen_set:
                idx_first_dup = i
                break
            seen_set.add(tok)
        new_dist_map = distribution_map[:idx_first_dup] + list(
            set(distribution_map) - set(distribution_map[:idx_first_dup]))
        assert len(new_dist_map) == len(set(new_dist_map))
        new_probs = np.array(
            probabilities[:idx_first_dup] \
            + [0. for _ in range(len(set(distribution_map)) \
                                 - idx_first_dup)])
        assert len(new_probs) == len(new_dist_map)

        for i, token_name in enumerate(
                distribution_map[idx_first_dup:]):
            if token_name not in new_dist_map:
                new_dist_map.append(token_name)

            new_index = new_dist_map.index(token_name)
            new_probs[new_index] += probabilities[i +
                                                  idx_first_dup]
        new_probs = new_probs.tolist()
    else:
        new_dist_map = distribution_map
        new_probs = probabilities

    assert len(new_dist_map) == len(new_probs)

    return new_dist_map, new_probs

class SQLPrediction(namedtuple('SQLPrediction',
                               ('predictions',
                                'sequence',
                                'probability'))):
    """Contains prediction for a sequence."""
    __slots__ = ()

    def __str__(self):
        return str(self.probability) + "\t" + " ".join(self.sequence)

class SequencePredictorWithSchema(torch.nn.Module):
    """ Predicts a sequence.

    Attributes:
        lstms (list of dy.RNNBuilder): The RNN used.
        token_predictor (TokenPredictor): Used to actually predict tokens.
    """
    def __init__(self,
                 params,
                 input_size,
                 output_embedder,
                 column_name_token_embedder,
                 token_predictor):
        super().__init__()

        self.lstms = torch_utils.create_multilayer_lstm_params(params.decoder_num_layers, input_size, params.decoder_state_size, "LSTM-d")
        self.token_predictor = token_predictor
        self.output_embedder = output_embedder
        self.column_name_token_embedder = column_name_token_embedder
        self.start_token_embedding = torch_utils.add_params((params.output_embedding_size,), "y-0")

        self.input_size = input_size
        self.params = params

    def _initialize_decoder_lstm(self, encoder_state):
        decoder_lstm_states = []
        for i, lstm in enumerate(self.lstms):
            encoder_layer_num = 0
            if len(encoder_state[0]) > 1:
                encoder_layer_num = i

            # check which one is h_0, which is c_0
            c_0 = encoder_state[0][encoder_layer_num].view(1,-1)
            h_0 = encoder_state[1][encoder_layer_num].view(1,-1)

            decoder_lstm_states.append((h_0, c_0))
        return decoder_lstm_states

    def get_output_token_embedding(self, output_token, input_schema, snippets):
        if self.params.use_snippets and snippet_handler.is_snippet(output_token):
            output_token_embedding = embedder.bow_snippets(output_token, snippets, self.output_embedder, input_schema)
        else:
            if input_schema:
                try:
                    assert self.output_embedder.in_vocabulary(output_token) or input_schema.in_vocabulary(output_token, surface_form=True)
                    if self.output_embedder.in_vocabulary(output_token):
                        output_token_embedding = self.output_embedder(output_token)
                    else:
                        output_token_embedding = input_schema.column_name_embedder(output_token, surface_form=True)
                except AssertionError:
                    print("\nWARNING: output_token '{}' is not found in vocabulary!".format(output_token))
                    output_token_embedding = self.output_embedder(output_token) # will turn to UNK
            else:
                output_token_embedding = self.output_embedder(output_token)
        return output_token_embedding

    def get_decoder_input(self, output_token_embedding, prediction):
        if self.params.use_schema_attention and self.params.use_query_attention:
            decoder_input = torch.cat([output_token_embedding, prediction.utterance_attention_results.vector, prediction.schema_attention_results.vector, prediction.query_attention_results.vector], dim=0)
        elif self.params.use_schema_attention:
            decoder_input = torch.cat([output_token_embedding, prediction.utterance_attention_results.vector, prediction.schema_attention_results.vector], dim=0)
        else:
            decoder_input = torch.cat([output_token_embedding, prediction.utterance_attention_results.vector], dim=0)
        return decoder_input

    def forward(self,
                final_encoder_state,
                encoder_states,
                schema_states,
                max_generation_length,
                snippets=None,
                gold_sequence=None,
                input_sequence=None,
                previous_queries=None,
                previous_query_states=None,
                input_schema=None,
                dropout_amount=0.):
        """ Generates a sequence. """
        index = 0

        context_vector_size = self.input_size - self.params.output_embedding_size

        # Decoder states: just the initialized decoder.
        # Current input to decoder: phi(start_token) ; zeros the size of the
        # context vector
        predictions = []
        sequence = []
        probability = 1.

        decoder_states = self._initialize_decoder_lstm(final_encoder_state)

        if self.start_token_embedding.is_cuda:
            decoder_input = torch.cat([self.start_token_embedding, torch.cuda.FloatTensor(context_vector_size).fill_(0)], dim=0)
        else:
            decoder_input = torch.cat([self.start_token_embedding, torch.zeros(context_vector_size)], dim=0)

        continue_generating = True
        while continue_generating:
            if len(sequence) == 0 or sequence[-1] != EOS_TOK:
                _, decoder_state, decoder_states = torch_utils.forward_one_multilayer(self.lstms, decoder_input, decoder_states, dropout_amount)
                prediction_input = PredictionInputWithSchema(decoder_state=decoder_state,
                                                             input_hidden_states=encoder_states,
                                                             schema_states=schema_states,
                                                             snippets=snippets,
                                                             input_sequence=input_sequence,
                                                             previous_queries=previous_queries,
                                                             previous_query_states=previous_query_states,
                                                             input_schema=input_schema)

                prediction = self.token_predictor(prediction_input, dropout_amount=dropout_amount)

                predictions.append(prediction)

                if gold_sequence:
                    output_token = gold_sequence[index]

                    output_token_embedding = self.get_output_token_embedding(output_token, input_schema, snippets)

                    decoder_input = self.get_decoder_input(output_token_embedding, prediction)

                    sequence.append(gold_sequence[index])

                    if index >= len(gold_sequence) - 1:
                        continue_generating = False
                else:
                    assert prediction.scores.dim() == 1
                    probabilities = F.softmax(prediction.scores, dim=0).cpu().data.numpy().tolist()

                    distribution_map = prediction.aligned_tokens
                    assert len(probabilities) == len(distribution_map)

                    if self.params.use_previous_query and self.params.use_copy_switch and len(previous_queries) > 0:
                        assert prediction.query_scores.dim() == 1
                        query_token_probabilities = F.softmax(prediction.query_scores, dim=0).cpu().data.numpy().tolist()

                        query_token_distribution_map = prediction.query_tokens

                        assert len(query_token_probabilities) == len(query_token_distribution_map)

                        copy_switch = prediction.copy_switch.cpu().data.numpy()

                        # Merge the two
                        probabilities = ((np.array(probabilities) * (1 - copy_switch)).tolist() + 
                                         (np.array(query_token_probabilities) * copy_switch).tolist()
                                         )
                        distribution_map =  distribution_map + query_token_distribution_map
                        assert len(probabilities) == len(distribution_map)

                    # Get a new probabilities and distribution_map consolidating duplicates
                    distribution_map, probabilities = flatten_distribution(distribution_map, probabilities)

                    # Modify the probability distribution so that the UNK token can never be produced
                    probabilities[distribution_map.index(UNK_TOK)] = 0.
                    argmax_index = int(np.argmax(probabilities))

                    argmax_token = distribution_map[argmax_index]
                    sequence.append(argmax_token)

                    output_token_embedding = self.get_output_token_embedding(argmax_token, input_schema, snippets)

                    decoder_input = self.get_decoder_input(output_token_embedding, prediction)

                    probability *= probabilities[argmax_index]

                    continue_generating = False
                    if index < max_generation_length and argmax_token != EOS_TOK:
                        continue_generating = True

            index += 1

        return SQLPrediction(predictions,
                             sequence,
                             probability)

    def update_tag_seq(self, keyword, token_idx, token, prob, tag_seq, sql, dec_idx):
        if token in {'max', 'min', 'count', 'sum', 'avg'}:
            if keyword == "select":
                tag = SELECT_AGG_v2
            elif keyword == "order_by":
                tag = ORDER_AGG_v2
            elif keyword == "having":
                tag = HAV_AGG_v2
            else:
                raise Exception("Agg {} is invalid with keyword {}!".format(token, keyword))
            agg = (token, AGG_OPS.index(token))
            su = (tag, None, agg, False, [prob], dec_idx)

            tag_seq.append(su)

        elif token == 'distinct':
            assert keyword in {"select", "order_by", "having"}
            if sql[-2] != '(': # only consider cases like "count ( distinct c1 )"
                return tag_seq

            assert tag_seq[-1][0] in {SELECT_AGG_v2, ORDER_AGG_v2, HAV_AGG_v2} and \
                    tag_seq[-1][1] is None
            # revise unit
            su = tag_seq[-1]
            su = (su[0], None, su[2], True, su[4] + [prob], su[5])
            tag_seq[-1] = su

        elif token_idx >= len(self.token_predictor.vocabulary):  # column
            if "*" in token:
                if "." in token:
                    tab_name, col_name = token.split('.')
                else:
                    tab_name = None
                    col_name = "*"
                col_idx = token_idx # 0; revised 01/30
            else:
                tab_name, col_name = token.split('.')
                col_idx = token_idx #- len(self.token_predictor.vocabulary); revised 01/30

            col = (tab_name, col_name, col_idx)
            if keyword in {"select", "order_by", "having"}:
                if len(tag_seq) and tag_seq[-1][0] in {SELECT_AGG_v2, ORDER_AGG_v2, HAV_AGG_v2} and \
                        tag_seq[-1][1] is None:
                    su = tag_seq[-1]
                    su = (su[0], col, su[2], su[3], su[4] + [prob], su[5])
                    tag_seq[-1] = su
                else:
                    if keyword == "select":
                        tag = SELECT_AGG_v2
                    elif keyword == "order_by":
                        tag = ORDER_AGG_v2
                    else:
                        assert keyword == "having"
                        tag = HAV_AGG_v2
                    su = (tag, col, ("none_agg", AGG_OPS.index("none")), False, [prob], dec_idx)
                    tag_seq.append(su)
            else:
                if keyword == "where":
                    tag = WHERE_COL
                elif keyword == "group_by":
                    tag = GROUP_COL
                else:
                    raise Exception("Col {} is invalid with keyword {}!".format(token, keyword))
                su = (tag, col, prob, dec_idx)
                tag_seq.append(su)

        elif token in list(NEW_WHERE_OPS) + ['not']: # ('=','>','<','>=','<=','!=','like','not in','in','between', 'not like')
            if (token == "=" and sql[-2] in {'<', '>'}) or \
                    (token == "in" and sql[-2] == "not") or \
                    (token == "like" and sql[-2] == "not"):
                assert tag_seq[-1][0] in {WHERE_OP, HAV_OP_v2}
                if token == "=":
                    op_name = "".join(sql[-2:])
                else:
                    op_name = " ".join(sql[-2:])
                op = (op_name, NEW_WHERE_OPS.index(op_name))
                su = tag_seq[-1]
                avg_prob = np.exp((np.log(prob) + np.log(su[3])) / 2)
                tag_seq[-1] = (su[0], su[1], op, avg_prob, su[4])
            else:
                if keyword == "where":
                    tag = WHERE_OP
                    assert tag_seq[-1][0] == WHERE_COL
                    col_agg = (tag_seq[-1][1],)
                else:
                    assert keyword == "having"
                    tag = HAV_OP_v2
                    assert tag_seq[-1][0] == HAV_AGG_v2
                    col_agg = (tag_seq[-1][1], tag_seq[-1][2], tag_seq[-1][3])

                if token == 'not':
                    op = None
                else:
                    op = (token, NEW_WHERE_OPS.index(token))
                su = (tag, col_agg, op, prob, dec_idx)
                tag_seq.append(su)

        elif token == "value":
            if keyword == "where":
                op_tag = WHERE_OP
                tag = WHERE_ROOT_TERM
            else:
                assert keyword == "having"
                op_tag = HAV_OP_v2
                tag = HAV_ROOT_TERM_v2
            op_pos = helper_find_closest_bw(tag_seq, len(tag_seq) - 1, tgt_name=op_tag)
            assert op_pos != -1
            if tag_seq[op_pos][2][0] == "between" and " ".join(sql[-4:]) == "between value and value":
                return tag_seq
            su = (tag, tag_seq[op_pos][1], tag_seq[op_pos][2], 'terminal', prob, dec_idx)
            tag_seq.append(su)

        elif token == "(":
            if sql[-2] in {'max', 'min', 'count', 'sum', 'avg'}:
                assert tag_seq[-1][0] in {SELECT_AGG_v2, ORDER_AGG_v2, HAV_AGG_v2} # "count ( c1 )"
                su = tag_seq[-1]
                su = (su[0], su[1], su[2], su[3], su[4] + [prob], su[5])
                tag_seq[-1] = su
            elif sql[-2] in {'distinct', 'select', 'order_by', 'having'}: # e.g., "select distinct ( c1 )"
                if keyword == "select":
                    tag = SELECT_AGG_v2
                elif keyword == "order_by":
                    tag = ORDER_AGG_v2
                else:
                    assert keyword == "having"
                    tag = HAV_AGG_v2
                su = (tag, None, ("none_agg", AGG_OPS.index("none")), False, [prob], dec_idx)
                tag_seq.append(su)
            elif len(tag_seq) > 0:
                if tag_seq[-1][0] == WHERE_OP:
                    op = tag_seq[-1][2]
                    col = tag_seq[-1][1][0]
                    su = (WHERE_ROOT_TERM, (col,), op, 'root', prob, dec_idx)
                    tag_seq.append(su)
                elif tag_seq[-1][0] == HAV_OP_v2:
                    op = tag_seq[-1][2]
                    col_agg = tag_seq[-1][1]
                    su = (HAV_ROOT_TERM_v2, col_agg, op, 'root', prob, dec_idx)
                    tag_seq.append(su)
                else:
                    print("WARNING in tag_seq generation from token '(', keyword {}\n"
                          "Current_tag_seq: {}\nCurrent sql: {}".format(
                        keyword, tag_seq, sql))
            else:
                print("WARNING in tag_seq generation from token '(', keyword {}\n"
                      "Current_tag_seq: {}\nCurrent sql: {}".format(
                    keyword, tag_seq, sql))

        elif token == ")":
            if tag_seq[-1][0] in {SELECT_AGG_v2, ORDER_AGG_v2, HAV_AGG_v2}:
                if sql[-2] not in self.token_predictor.vocabulary.tokens and sql[-3] in {'(', 'distinct'}:
                    su = tag_seq[-1]
                    su = (su[0], su[1], su[2], su[3], su[4] + [prob], su[5])
                    tag_seq[-1] = su
                    return tag_seq

            if keyword in {'group_by', 'order_by'}:
                _start_idx = len(sql) - 1
                while sql[_start_idx] != keyword:
                    _start_idx -= 1
                if sql[_start_idx + 1] == '(' and sql[-2] not in self.token_predictor.vocabulary.tokens:
                    return tag_seq # "group_by ( c1 )", "order_by (c1 - c2)"

            su = (OUTSIDE, END_NESTED, prob, dec_idx)
            tag_seq.append(su)

        elif token in ORDER_OPS:
            assert tag_seq[-1][0] == ORDER_AGG_v2
            (col, agg, bool_distinct) = tag_seq[-1][1:4]
            su = (ORDER_DESC_ASC, (col, agg, bool_distinct), token, prob, dec_idx)
            tag_seq.append(su)

        elif token == "limit_value":
            if tag_seq[-1][0] == ORDER_AGG_v2:
                order_by_agg = tag_seq[-1]
            else:
                assert tag_seq[-2][0] == ORDER_AGG_v2
                order_by_agg = tag_seq[-2]
            col, agg, bool_distinct = order_by_agg[1:4]
            su = (ORDER_LIMIT, (col, agg, bool_distinct), token, prob, dec_idx)
            tag_seq.append(su)

        elif token in NEW_SQL_OPS:
            su = (IUEN_v2, (token, NEW_SQL_OPS.index(token)), prob, dec_idx)
            tag_seq.append(su)

        return tag_seq

    def beam_search(self, final_encoder_state, encoder_states, schema_states, max_generation_length,
                    snippets=None, input_sequence=None, previous_queries=None, previous_query_states=None,
                    input_schema=None, dropout_amount=0., stop_step=None, beam_size=1, dec_prefix=None,
                    avoid_items=None, confirmed_items=None, tag_constraint=True, reset_confirmed_prob=True,
                    bool_verbal=False):

        # self.token_predictor.vocabulary:
        #  ['_UNK', '_EOS', 'select', 'value', 'where', '=', ')',
        # '(', ',', 'count', 'group_by', 'order_by', 'distinct', 'and',
        # 'desc', 'limit_value', '>', 'avg', 'having', 'max', 'in', '<',
        # 'sum', 'intersect', 'not', 'min', 'asc', 'or', 'except', 'like',
        # '!=', 'union', 'between', '-', '+', '/']

        # semantic tags for which user feedback needs to extend from dec_idx to its follow-up
        # confirmed_items: WHERE_COL, HAV_AGG
        # avoid_items: SELECT/HAV/ORDER_AGG, WHERE/GROUP_COL

        cat2valid_indices = {'agg': [self.token_predictor.vocabulary.token_to_id(tok) for tok in
                                     {'max', 'min', 'count', 'sum', 'avg'}],
                             'op': [self.token_predictor.vocabulary.token_to_id(tok) for tok in
                                    {'not', 'between', '=', '>', '<', '!=', 'in', 'like'}],
                             'iuen': [self.token_predictor.vocabulary.token_to_id(tok) for tok in
                                     {'intersect', 'union', 'except'}],
                             'order': [self.token_predictor.vocabulary.token_to_id(tok) for tok in {'desc', 'asc'}]}

        def get_valid_indices(token_idx, token, distribution_map, keyword):
            semantic_label = None
            if token in {'max', 'min', 'count', 'sum', 'avg'}:
                valid_indices = cat2valid_indices['agg']
            elif token in {'not', 'between', '=', '>', '<', '!=', 'in', 'like'}:
                valid_indices = cat2valid_indices['op']
            elif token in {'intersect', 'union', 'except'}:
                valid_indices = cat2valid_indices['iuen']
            elif token in {'desc', 'asc'}:
                valid_indices = cat2valid_indices['order']
            elif token_idx >= len(self.token_predictor.vocabulary): #column
                valid_indices = list(range(len(self.token_predictor.vocabulary), len(distribution_map)))
                if keyword == "where":
                    semantic_label = WHERE_COL
                elif keyword == "group_by":
                    semantic_label = GROUP_COL
            elif token in {'select', 'where', 'group_by', 'having', 'order_by', '_EOS'}:
                valid_indices = [self.token_predictor.vocabulary.token_to_id(tok) for tok in
                                 ['select', 'where', 'group_by', 'having', 'order_by',
                                  'intersect', 'union', 'except', '_EOS']]
            else:
                valid_indices = range(len(distribution_map))
                # raise Exception("Unexpected syntactic category for token {}!".format(token))

            return valid_indices, semantic_label

        def agg_col_beam_search(hypothesis, avoid_decisions=None, confirmed_decision=None):
            # load dec_prefix
            bool_confirmed = False
            selections = None
            if len(hypothesis.dec_prefix):
                selections = hypothesis.dec_prefix.pop()
                assert isinstance(selections, list)
                selections = selections[::-1]
            elif confirmed_decision is not None:
                selections = confirmed_decision[::-1]
                assert isinstance(selections, list)
                bool_confirmed = True

            actual_beam_size = beam_size
            if avoid_decisions is not None:
                actual_beam_size = beam_size + len(avoid_decisions)

            cur_hypotheses = [hypothesis]
            completed_hypotheses = []

            for intermediate_step in range(5):
                # at most 5 steps, e.g.,
                # [col1]
                # [(, col1, )]
                # [count, (, col1, )]
                # ['count', '(', 'distinct', 'col1', ')']
                new_hypotheses = []

                for hyp in cur_hypotheses:
                    _, decoder_state, new_decoder_states = torch_utils.forward_one_multilayer(
                        self.lstms, hyp.decoder_input, hyp.decoder_states, dropout_amount)
                    hyp.decoder_states = new_decoder_states  # update hyp decoder_states

                    prediction_input = PredictionInputWithSchema(decoder_state=decoder_state,
                                                                 input_hidden_states=encoder_states,
                                                                 schema_states=schema_states,
                                                                 snippets=snippets,
                                                                 input_sequence=input_sequence,
                                                                 previous_queries=previous_queries,
                                                                 previous_query_states=previous_query_states,
                                                                 input_schema=input_schema)

                    prediction = self.token_predictor(prediction_input, dropout_amount=dropout_amount)

                    assert prediction.scores.dim() == 1
                    probabilities = F.softmax(prediction.scores, dim=0).cpu().data.numpy().tolist()

                    distribution_map = prediction.aligned_tokens
                    assert len(probabilities) == len(distribution_map)

                    # Get a new probabilities and distribution_map consolidating duplicates
                    distribution_map, probabilities = flatten_distribution(distribution_map, probabilities)
                    probabilities[distribution_map.index(UNK_TOK)] = 0.

                    if selections is not None:
                        if len(selections) == 0: # end of prefix for this component
                            if bool_confirmed and reset_confirmed_prob:
                                temp_su = list(hyp.tag_seq[-1])  # set prob to 1.0 for confirmed decisions
                                temp_su[-2] = 1.0
                                hyp.tag_seq[-1] = tuple(temp_su)
                            else:
                                su = hyp.tag_seq[-1]
                                avg_prob = np.exp(sum([np.log(prob_i) for prob_i in su[4]]) / len(su[4]))
                                su = (su[0], su[1], su[2], su[3], avg_prob, su[5])
                                hyp.tag_seq[-1] = su
                            completed_hypotheses.append(hyp)
                            return completed_hypotheses

                        candidates = [selections.pop()]
                    else:
                        if tag_constraint:
                            if intermediate_step == 0:
                                valid_indices = cat2valid_indices['agg'] + \
                                                [self.token_predictor.vocabulary.token_to_id('(')] + \
                                                list(range(len(self.token_predictor.vocabulary), len(distribution_map)))
                            elif intermediate_step == 1:
                                if hyp.sql[-1] == '(':
                                    valid_indices = list(range(len(self.token_predictor.vocabulary), len(distribution_map)))
                                else:
                                    valid_indices = [self.token_predictor.vocabulary.token_to_id('(')]
                            elif intermediate_step == 2:
                                if hyp.sql[-1] == '(':
                                    valid_indices = [self.token_predictor.vocabulary.token_to_id('distinct')] + \
                                                    list(range(len(self.token_predictor.vocabulary), len(distribution_map)))
                                else:
                                    valid_indices = [self.token_predictor.vocabulary.token_to_id(')')]
                            elif intermediate_step == 3:
                                if hyp.sql[-1] == 'distinct':
                                    valid_indices = list(range(len(self.token_predictor.vocabulary), len(distribution_map)))
                                else:
                                    valid_indices = [self.token_predictor.vocabulary.token_to_id(')')]
                            else:
                                valid_indices = [self.token_predictor.vocabulary.token_to_id(')')]

                            candidates = []
                            for idx in np.argsort(probabilities)[::-1]:
                                if idx in valid_indices:
                                    candidates.append(idx)
                                if len(candidates) == actual_beam_size or len(candidates) == len(valid_indices):
                                    break
                        else:
                            candidates = np.argsort(probabilities)[::-1][:actual_beam_size]

                    for idx in candidates:
                        if len(candidates) == 1:
                            step_hyp = hyp
                        else:
                            step_hyp = hyp.copy()

                        token = distribution_map[idx]
                        if intermediate_step == 0:
                            step_hyp.dec_seq.append([idx])
                        else:
                            step_hyp.dec_seq[-1].append(idx)
                        step_hyp.sql.append(token)
                        step_hyp.add_logprob(np.log(probabilities[idx]))

                        # if token not in {'(', ')'}:
                        step_hyp.tag_seq = self.update_tag_seq(
                            step_hyp.keyword, idx, token, probabilities[idx], step_hyp.tag_seq,
                            step_hyp.sql, len(step_hyp.dec_seq) - 1)

                        # update hyp.decoder_input
                        output_token_embedding = self.get_output_token_embedding(token, input_schema, snippets)
                        step_hyp.decoder_input = self.get_decoder_input(output_token_embedding, prediction)

                        if (idx >= len(self.token_predictor.vocabulary) and intermediate_step == 0) or \
                            token == ')':
                            step_hyp.dec_seq_idx += 1
                            completed_hypotheses.append(step_hyp)
                        else:
                            new_hypotheses.append(step_hyp)

                if len(new_hypotheses) == 0:
                    break

                cur_hypotheses = Hypothesis.sort_hypotheses(new_hypotheses, actual_beam_size, 0.0)

            # top K from completed hypotheses
            if selections is None and avoid_decisions is not None:
                completed_hypotheses = [hyp for hyp in completed_hypotheses if hyp.dec_seq[-1] not in avoid_decisions]
            _completed_hypotheses = Hypothesis.sort_hypotheses(completed_hypotheses, beam_size, 0.0)

            completed_hypotheses = []
            for _hyp in _completed_hypotheses:
                assert _hyp.tag_seq[-1][0] in {SELECT_AGG_v2, ORDER_AGG_v2, HAV_AGG_v2}
                su = _hyp.tag_seq[-1]

                avg_prob = np.exp(sum([np.log(prob_i) for prob_i in su[4]]) / len(su[4]))
                if selections is not None:
                    assert len(selections) == 0 and len(_completed_hypotheses) == 1, \
                        "Exception in agg_col_beam_search: len(_completed_hypotheses)={}.".format(
                            len(_completed_hypotheses))
                    if bool_confirmed and reset_confirmed_prob:
                        avg_prob = 1.0

                su = (su[0], su[1], su[2], su[3], avg_prob, su[5])
                _hyp.tag_seq[-1] = su
                completed_hypotheses.append(_hyp)

            return completed_hypotheses

        def op_greedy(hypothesis):
            assert beam_size == 1
            if len(hypothesis.dec_prefix):
                decisions = hypothesis.dec_prefix.pop()
                dec_steps = len(decisions)
            else:
                decisions = None
                dec_steps = 2 # at most 2 steps

            # ('=','>','<','>=','<=','!=','like','not in','in','between')
            for intermediate_step in range(dec_steps):
                # preview the next decoding
                _, decoder_state, new_decoder_states = torch_utils.forward_one_multilayer(
                    self.lstms, hypothesis.decoder_input, hypothesis.decoder_states, dropout_amount)

                prediction_input = PredictionInputWithSchema(decoder_state=decoder_state,
                                                             input_hidden_states=encoder_states,
                                                             schema_states=schema_states,
                                                             snippets=snippets,
                                                             input_sequence=input_sequence,
                                                             previous_queries=previous_queries,
                                                             previous_query_states=previous_query_states,
                                                             input_schema=input_schema)

                prediction = self.token_predictor(prediction_input, dropout_amount=dropout_amount)

                assert prediction.scores.dim() == 1
                probabilities = F.softmax(prediction.scores, dim=0).cpu().data.numpy().tolist()

                distribution_map = prediction.aligned_tokens
                assert len(probabilities) == len(distribution_map)

                # Get a new probabilities and distribution_map consolidating duplicates
                distribution_map, probabilities = flatten_distribution(distribution_map, probabilities)
                probabilities[distribution_map.index(UNK_TOK)] = 0.

                if decisions is None:
                    tok_idx = np.argmax(probabilities)
                else:
                    tok_idx = decisions[intermediate_step]
                token = distribution_map[tok_idx]

                if intermediate_step == 0:
                    hypothesis.sql.append(token)
                    hypothesis.add_logprob(np.log(probabilities[tok_idx]))
                    hypothesis.dec_seq.append([tok_idx])

                    hypothesis.tag_seq = self.update_tag_seq(
                        hypothesis.keyword, tok_idx, token, probabilities[tok_idx], hypothesis.tag_seq,
                        hypothesis.sql, len(hypothesis.dec_seq) - 1)

                    # update hyp.decoder_input
                    output_token_embedding = self.get_output_token_embedding(token, input_schema, snippets)
                    hypothesis.decoder_input = self.get_decoder_input(output_token_embedding, prediction)
                    hypothesis.decoder_states = new_decoder_states  # update hyp decoder_states

                    if token not in {'>', '<', 'not'} or dec_steps == 1:
                        hypothesis.dec_seq_idx += 1
                        return hypothesis
                else:
                    if token in {'=', 'like', 'in'}: # >=, <=, not like, not in
                        hypothesis.sql.append(token)
                        hypothesis.add_logprob(np.log(probabilities[tok_idx]))
                        hypothesis.dec_seq[-1].append(tok_idx)

                        hypothesis.tag_seq = self.update_tag_seq(
                            hypothesis.keyword, tok_idx, token, probabilities[tok_idx], hypothesis.tag_seq,
                            hypothesis.sql, len(hypothesis.dec_seq) - 1)

                        # update hyp.decoder_input
                        output_token_embedding = self.get_output_token_embedding(token, input_schema, snippets)
                        hypothesis.decoder_input = self.get_decoder_input(output_token_embedding, prediction)
                        hypothesis.decoder_states = new_decoder_states  # update hyp decoder_states
                        hypothesis.dec_seq_idx += 1
                        return hypothesis
                    else:
                        # the next greedy token is not part of the operator
                        hypothesis.dec_seq_idx += 1
                        return hypothesis

        def op_beam_search(hypothesis, avoid_decisions=None):
            # ('=','>','<','>=','<=','!=','like','not in','in','between')

            # load dec_prefix
            if len(hypothesis.dec_prefix):
                candidates = [hypothesis.dec_prefix.pop()]
            else:
                candidates = []
                for op_tokens in [['='],['>'],['<'],['!='],['like'],['in'],['between'],
                                  ['>', '='], ['<', '='], ['not', 'in'], ['not', 'like']]:
                    dec_indices = [self.token_predictor.vocabulary.token_to_id(tok) for tok in op_tokens]
                    if avoid_decisions is None or dec_indices not in avoid_decisions:
                        candidates.append(dec_indices)

            completed_hypotheses = []

            for cand in candidates:
                if len(candidates) == 1:
                    new_hypothesis = hypothesis
                else:
                    new_hypothesis = hypothesis.copy()

                for cand_idx, tok_idx in enumerate(cand):
                    _, decoder_state, new_decoder_states = torch_utils.forward_one_multilayer(
                        self.lstms, new_hypothesis.decoder_input, new_hypothesis.decoder_states, dropout_amount)
                    new_hypothesis.decoder_states = new_decoder_states  # update hyp decoder_states

                    prediction_input = PredictionInputWithSchema(decoder_state=decoder_state,
                                                                 input_hidden_states=encoder_states,
                                                                 schema_states=schema_states,
                                                                 snippets=snippets,
                                                                 input_sequence=input_sequence,
                                                                 previous_queries=previous_queries,
                                                                 previous_query_states=previous_query_states,
                                                                 input_schema=input_schema)

                    prediction = self.token_predictor(prediction_input, dropout_amount=dropout_amount)

                    assert prediction.scores.dim() == 1
                    probabilities = F.softmax(prediction.scores, dim=0).cpu().data.numpy().tolist()

                    distribution_map = prediction.aligned_tokens
                    assert len(probabilities) == len(distribution_map)

                    # Get a new probabilities and distribution_map consolidating duplicates
                    distribution_map, probabilities = flatten_distribution(distribution_map, probabilities)
                    probabilities[distribution_map.index(UNK_TOK)] = 0.

                    token = distribution_map[tok_idx]
                    new_hypothesis.sql.append(token)
                    new_hypothesis.add_logprob(np.log(probabilities[tok_idx]))
                    if cand_idx == 0:
                        new_hypothesis.dec_seq.append([tok_idx])
                    else:
                        new_hypothesis.dec_seq[-1].append(tok_idx)
                    new_hypothesis.tag_seq = self.update_tag_seq(
                        new_hypothesis.keyword, tok_idx, token, probabilities[tok_idx], new_hypothesis.tag_seq,
                        new_hypothesis.sql, len(new_hypothesis.dec_seq) - 1)

                    # update hyp.decoder_input
                    output_token_embedding = self.get_output_token_embedding(token, input_schema, snippets)
                    new_hypothesis.decoder_input = self.get_decoder_input(output_token_embedding, prediction)

                new_hypothesis.dec_seq_idx += 1
                completed_hypotheses.append(new_hypothesis)

            # top K from completed hypotheses
            completed_hypotheses = Hypothesis.sort_hypotheses(completed_hypotheses, beam_size, 0.0)

            return completed_hypotheses

        if dec_prefix is None:
            dec_prefix = []
        else:
            dec_prefix = dec_prefix[::-1]

        context_vector_size = self.input_size - self.params.output_embedding_size

        if self.start_token_embedding.is_cuda:
            decoder_input = torch.cat([self.start_token_embedding, torch.cuda.FloatTensor(context_vector_size).fill_(0)], dim=0)
        else:
            decoder_input = torch.cat([self.start_token_embedding, torch.zeros(context_vector_size)], dim=0)

        hypotheses = [Hypothesis(dec_prefix, self._initialize_decoder_lstm(final_encoder_state), decoder_input)]
        completed_hypotheses = []

        while True:
            new_hypotheses = []

            for hyp in hypotheses:
                # decoding for one step
                _, decoder_state, new_decoder_states = torch_utils.forward_one_multilayer(
                    self.lstms, hyp.decoder_input, hyp.decoder_states, dropout_amount)

                prediction_input = PredictionInputWithSchema(decoder_state=decoder_state,
                                                             input_hidden_states=encoder_states,
                                                             schema_states=schema_states,
                                                             snippets=snippets,
                                                             input_sequence=input_sequence,
                                                             previous_queries=previous_queries,
                                                             previous_query_states=previous_query_states,
                                                             input_schema=input_schema)

                prediction = self.token_predictor(prediction_input, dropout_amount=dropout_amount)

                assert prediction.scores.dim() == 1
                probabilities = F.softmax(prediction.scores, dim=0).cpu().data.numpy().tolist()

                distribution_map = prediction.aligned_tokens
                assert len(probabilities) == len(distribution_map)

                # Get a new probabilities and distribution_map consolidating duplicates
                distribution_map, probabilities = flatten_distribution(distribution_map, probabilities)
                probabilities[distribution_map.index(UNK_TOK)] = 0.

                if hyp.keyword in {'select', 'order_by', 'having'} and len(hyp.sql) > 0 and \
                        hyp.sql[-1] not in {'between', '=', '>', '<', '!=', 'in', 'like'} and \
                        (np.argmax(probabilities) >= len(self.token_predictor.vocabulary) or
                         distribution_map[np.argmax(probabilities)] in {'max', 'min', 'count', 'sum', 'avg', '('} or
                         (len(hyp.dec_prefix) and isinstance(hyp.dec_prefix[-1], list) and
                          (np.array(hyp.dec_prefix[-1]) >= len(self.token_predictor.vocabulary)).any())):

                    # SELECT/ORDER/HAV_AGG
                    # [col1]
                    # [(, col1, )]
                    # [count, (, col1, )]
                    # ['count', '(', 'distinct', 'col1', ')']

                    avoid_decisions, confirmed_decision = None, None
                    # check confirmed items
                    if confirmed_items is not None and hyp.dec_seq_idx in confirmed_items:
                        confirmed_decision = confirmed_items[hyp.dec_seq_idx][0]
                        if hyp.keyword == 'having' and len(confirmed_items[hyp.dec_seq_idx]) > 1:
                            hyp.confirmed_items[-1][HAV_AGG_v2].append(confirmed_items[hyp.dec_seq_idx][1:])
                    else:
                        # check confirmed items from semantic_tag
                        if hyp.keyword == 'having' and HAV_AGG_v2 in hyp.confirmed_items[-1]:
                            confirmed_decision = hyp.confirmed_items[-1][HAV_AGG_v2][0]
                            hyp.confirmed_items[-1][HAV_AGG_v2] = hyp.confirmed_items[-1][HAV_AGG_v2][1:]
                            if len(hyp.confirmed_items[-1][HAV_AGG_v2]) == 0:
                                hyp.confirmed_items[-1].pop(HAV_AGG_v2)

                    if confirmed_decision is None:
                        if hyp.keyword == 'select':
                            sem_tag = SELECT_AGG_v2
                        elif hyp.keyword == 'order_by':
                            sem_tag = ORDER_AGG_v2
                        else:
                            sem_tag = HAV_AGG_v2

                        # check avoid items
                        if avoid_items is not None and hyp.dec_seq_idx in avoid_items:
                            avoid_decisions = avoid_items[hyp.dec_seq_idx]
                            if sem_tag in hyp.avoid_items[-1]:
                                for idx_list in hyp.avoid_items[-1][sem_tag]:
                                    if idx_list not in avoid_decisions:
                                        avoid_decisions.append(idx_list)

                            # update hyp.avoid_items
                            hyp.avoid_items[-1][sem_tag].extend(avoid_items[hyp.dec_seq_idx])
                        else:
                            if sem_tag in hyp.avoid_items[-1]:
                                avoid_decisions = hyp.avoid_items[-1][sem_tag]

                    cur_new_hypotheses = agg_col_beam_search(hyp, avoid_decisions=avoid_decisions,
                                                             confirmed_decision=confirmed_decision)

                    # new_hypotheses.extend(cur_new_hypotheses)
                    for step_hyp in cur_new_hypotheses:
                        if EOS_TOK in step_hyp.sql or len(step_hyp.sql) >= max_generation_length:
                            completed_hypotheses.append(step_hyp)
                        else:
                            new_hypotheses.append(step_hyp)

                elif hyp.keyword in {'where', 'having'} and hyp.tag_seq[-1][0] in {WHERE_COL, HAV_AGG_v2}:
                    avoid_decisions, confirmed_decision = None, None
                    if avoid_items is not None and hyp.dec_seq_idx in avoid_items:
                        avoid_decisions = avoid_items[hyp.dec_seq_idx]
                    assert confirmed_items is None or hyp.dec_seq_idx not in confirmed_items

                    if beam_size > 1:
                        cur_new_hypotheses = op_beam_search(hyp, avoid_decisions=avoid_decisions)
                    else:
                        cur_new_hypotheses = [op_greedy(hyp)]

                    # new_hypotheses.extend(cur_new_hypotheses)
                    for step_hyp in cur_new_hypotheses:
                        if EOS_TOK in step_hyp.sql or len(step_hyp.sql) >= max_generation_length:
                            completed_hypotheses.append(step_hyp)
                        else:
                            new_hypotheses.append(step_hyp)

                else:
                    bool_confirmed = False
                    if len(hyp.dec_prefix):
                        candidates = [hyp.dec_prefix.pop()]
                    elif confirmed_items is not None and hyp.dec_seq_idx in confirmed_items:
                        candidates = [confirmed_items[hyp.dec_seq_idx][0]]
                        bool_confirmed = True
                    else:
                        if tag_constraint:
                            # grammar constraint
                            if len(hyp.tag_seq) and hyp.tag_seq[-1][0] in {WHERE_OP, HAV_OP_v2}:
                                valid_indices = [self.token_predictor.vocabulary.token_to_id('('),
                                                 self.token_predictor.vocabulary.token_to_id('value')]

                                probs = [probabilities[_idx] for _idx in valid_indices]
                                if probs[0] > probs[1]:
                                    candidates = valid_indices
                                else:
                                    candidates = valid_indices[::-1]
                                candidates = candidates[:beam_size]

                            else:
                                if avoid_items is not None and hyp.dec_seq_idx in avoid_items:
                                    last_negated = avoid_items[hyp.dec_seq_idx][-1] # last negation decides the tag
                                    last_negated_token = distribution_map[last_negated]
                                    valid_indices, argmax_label = get_valid_indices(
                                        last_negated, last_negated_token, distribution_map, hyp.keyword)
                                else:
                                    argmax_cand = np.argmax(probabilities)
                                    argmax_cand_token = distribution_map[argmax_cand]
                                    valid_indices, argmax_label = get_valid_indices(
                                        argmax_cand, argmax_cand_token, distribution_map, hyp.keyword)

                                if argmax_label == WHERE_COL and argmax_label in hyp.confirmed_items[-1]:
                                    candidates = [hyp.confirmed_items[-1][argmax_label][0]]
                                    bool_confirmed = True
                                    hyp.confirmed_items[-1][argmax_label] = hyp.confirmed_items[-1][argmax_label][1:]
                                    if len(hyp.confirmed_items[-1][argmax_label]) == 0:
                                        hyp.confirmed_items[-1].pop(argmax_label)
                                else:
                                    # check avoid_items
                                    if avoid_items is not None and hyp.dec_seq_idx in avoid_items:
                                        valid_indices = [cand for cand in valid_indices
                                                         if cand not in avoid_items[hyp.dec_seq_idx]]
                                    # check avoid_items from semantic_tag
                                    if argmax_label in {WHERE_COL, GROUP_COL} and argmax_label in hyp.avoid_items[-1]:
                                        valid_indices = [cand for cand in valid_indices
                                                         if cand not in hyp.avoid_items[-1][argmax_label]]

                                    candidates = []
                                    for idx in np.argsort(probabilities)[::-1]:
                                        if idx in valid_indices:
                                            candidates.append(idx)
                                        if len(candidates) == beam_size or len(candidates) == len(valid_indices):
                                            break
                        else:
                            if avoid_items is not None and hyp.dec_seq_idx in avoid_items:
                                candidates = []
                                for idx in np.argsort(probabilities)[::-1]:
                                    if idx not in avoid_items[hyp.dec_seq_idx]:
                                        candidates.append(idx)
                                    if len(candidates) == beam_size:
                                        break
                            else:
                                candidates = np.argsort(probabilities)[-beam_size:][::-1]

                    for idx in candidates:
                        if len(candidates) == 1:
                            step_hyp = hyp
                        else:
                            step_hyp = hyp.copy()

                        token = distribution_map[idx]
                        step_hyp.dec_seq.append(idx)
                        step_hyp.sql.append(token)
                        step_hyp.add_logprob(np.log(probabilities[idx]))
                        step_hyp.tag_seq = self.update_tag_seq( # have to put after updating step_hyp.sql
                            step_hyp.keyword, idx, token, probabilities[idx], step_hyp.tag_seq,
                            step_hyp.sql, len(step_hyp.dec_seq) - 1)
                        if bool_confirmed and reset_confirmed_prob:
                            temp_su = list(step_hyp.tag_seq[-1]) # set prob to 1.0 for confirmed decisions
                            temp_su[-2] = 1.0
                            step_hyp.tag_seq[-1] = tuple(temp_su)

                        # update decoder meta
                        step_hyp.decoder_states = new_decoder_states
                        output_token_embedding = self.get_output_token_embedding(token, input_schema, snippets)
                        step_hyp.decoder_input = self.get_decoder_input(output_token_embedding, prediction)

                        # broadcast avoid/confirmed items
                        if confirmed_items is not None and step_hyp.dec_seq_idx in confirmed_items and \
                            len(confirmed_items[step_hyp.dec_seq_idx]) > 1:
                            if step_hyp.tag_seq[-1][0] == WHERE_COL:
                                assert step_hyp.keyword == step_hyp.nested_keywords[-1] == 'where'
                                step_hyp.confirmed_items[-1][WHERE_COL].extend(
                                    confirmed_items[step_hyp.dec_seq_idx][1:])

                        if avoid_items is not None and step_hyp.dec_seq_idx in avoid_items and \
                            step_hyp.tag_seq[-1][0] in {WHERE_COL, GROUP_COL}:
                            step_hyp.avoid_items[-1][step_hyp.tag_seq[-1][0]].extend(
                                avoid_items[step_hyp.dec_seq_idx])

                        # update dec_seq position
                        step_hyp.dec_seq_idx += 1

                        if token in {'select', 'order_by', 'having', 'where', 'group_by'}:
                            step_hyp.keyword = token

                            # detect nested sql
                            if token == 'select' and len(step_hyp.sql) > 2 and step_hyp.sql[-2] == '(': # not a nested where condition
                                assert step_hyp.nested_keywords[-1] in {'where', 'having'}
                                pass
                            elif len(step_hyp.nested_keywords) > 0:
                                step_hyp.nested_keywords.pop()
                                step_hyp.confirmed_items.pop()
                                step_hyp.avoid_items.pop()
                            step_hyp.nested_keywords.append(token)
                            step_hyp.confirmed_items.append(defaultdict(list))
                            step_hyp.avoid_items.append(defaultdict(list))

                        elif token == ')' and END_NESTED in step_hyp.tag_seq[-1]: # exit from a nested sql
                            step_hyp.nested_keywords.pop()
                            step_hyp.confirmed_items.pop()
                            step_hyp.avoid_items.pop()
                            step_hyp.keyword = step_hyp.nested_keywords[-1]

                        if step_hyp.sql[-1] == EOS_TOK or len(step_hyp.sql) >= max_generation_length:
                            completed_hypotheses.append(step_hyp)
                        else:
                            new_hypotheses.append(step_hyp)

            # sort the current hypotheses
            if len(new_hypotheses) == 0 or len(completed_hypotheses) >= 10 * beam_size:
                sorted_completed_hypotheses = Hypothesis.sort_hypotheses(completed_hypotheses, beam_size, 0.0)
                return sorted_completed_hypotheses

            hypotheses = Hypothesis.sort_hypotheses(new_hypotheses, beam_size, 0.0)
            if bool_verbal:
                Hypothesis.print_hypotheses(hypotheses)

            if stop_step is not None:
                dec_seq_length = len(hypotheses[0].dec_seq)
                if dec_seq_length >= stop_step + 1:
                    return hypotheses
