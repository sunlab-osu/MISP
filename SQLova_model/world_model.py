from MISP_SQL.world_model import WorldModel as BaseWorldModel
from MISP_SQL.utils import *
from .sqlova.utils.utils_wikisql import *


def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()


def cancel_dropout(m):
    if type(m) == nn.Dropout:
        m.eval()


class WorldModel(BaseWorldModel):
    def __init__(self, bert_info, semparser, num_options, num_passes=1, dropout_rate=0.0,
                 bool_structure_question=False):
        BaseWorldModel.__init__(self, semparser, num_options,
                                num_passes=num_passes, dropout_rate=dropout_rate)

        bert_config, model_bert, tokenizer, max_seq_length, num_target_layers = bert_info
        self.model_bert = model_bert
        self.tokenizer = tokenizer
        self.bert_config = bert_config
        self.max_seq_length = max_seq_length
        self.num_target_layers = num_target_layers

        self.bool_structure_question = bool_structure_question

    def decode_per_pass(self, input_item, dec_beam_size=1, dec_prefix=None, stop_step=None,
                        avoid_items=None, confirmed_items=None, dropout_rate=0.0,
                        bool_collect_choices=False, bool_verbal=False):
        if len(input_item) == 4:
            tb, nlu_t, nlu, hds = input_item

            wemb_n, wemb_h, l_n, l_hpu, l_hs, \
            nlu_tt, t_to_tt_idx, tt_to_t_idx \
                = get_wemb_bert(self.bert_config, self.model_bert, self.tokenizer, nlu_t, hds, self.max_seq_length,
                                num_out_layers_n=self.num_target_layers, num_out_layers_h=self.num_target_layers)
        else:
            wemb_n, l_n, wemb_h, l_hpu, l_hs, tb, nlu_t, nlu_tt, tt_to_t_idx, nlu = input_item

        if dropout_rate > 0.0:
            self.semparser.train()
            self.model_bert.apply(apply_dropout)

        hypotheses = self.semparser.interaction_beam_forward(
            wemb_n, l_n, wemb_h, l_hpu, l_hs, tb, nlu_t, nlu_tt, tt_to_t_idx, nlu,
            None if dec_beam_size == np.inf else dec_beam_size,
            [] if dec_prefix is None else dec_prefix,
            stop_step=stop_step, avoid_items=avoid_items, confirmed_items=confirmed_items,
            bool_collect_choices=bool_collect_choices,
            bool_verbal=bool_verbal)

        if stop_step is None and bool_verbal:
            for output_idx, hyp in enumerate(hypotheses):
                print("Predicted {}-th SQL: {}".format(output_idx, hyp.sql))

        if dropout_rate > 0.0:
            self.semparser.eval()
            self.model_bert.apply(cancel_dropout)

        return hypotheses

    def apply_pos_feedback(self, semantic_unit, dec_seq, dec_prefix):
        semantic_tag = semantic_unit[0]
        dec_seq_idx = semantic_unit[-1]

        # confirmed answer
        if semantic_tag in {WHERE_COL}:
            confirm_idx = semantic_unit[1][-1]
            self.confirmed_items[dec_seq_idx].add(confirm_idx)
            return dec_prefix
        else:
            # for SELECT_COL/AGG, WHERE_OP, WHERE_VAL, finalize the verified value
            try:
                assert dec_prefix == dec_seq[:dec_seq_idx]
            except AssertionError:
                print("AssertionError in apply_pos_feedback:\ndec_seq[:dec_seq_idx]={}\ndec_prefix={}".format(
                    dec_seq[:dec_seq_idx], dec_prefix))
            return dec_seq[:(dec_seq_idx + 1)]

    def apply_neg_feedback(self, semantic_unit, dec_seq, dec_prefix):
        dec_seq_idx = semantic_unit[-1]
        semantic_tag = semantic_unit[0]

        if semantic_tag in {SELECT_COL, WHERE_COL}:
            drop_idx = semantic_unit[1][-1]
            self.avoid_items[dec_seq_idx].add(drop_idx)
        elif semantic_tag == SELECT_AGG:
            drop_idx = semantic_unit[2][-1]
            self.avoid_items[dec_seq_idx].add(drop_idx)
        elif semantic_tag == WHERE_OP:
            drop_idx = semantic_unit[2][-1]
            self.avoid_items[dec_seq_idx].add(drop_idx)
        else:
            assert semantic_tag == WHERE_VAL  # re-decode
            st, ed = semantic_unit[3][:2]
            self.avoid_items[dec_seq_idx].add((st, ed))

        return dec_prefix

    def decode_revised_structure(self, semantic_unit, pointer, hyp, input_item, bool_verbal=False):
        semantic_tag = semantic_unit[0]
        assert semantic_tag != SELECT_COL, "Error: Cannot remove all SELECT_COL!"

        if semantic_tag == WHERE_COL:
            print("## WARNING: %s structure changes!" % semantic_tag)
            dec_seq_idx = semantic_unit[-1]
            dec_seq_item = list(hyp.dec_seq[dec_seq_idx])
            hyp.dec_seq[dec_seq_idx] = (dec_seq_item[0], 0, [])
            hyp = self.decode(input_item, dec_beam_size=1,
                              dec_prefix=hyp.dec_seq[:(dec_seq_idx + 1)],
                              avoid_items=self.avoid_items,
                              confirmed_items=self.confirmed_items,
                              bool_verbal=bool_verbal)[0]
            return pointer, hyp
        else:
            return pointer + 1, hyp

    def refresh_decoding(self, input_item, dec_prefix, old_hyp, semantic_unit,
                         pointer, sel_none_of_above, user_selections, bool_verbal=False):
        semantic_tag = semantic_unit[0]
        dec_seq_idx = semantic_unit[-1]

        if self.bool_structure_question and (sel_none_of_above + 1) in user_selections:
            assert semantic_tag == WHERE_COL

            dec_seq_idx = semantic_unit[-1]
            dec_seq_item = list(old_hyp.dec_seq[dec_seq_idx])
            dec_prefix.append((dec_seq_item[0], 0, []))
            hyp = self.decode(input_item, dec_beam_size=1,
                              dec_prefix=dec_prefix,
                              avoid_items=self.avoid_items,
                              confirmed_items=self.confirmed_items,
                              bool_verbal=bool_verbal)[0]
            print("DEBUG: new_hyp.sql = {}\n".format(hyp.sql))

            start_pos = pointer

        else:
            try:
                partial_hyp = self.decode(
                    input_item, dec_prefix=dec_prefix,
                    avoid_items=self.avoid_items,
                    confirmed_items=self.confirmed_items,
                    stop_step=dec_seq_idx,
                    bool_verbal=bool_verbal)[0]
            except Exception:  # e.g., when any WHERE_COL is redundant
                start_pos, hyp = self.decode_revised_structure(
                    semantic_unit, pointer, old_hyp, input_item,
                    bool_verbal=bool_verbal)
            else:
                # the following finds the next pointer to validate
                _, cand_pointers = semantic_unit_segment(partial_hyp.tag_seq)
                last_pointer = cand_pointers[-1]
                if last_pointer < pointer:  # structure changed, e.g., #cols reduce
                    start_pos = last_pointer + 1
                else:
                    start_pos = pointer + 1

                # generate a new hypothesis after interaction
                hyp = self.decode(
                    input_item, dec_prefix=dec_prefix,
                    avoid_items=self.avoid_items,
                    confirmed_items=self.confirmed_items,
                    bool_verbal=bool_verbal)[0]

        return start_pos, hyp
