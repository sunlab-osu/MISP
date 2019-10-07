import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from itertools import combinations
from modules.word_embedding import WordEmbedding
from modules.aggregator_predict import AggPredictor
from modules.selection_predict import SelPredictor
from modules.sqlnet_condition_predict import SQLNetCondPredictor
from SQLNet_model.sqlnet.utils import generate_sql_q1

from interaction_framework.ISQL import Hypothesis as BasicHypothesis
from interaction_framework.question_gen import OUTSIDE, SELECT_COL, SELECT_AGG, WHERE_COL, WHERE_OP, WHERE_VAL

AGG_OPS = ['None', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
COND_OPS = ['=', '>', '<', 'OP']


class Stack:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def peek(self):
        return self.items[len(self.items) - 1]

    def size(self):
        return len(self.items)

    def insert(self, i, x):
        return self.items.insert(i, x)


class Hypothesis(BasicHypothesis):
    def __init__(self, dec_prefix):
        BasicHypothesis.__init__(self, dec_prefix)
        self.stack = Stack()
        self.stack.push(("sc", None))
        self.tag_seq = [(OUTSIDE, ("sc", None), 1.0, None)]
        self.sql_i = {}


class SQLNet(nn.Module):
    def __init__(self, word_emb, N_word, N_h=100, N_depth=2,
            gpu=False, use_ca=True, trainable_emb=False, dr=0.3, temperature=False):
        super(SQLNet, self).__init__()
        self.use_ca = use_ca
        self.trainable_emb = trainable_emb
        self.temperature = temperature

        self.gpu = gpu
        self.N_h = N_h
        self.N_depth = N_depth

        self.max_col_num = 45
        self.max_tok_num = 200
        self.SQL_TOK = ['<UNK>', '<END>', 'WHERE', 'AND',
                'EQL', 'GT', 'LT', '<BEG>']
        self.COND_OPS = ['EQL', 'GT', 'LT']

        #Word embedding
        if trainable_emb:
            self.agg_embed_layer = WordEmbedding(word_emb, N_word, gpu,
                    self.SQL_TOK, our_model=True, trainable=trainable_emb)
            self.sel_embed_layer = WordEmbedding(word_emb, N_word, gpu,
                    self.SQL_TOK, our_model=True, trainable=trainable_emb)
            self.cond_embed_layer = WordEmbedding(word_emb, N_word, gpu,
                    self.SQL_TOK, our_model=True, trainable=trainable_emb)
        else:
            self.embed_layer = WordEmbedding(word_emb, N_word, gpu,
                    self.SQL_TOK, our_model=True, trainable=trainable_emb)
        
        #Predict aggregator
        self.agg_pred = AggPredictor(N_word, N_h, N_depth, use_ca=use_ca, dr=dr, temperature=temperature)

        #Predict selected column
        self.sel_pred = SelPredictor(N_word, N_h, N_depth,
                self.max_tok_num, use_ca=use_ca, dr=dr, temperature=temperature)

        #Predict number of cond
        self.cond_pred = SQLNetCondPredictor(N_word, N_h, N_depth,
                self.max_col_num, self.max_tok_num, use_ca, gpu, dr=dr, temperature=temperature)


        self.CE = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.log_softmax = nn.LogSoftmax()
        self.bce_logit = nn.BCEWithLogitsLoss()
        if gpu:
            self.cuda()

    def generate_gt_where_seq(self, q, col, query):
        ret_seq = []
        for cur_q, cur_col, cur_query in zip(q, col, query):
            cur_values = []
            st = cur_query.index(u'WHERE')+1 if \
                    u'WHERE' in cur_query else len(cur_query)
            all_toks = ['<BEG>'] + cur_q + ['<END>']
            while st < len(cur_query):
                ed = len(cur_query) if 'AND' not in cur_query[st:]\
                        else cur_query[st:].index('AND') + st
                if 'EQL' in cur_query[st:ed]:
                    op = cur_query[st:ed].index('EQL') + st
                elif 'GT' in cur_query[st:ed]:
                    op = cur_query[st:ed].index('GT') + st
                elif 'LT' in cur_query[st:ed]:
                    op = cur_query[st:ed].index('LT') + st
                else:
                    raise RuntimeError("No operator in it!")
                this_str = ['<BEG>'] + cur_query[op+1:ed] + ['<END>']
                cur_seq = [all_toks.index(s) if s in all_toks \
                        else 0 for s in this_str]
                cur_values.append(cur_seq)
                st = ed+1
            ret_seq.append(cur_values)
        return ret_seq

    def forward(self, q, col, col_num, pred_entry,
            gt_where = None, gt_cond=None, reinforce=False, gt_sel=None):
        B = len(q)
        pred_agg, pred_sel, pred_cond = pred_entry

        agg_score = None
        sel_score = None
        cond_score = None

        #Predict aggregator
        if self.trainable_emb:
            if pred_agg:
                x_emb_var, x_len = self.agg_embed_layer.gen_x_batch(q, col)
                col_inp_var, col_name_len, col_len = \
                        self.agg_embed_layer.gen_col_batch(col)
                max_x_len = max(x_len)
                agg_score = self.agg_pred(x_emb_var, x_len, col_inp_var,
                        col_name_len, col_len, col_num, gt_sel=gt_sel)

            if pred_sel:
                x_emb_var, x_len = self.sel_embed_layer.gen_x_batch(q, col)
                col_inp_var, col_name_len, col_len = \
                        self.sel_embed_layer.gen_col_batch(col)
                max_x_len = max(x_len)
                sel_score = self.sel_pred(x_emb_var, x_len, col_inp_var,
                        col_name_len, col_len, col_num)

            if pred_cond:
                x_emb_var, x_len = self.cond_embed_layer.gen_x_batch(q, col)
                col_inp_var, col_name_len, col_len = \
                        self.cond_embed_layer.gen_col_batch(col)
                max_x_len = max(x_len)
                cond_score = self.cond_pred(x_emb_var, x_len, col_inp_var,
                        col_name_len, col_len, col_num,
                        gt_where, gt_cond, reinforce=reinforce)
        else:
            x_emb_var, x_len = self.embed_layer.gen_x_batch(q, col)
            col_inp_var, col_name_len, col_len = \
                    self.embed_layer.gen_col_batch(col)
            max_x_len = max(x_len)
            if pred_agg:
                agg_score = self.agg_pred(x_emb_var, x_len, col_inp_var,
                        col_name_len, col_len, col_num, gt_sel=gt_sel)

            if pred_sel:
                sel_score = self.sel_pred(x_emb_var, x_len, col_inp_var,
                        col_name_len, col_len, col_num)

            if pred_cond:
                cond_score = self.cond_pred(x_emb_var, x_len, col_inp_var,
                        col_name_len, col_len, col_num,
                        gt_where, gt_cond, reinforce=reinforce)

        return (agg_score, sel_score, cond_score)

    def interaction_beam_forward(self, q, col, raw_q, raw_col, col_num, beam_size, dec_prefix,
                                 stop_step=None, avoid_items=None, confirmed_items=None, dropout_rate=0.0,
                                 bool_verbal=False):
        """
        @author: Ziyu Yao
        Beam search decoding for interactive sql generation.
        Only support batch size=1 and self.trainable_emb=True.
        """
        assert self.trainable_emb, "Support trainable_emb=True only."
        assert len(q) == 1

        dec_prefix = dec_prefix[::-1]
        hypotheses = [Hypothesis(dec_prefix)]
        completed_hypotheses = []
        table_name = None

        while True:
            new_hypotheses = []

            for hyp in hypotheses:
                if hyp.stack.isEmpty():
                    # sort conds by its col idx
                    conds = hyp.sql_i['conds']
                    sorted_conds = sorted(conds, key=lambda x: x[0])
                    hyp.sql_i['conds'] = sorted_conds
                    hyp.sql = generate_sql_q1(hyp.sql_i, raw_q[0], raw_col[0])
                    if bool_verbal:
                        print("Completed %d-th hypotheses: " % len(completed_hypotheses))
                        print("tag_seq:{}".format(hyp.tag_seq))
                        print("dec_seq: {}".format(hyp.dec_seq))
                        print("sql_i: {}".format(hyp.sql_i))
                        print("sql: {}".format(hyp.sql))
                    completed_hypotheses.append(hyp)  # add to completion
                else:
                    vet = hyp.stack.pop()
                    if vet[0] == "sc":
                        x_emb_var, x_len = self.sel_embed_layer.gen_x_batch(q, col, dropout_rate=dropout_rate)
                        col_inp_var, col_name_len, col_len = self.sel_embed_layer.gen_col_batch(col, dropout_rate=dropout_rate)
                        sel_score = self.sel_pred(x_emb_var, x_len, col_inp_var,
                                                  col_name_len, col_len, col_num, dropout_rate=dropout_rate).view(1, -1)
                        prob_sc = self.softmax(sel_score).data.cpu().numpy()[0]
                        hyp.tag_seq.append((OUTSIDE, 'select', 1.0, None))

                        if len(hyp.dec_prefix):
                            partial_vet, sc_idx = hyp.dec_prefix.pop()
                            assert partial_vet == vet
                            sc_candidates = [sc_idx]
                        else:
                            sc_candidates = np.argsort(-prob_sc)
                            # rm avoid candidates
                            if avoid_items is not None and hyp.dec_seq_idx in avoid_items:
                                sc_candidates = [sc_idx for sc_idx in sc_candidates if sc_idx not in avoid_items[hyp.dec_seq_idx]]
                            sc_candidates = sc_candidates[:beam_size]

                        for sc_idx in sc_candidates:
                            if len(sc_candidates) == 1:
                                step_hyp = hyp
                            else:
                                step_hyp = hyp.copy()
                            sc_name = raw_col[0][sc_idx]

                            step_hyp.sql_i['sel'] = sc_idx
                            step_hyp.dec_seq.append((vet, sc_idx))
                            step_hyp.tag_seq.append((SELECT_COL, (table_name, sc_name, sc_idx), prob_sc[sc_idx],
                                                     step_hyp.dec_seq_idx))
                            step_hyp.add_logprob(np.log(prob_sc[sc_idx]))
                            step_hyp.stack.push(("sa", (sc_idx, sc_name)))
                            step_hyp.dec_seq_idx += 1

                            new_hypotheses.append(step_hyp)

                    elif vet[0] == "sa":
                        sc_idx, sc_name = vet[1]
                        x_emb_var, x_len = self.agg_embed_layer.gen_x_batch(q, col, dropout_rate=dropout_rate)
                        col_inp_var, col_name_len, col_len = self.agg_embed_layer.gen_col_batch(col, dropout_rate=dropout_rate)
                        agg_score = self.agg_pred(
                            x_emb_var, x_len, col_inp_var, col_name_len, col_len, col_num,
                            gt_sel=[sc_idx], dropout_rate=dropout_rate).view(1, -1)
                        prob_sa = self.softmax(agg_score).data.cpu().numpy()[0]

                        if len(hyp.dec_prefix):
                            partial_vet, sa_idx = hyp.dec_prefix.pop()
                            assert partial_vet == vet
                            sa_candidates = [sa_idx]
                        else:
                            sa_candidates = np.argsort(-prob_sa)

                            if avoid_items is not None and hyp.dec_seq_idx in avoid_items:
                                sa_candidates = [sa_idx for sa_idx in sa_candidates if sa_idx not in
                                                 avoid_items[hyp.dec_seq_idx]]
                            sa_candidates = sa_candidates[:beam_size]

                        for sa_idx in sa_candidates:
                            if len(sa_candidates) == 1:
                                step_hyp = hyp
                            else:
                                step_hyp = hyp.copy()
                            sa_name = AGG_OPS[sa_idx]
                            if sa_name == 'None':
                                sa_name = 'none_agg' # for q gen usage

                            step_hyp.sql_i['agg'] = sa_idx
                            step_hyp.dec_seq.append((vet, sa_idx))
                            step_hyp.tag_seq.append((SELECT_AGG, (table_name, sc_name, sc_idx), sa_name, prob_sa[sa_idx],
                                                     step_hyp.dec_seq_idx))
                            step_hyp.add_logprob(np.log(prob_sa[sa_idx]))
                            step_hyp.stack.push(("wc", None))
                            step_hyp.dec_seq_idx += 1

                            new_hypotheses.append(step_hyp)

                    elif vet[0] == "wc":
                        hyp.tag_seq.append((OUTSIDE, 'where', 1.0, None))
                        hyp.sql_i['conds'] = []

                        step_hypotheses = []

                        x_emb_var, x_len = self.cond_embed_layer.gen_x_batch(q, col, dropout_rate=dropout_rate)
                        col_inp_var, col_name_len, col_len = self.cond_embed_layer.gen_col_batch(col, dropout_rate=dropout_rate)

                        # wn, wc
                        cond_num_score, cond_col_score = self.cond_pred.cols_forward(
                            x_emb_var, x_len, col_inp_var, col_name_len, col_len, dropout_rate=dropout_rate)
                        prob_wn = self.softmax(cond_num_score.view(1, -1)).data.cpu().numpy()[0]
                        prob_wc = self.sigmoid(cond_col_score.view(1, -1)).data.cpu().numpy()[0]

                        if len(hyp.dec_prefix):
                            partial_vet, wn, wc_list = hyp.dec_prefix.pop()
                            assert partial_vet == vet
                            col_num_cols_pair = [(wn, wc_list)]
                        else:
                            col_num_cols_pair = []
                            sorted_col_num = np.argsort(-prob_wn)
                            sorted_cols = np.argsort(-prob_wc)

                            # filter avoid_items
                            if avoid_items is not None and hyp.dec_seq_idx in avoid_items:
                                sorted_cols = [col_idx for col_idx in sorted_cols if col_idx not in avoid_items[hyp.dec_seq_idx]]
                                sorted_col_num = [col_num for col_num in sorted_col_num if col_num <= len(sorted_cols)]

                            # fix confirmed items
                            if confirmed_items is not None and hyp.dec_seq_idx in confirmed_items:
                                fixed_cols = list(confirmed_items[hyp.dec_seq_idx])
                                sorted_col_num = [col_num - len(fixed_cols) for col_num in sorted_col_num if
                                                  col_num >= len(fixed_cols)]
                                sorted_cols = [col_idx for col_idx in sorted_cols if col_idx not in fixed_cols]
                            else:
                                fixed_cols = []

                            for col_num in sorted_col_num: #[:beam_size]
                                if col_num == 0:
                                    col_num_cols_pair.append((len(fixed_cols), fixed_cols))
                                elif col_num == 1:
                                    col_num_cols_pair.extend(
                                        [(len(fixed_cols) + 1, fixed_cols + [col_idx]) for col_idx in sorted_cols[:beam_size]])
                                elif beam_size == 1:
                                    top_cols = list(sorted_cols[:col_num])
                                    # top_cols.sort()
                                    col_num_cols_pair.append((len(fixed_cols) + col_num, fixed_cols + top_cols))
                                else:
                                    combs = combinations(sorted_cols[:10], col_num)  # to reduce beam search time
                                    comb_score = []
                                    for comb in combs:
                                        score = sum([np.log(prob_wc[c_idx]) for c_idx in comb])
                                        comb_score.append((comb, score))
                                    sorted_comb_score = sorted(comb_score, key=lambda x: x[1], reverse=True)[:beam_size]
                                    for comb, _ in sorted_comb_score:
                                        comb_cols = list(comb)
                                        # comb_cols.sort()
                                        col_num_cols_pair.append((len(fixed_cols) + col_num, fixed_cols + comb_cols))

                        for col_num, cols in col_num_cols_pair:
                            if len(col_num_cols_pair) == 1:
                                step_hyp = hyp
                            else:
                                step_hyp = hyp.copy()

                            step_hyp.dec_seq.append((vet, col_num, cols))
                            step_hyp.add_logprob(np.log(prob_wn[col_num]))

                            for wc_idx in cols:
                                wc_name = raw_col[0][wc_idx]
                                step_hyp.tag_seq.append((WHERE_COL, (table_name, wc_name, wc_idx), prob_wc[wc_idx], step_hyp.dec_seq_idx))
                                step_hyp.add_logprob(np.log(prob_wc[wc_idx]))
                                step_hyp.stack.push(("wo", (wc_idx, wc_name)))

                            step_hyp.dec_seq_idx += 1
                            step_hypotheses.append(step_hyp)

                        step_hypotheses = Hypothesis.sort_hypotheses(step_hypotheses, beam_size, 0.0)
                        new_hypotheses.extend(step_hypotheses)

                    elif vet[0] == "wo":
                        wc_idx, wc_name = vet[1]
                        chosen_col_gt = [[wc_idx]]

                        x_emb_var, x_len = self.cond_embed_layer.gen_x_batch(q, col, dropout_rate=dropout_rate)
                        col_inp_var, col_name_len, col_len = self.cond_embed_layer.gen_col_batch(col, dropout_rate=dropout_rate)
                        cond_op_score = self.cond_pred.op_forward(
                            x_emb_var, x_len, col_inp_var, col_name_len, col_len, chosen_col_gt, dropout_rate=dropout_rate).view(1, 4, -1) #[B=1, 4, |OPS|]
                        prob_wo = self.softmax(cond_op_score[:,0,:]).data.cpu().numpy()[0]

                        if len(hyp.dec_prefix):
                            partial_vet, wo_idx = hyp.dec_prefix.pop()
                            assert partial_vet == vet
                            wo_candidates = [wo_idx]
                        else:
                            wo_candidates = np.argsort(-prob_wo)

                            if avoid_items is not None and hyp.dec_seq_idx in avoid_items:
                                wo_candidates = [wo_idx for wo_idx in wo_candidates if wo_idx not in
                                                 avoid_items[hyp.dec_seq_idx]]
                            wo_candidates = wo_candidates[:beam_size]

                        for wo_idx in wo_candidates:
                            if len(wo_candidates) == 1:
                                step_hyp = hyp
                            else:
                                step_hyp = hyp.copy()
                            wo_name = COND_OPS[wo_idx]

                            step_hyp.dec_seq.append((vet, wo_idx))
                            step_hyp.tag_seq.append((WHERE_OP, ((table_name, wc_name, wc_idx),), wo_name, prob_wo[wo_idx],
                                                     step_hyp.dec_seq_idx))
                            step_hyp.add_logprob(np.log(prob_wo[wo_idx]))
                            step_hyp.stack.push(("wv", (wc_idx, wc_name, wo_idx, wo_name)))
                            step_hyp.dec_seq_idx += 1

                            new_hypotheses.append(step_hyp)

                    elif vet[0] == "wv":
                        wc_idx, wc_name, wo_idx, wo_name = vet[1]
                        x_emb_var, x_len = self.cond_embed_layer.gen_x_batch(q, col, dropout_rate=dropout_rate)
                        col_inp_var, col_name_len, col_len = self.cond_embed_layer.gen_col_batch(col, dropout_rate=dropout_rate)

                        given_idxes, avoid_idxes_list = None, None
                        if len(hyp.dec_prefix):
                            partial_vet, given_idxes = hyp.dec_prefix.pop()
                            assert partial_vet == vet
                        elif avoid_items is not None and hyp.dec_seq_idx in avoid_items:
                            avoid_idxes_list = list(avoid_items[hyp.dec_seq_idx])

                        str_idxes_prob_pairs = self.cond_pred.val_beam_search(
                            x_emb_var, x_len, col_inp_var, col_name_len, col_len, [[wc_idx]], beam_size,
                            avoid_idxes_list=avoid_idxes_list, given_idxes=given_idxes, dropout_rate=dropout_rate)

                        all_toks = ['<BEG>'] + q[0] + ['<END>']
                        for str_idxes, logprob in str_idxes_prob_pairs:
                            if len(str_idxes_prob_pairs) == 1:
                                step_hyp = hyp
                            else:
                                step_hyp = hyp.copy()

                            # get val_str
                            cur_cond_str_toks = []
                            for wd_idx in str_idxes[1:]:
                                str_val = all_toks[wd_idx]
                                if str_val == '<END>':
                                    break
                                cur_cond_str_toks.append(str_val)
                            val_str = SQLNet.merge_tokens(cur_cond_str_toks, raw_q[0])

                            step_hyp.sql_i['conds'].append([wc_idx, wo_idx, val_str])
                            step_hyp.dec_seq.append((vet, str_idxes))
                            step_hyp.tag_seq.append((WHERE_VAL, ((table_name, wc_name, wc_idx),), wo_name,
                                                     (str_idxes, val_str), np.exp(logprob), hyp.dec_seq_idx))
                            step_hyp.add_logprob(logprob)
                            step_hyp.dec_seq_idx += 1

                            new_hypotheses.append(step_hyp)

            if len(new_hypotheses) == 0:
                # sort completed hypotheses
                sorted_completed_hypotheses = Hypothesis.sort_hypotheses(completed_hypotheses, beam_size, 0.0)
                return sorted_completed_hypotheses

            # if bool_verbal:
            #     print("Before sorting...")
            #     Hypothesis.print_hypotheses(new_hypotheses)
            hypotheses = Hypothesis.sort_hypotheses(new_hypotheses, beam_size, 0.0)
            if bool_verbal:
                print("\nAfter sorting...")
                Hypothesis.print_hypotheses(hypotheses)

            if stop_step is not None: # for one-step beam search; the partial_seq lengths must be the same for all hyps
                dec_seq_length = len(hypotheses[0].dec_seq)
                if dec_seq_length == stop_step + 1:
                    for hyp in hypotheses:
                        assert len(hyp.dec_seq) == dec_seq_length
                    return hypotheses

    def loss(self, score, truth_num, pred_entry, gt_where):
        pred_agg, pred_sel, pred_cond = pred_entry
        agg_score, sel_score, cond_score = score

        loss = 0
        loss_agg, loss_sel, loss_cond = 0., 0., 0.
        if pred_agg:
            agg_truth = map(lambda x:x[0], truth_num)
            data = torch.from_numpy(np.array(agg_truth))
            if self.gpu:
                agg_truth_var = Variable(data.cuda())
            else:
                agg_truth_var = Variable(data)

            loss_agg = self.CE(agg_score, agg_truth_var)
            loss += loss_agg

        if pred_sel:
            sel_truth = map(lambda x:x[1], truth_num)
            data = torch.from_numpy(np.array(sel_truth))
            if self.gpu:
                sel_truth_var = Variable(data.cuda())
            else:
                sel_truth_var = Variable(data)

            loss_sel = self.CE(sel_score, sel_truth_var)
            loss += loss_sel

        if pred_cond:
            B = len(truth_num)
            cond_num_score, cond_col_score,\
                    cond_op_score, cond_str_score = cond_score
            #Evaluate the number of conditions
            cond_num_truth = map(lambda x:x[2], truth_num)
            data = torch.from_numpy(np.array(cond_num_truth))
            if self.gpu:
                cond_num_truth_var = Variable(data.cuda())
            else:
                cond_num_truth_var = Variable(data)

            cond_num_loss = self.CE(cond_num_score, cond_num_truth_var)
            loss_cond += cond_num_loss
            loss += cond_num_loss

            #Evaluate the columns of conditions
            T = len(cond_col_score[0])
            truth_prob = np.zeros((B, T), dtype=np.float32)
            for b in range(B):
                if len(truth_num[b][3]) > 0:
                    truth_prob[b][list(truth_num[b][3])] = 1
            data = torch.from_numpy(truth_prob)
            if self.gpu:
                cond_col_truth_var = Variable(data.cuda())
            else:
                cond_col_truth_var = Variable(data)

            sigm = nn.Sigmoid()
            cond_col_prob = sigm(cond_col_score)
            bce_loss = -torch.mean( 3*(cond_col_truth_var * \
                    torch.log(cond_col_prob+1e-10)) + \
                    (1-cond_col_truth_var) * torch.log(1-cond_col_prob+1e-10) )
            loss_cond += bce_loss
            loss += bce_loss

            #Evaluate the operator of conditions
            for b in range(len(truth_num)):
                if len(truth_num[b][4]) == 0:
                    continue
                data = torch.from_numpy(np.array(truth_num[b][4]))
                if self.gpu:
                    cond_op_truth_var = Variable(data.cuda())
                else:
                    cond_op_truth_var = Variable(data)
                cond_op_pred = cond_op_score[b, :len(truth_num[b][4])]
                cond_op_loss_b = (self.CE(cond_op_pred, cond_op_truth_var) / len(truth_num))
                loss_cond += cond_op_loss_b
                loss += cond_op_loss_b

            #Evaluate the strings of conditions
            for b in range(len(gt_where)):
                for idx in range(len(gt_where[b])):
                    cond_str_truth = gt_where[b][idx]
                    if len(cond_str_truth) == 1:
                        continue
                    data = torch.from_numpy(np.array(cond_str_truth[1:]))
                    if self.gpu:
                        cond_str_truth_var = Variable(data.cuda())
                    else:
                        cond_str_truth_var = Variable(data)
                    str_end = len(cond_str_truth)-1
                    cond_str_pred = cond_str_score[b, idx, :str_end]
                    cond_str_loss_b = (self.CE(cond_str_pred, cond_str_truth_var) / (len(gt_where) * len(gt_where[b])))
                    loss_cond += cond_str_loss_b
                    loss += cond_str_loss_b

        if self.temperature:
            return [loss, loss_sel, loss_agg, loss_cond]

        return [loss]

    def check_acc(self, vis_info, pred_queries, gt_queries, pred_entry):
        def pretty_print(vis_data):
            print 'question:', vis_data[0]
            print 'headers: (%s)'%(' || '.join(vis_data[1]))
            print 'query:', vis_data[2]

        def gen_cond_str(conds, header):
            if len(conds) == 0:
                return 'None'
            cond_str = []
            for cond in conds:
                cond_str.append(header[cond[0]] + ' ' +
                    self.COND_OPS[cond[1]] + ' ' + unicode(cond[2]).lower())
            return 'WHERE ' + ' AND '.join(cond_str)

        pred_agg, pred_sel, pred_cond = pred_entry

        B = len(gt_queries)

        tot_err = agg_err = sel_err = cond_err = 0.0
        cond_num_err = cond_col_err = cond_op_err = cond_val_err = 0.0
        agg_ops = ['None', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
        for b, (pred_qry, gt_qry) in enumerate(zip(pred_queries, gt_queries)):
            good = True
            if pred_agg:
                agg_pred = pred_qry['agg']
                agg_gt = gt_qry['agg']
                if agg_pred != agg_gt:
                    agg_err += 1
                    good = False

            if pred_sel:
                sel_pred = pred_qry['sel']
                sel_gt = gt_qry['sel']
                if sel_pred != sel_gt:
                    sel_err += 1
                    good = False

            if pred_cond:
                cond_pred = pred_qry['conds']
                cond_gt = gt_qry['conds']
                flag = True
                if len(cond_pred) != len(cond_gt):
                    flag = False
                    cond_num_err += 1

                if flag and set(x[0] for x in cond_pred) != \
                        set(x[0] for x in cond_gt):
                    flag = False
                    cond_col_err += 1

                for idx in range(len(cond_pred)):
                    if not flag:
                        break
                    gt_idx = tuple(
                            x[0] for x in cond_gt).index(cond_pred[idx][0])
                    if flag and cond_gt[gt_idx][1] != cond_pred[idx][1]:
                        flag = False
                        cond_op_err += 1

                for idx in range(len(cond_pred)):
                    if not flag:
                        break
                    gt_idx = tuple(
                            x[0] for x in cond_gt).index(cond_pred[idx][0])
                    if flag and unicode(cond_gt[gt_idx][2]).lower() != \
                            unicode(cond_pred[idx][2]).lower():
                        flag = False
                        cond_val_err += 1

                if not flag:
                    cond_err += 1
                    good = False

            if not good:
                tot_err += 1

        return np.array((agg_err, sel_err, cond_err)), tot_err

    @staticmethod
    def merge_tokens(tok_list, raw_tok_str):
        tok_str = raw_tok_str.lower()
        alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789$('
        special = {'-LRB-': '(',
                   '-RRB-': ')',
                   '-LSB-': '[',
                   '-RSB-': ']',
                   '``': '"',
                   '\'\'': '"',
                   '--': u'\u2013'}
        ret = ''
        double_quote_appear = 0
        for raw_tok in tok_list:
            if not raw_tok:
                continue
            tok = special.get(raw_tok, raw_tok)
            if tok == '"':
                double_quote_appear = 1 - double_quote_appear

            if len(ret) == 0:
                pass
            elif len(ret) > 0 and ret + ' ' + tok in tok_str:
                ret = ret + ' '
            elif len(ret) > 0 and ret + tok in tok_str:
                pass
            elif tok == '"':
                if double_quote_appear:
                    ret = ret + ' '
            elif tok[0] not in alphabet:
                pass
            elif (ret[-1] not in ['(', '/', u'\u2013', '#', '$', '&']) \
                    and (ret[-1] != '"' or not double_quote_appear):
                ret = ret + ' '
            ret = ret + tok
        return ret.strip()

    def gen_query(self, score, q, col, raw_q, raw_col,
            pred_entry, reinforce=False, verbose=False):
        pred_agg, pred_sel, pred_cond = pred_entry
        agg_score, sel_score, cond_score = score

        ret_queries = []
        if pred_agg:
            B = len(agg_score)
        elif pred_sel:
            B = len(sel_score)
        elif pred_cond:
            B = len(cond_score[0])
        for b in range(B):
            cur_query = {}
            if pred_agg:
                cur_query['agg'] = np.argmax(agg_score[b].data.cpu().numpy())
            if pred_sel:
                cur_query['sel'] = np.argmax(sel_score[b].data.cpu().numpy())
            if pred_cond:
                cur_query['conds'] = []
                cond_num_score,cond_col_score,cond_op_score,cond_str_score =\
                        [x.data.cpu().numpy() for x in cond_score]
                cond_num = np.argmax(cond_num_score[b])
                all_toks = ['<BEG>'] + q[b] + ['<END>']
                max_idxes = np.argsort(-cond_col_score[b])[:cond_num]
                for idx in range(cond_num):
                    cur_cond = []
                    cur_cond.append(max_idxes[idx])
                    cur_cond.append(np.argmax(cond_op_score[b][idx]))
                    cur_cond_str_toks = []
                    for str_score in cond_str_score[b][idx]:
                        str_tok = np.argmax(str_score[:len(all_toks)])
                        str_val = all_toks[str_tok]
                        if str_val == '<END>':
                            break
                        cur_cond_str_toks.append(str_val)
                    cur_cond.append(SQLNet.merge_tokens(cur_cond_str_toks, raw_q[b]))
                    cur_query['conds'].append(cur_cond)
            ret_queries.append(cur_query)

        return ret_queries