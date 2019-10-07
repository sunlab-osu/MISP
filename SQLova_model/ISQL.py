# Interactive SQL generator

import numpy as np
import traceback

from interaction_framework.ISQL import ISQL
from interaction_framework.question_gen import SELECT_COL, SELECT_AGG, WHERE_COL, WHERE_OP, WHERE_VAL
from .sqlova.model.nl2sql.wikisql_models import AGG_OPS, COND_OPS
from .sqlova.utils.utils_wikisql import *

import pdb


def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()


def cancel_dropout(m):
    if type(m) == nn.Dropout:
        m.eval()


class ISQLSQLova(ISQL):
    def __init__(self, bert_info, semparser, error_detector, question_generator,
                 user_simulator, num_options, bool_structure_rev=False, num_passes=1, dropout_rate=0.):
        ISQL.__init__(self, semparser, error_detector, question_generator, user_simulator, num_options,
                      bool_structure_rev=bool_structure_rev, num_passes=num_passes, dropout_rate=dropout_rate)

        bert_config, model_bert, tokenizer, max_seq_length, num_target_layers = bert_info
        self.model_bert = model_bert
        self.tokenizer = tokenizer
        self.bert_config = bert_config
        self.max_seq_length = max_seq_length
        self.num_target_layers = num_target_layers

    def decode_per_pass(self, input_item, dec_beam_size=1, dec_prefix=None, stop_step=None,
                        avoid_items=None, confirmed_items=None, bool_verbal=False, dropout_rate=0.0):
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
            bool_verbal=bool_verbal) #, dropout_rate=dropout_rate

        if stop_step is None and bool_verbal:
            for output_idx, hyp in enumerate(hypotheses):
                print("Predicted {}-th SQL: {}".format(output_idx, hyp.sql))

        self.semparser.eval()
        self.model_bert.apply(cancel_dropout)

        return hypotheses

    def apply_pos_feedback(self, seg):
        seg_id = seg[0][0]
        dec_seq_idx = seg[0][-1]
        # confirmed answer
        if seg_id in {WHERE_COL}:
            confirm_idx = seg[0][1][-1]
            self.user_sim.confirmed_items[dec_seq_idx].add(confirm_idx)
        else:
            pass # for SELECT_COL/AGG, WHERE_OP, WHERE_VAL, simply pass

    def apply_neg_feedback(self, seg, dec_seq, BIO_history, pointer, fix_id, input_item,
                           bool_verbal=False):
        dec_seq_idx = seg[0][-1]
        seg_id = seg[0][0]
        if fix_id == 0:  # drop
            if seg_id in {SELECT_COL, WHERE_COL}:
                drop_idx = seg[0][1][-1]
            elif seg_id == SELECT_AGG:
                if seg[0][2] == 'none_agg':
                    drop_idx = 0
                else:
                    drop_idx = AGG_OPS.index(seg[0][2])
            elif seg_id == WHERE_OP:
                drop_idx = COND_OPS.index(seg[0][2])
            else:
                raise Exception("Invalid seg_id %s with fix_id 0 (seg %s)" % (seg_id, seg))
            self.user_sim.avoid_items[dec_seq_idx].add(drop_idx)
            try:
                new_hyp = self.decode(input_item, dec_beam_size=1, dec_prefix=dec_seq[:dec_seq_idx],
                                      avoid_items=self.user_sim.avoid_items,
                                      confirmed_items=self.user_sim.confirmed_items,
                                      bool_verbal=bool_verbal)[0]
            except:
                print("## WARNING: out of option for %s" % seg_id)
                if self.bool_structure_rev and seg_id == WHERE_COL:
                    print("## WARNING: %s structure changes!" % seg_id)
                    dec_seq_item = list(dec_seq[dec_seq_idx])
                    dec_seq[dec_seq_idx] = (dec_seq_item[0], 0, [])
                    # _ = avoid_items.pop(dec_seq_idx)
                    new_hyp = self.decode(input_item, dec_beam_size=1, dec_prefix=dec_seq[:(dec_seq_idx + 1)],
                                          avoid_items=self.user_sim.avoid_items,
                                          confirmed_items=self.user_sim.confirmed_items,
                                          bool_verbal=bool_verbal)[0]
                else:
                    new_hyp = None
            return new_hyp

        else:
            assert fix_id == 2 and seg_id == WHERE_VAL  # re-decode
            st, ed = seg[0][3][:2]
            self.user_sim.avoid_items[dec_seq_idx].add((st, ed))
            try:
                cand_hypotheses = self.decode(input_item, dec_beam_size=self.num_options,
                                              dec_prefix=dec_seq[:dec_seq_idx],
                                              avoid_items=self.user_sim.avoid_items,
                                              confirmed_items=self.user_sim.confirmed_items,
                                              stop_step=dec_seq_idx,
                                              bool_verbal=bool_verbal)
            except Exception:
                print(traceback.print_exc())
                print("## WARNING: out of option for %s" % seg_id)
                cand_hypotheses = None

            return cand_hypotheses

    def evaluation(self, p_list, g_list, engine, tb):
        pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi, pr_sql_i = p_list
        g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi, sql_i = g_list

        cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, \
        cnt_wc1_list, cnt_wo1_list, \
        cnt_wvi1_list, cnt_wv1_list = get_cnt_sw_list(g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi,
                                                      pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi,
                                                      sql_i, pr_sql_i,
                                                      mode='test')

        cnt_lx1_list = get_cnt_lx_list(cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list,
                                       cnt_wo1_list, cnt_wv1_list)
        lx_correct = sum(cnt_lx1_list) # lx stands for logical form accuracy

        cnt_x1_list, g_ans, pr_ans = get_cnt_x_list(engine, tb, g_sc, g_sa, sql_i, pr_sc, pr_sa, pr_sql_i)
        x_correct = sum(cnt_x1_list)

        cnt_list1 = [cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list, cnt_wo1_list, cnt_wv1_list, cnt_lx1_list,
                     cnt_x1_list]
        if self.user_sim.user_type == "sim":
            print("lf correct: {}, x correct: {}, cnt_list: {}".format(lx_correct, x_correct, cnt_list1))

        return cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list, cnt_wo1_list, cnt_wv1_list, cnt_wvi1_list, \
               cnt_lx1_list, cnt_x1_list, cnt_list1, g_ans, pr_ans
