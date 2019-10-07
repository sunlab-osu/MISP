# Interactive SQL generator

import numpy as np
import traceback

from interaction_framework.ISQL import ISQL
from interaction_framework.question_gen import SELECT_COL, SELECT_AGG, WHERE_COL, WHERE_OP, WHERE_VAL
from .sqlnet.model.sqlnet import AGG_OPS, COND_OPS
import pdb


class ISQLSQLNet(ISQL):
    def __init__(self, semparser, error_detector, question_generator, user_simulator, num_options,
                 bool_structure_rev=False, num_passes=1, dropout_rate=0.):
        ISQL.__init__(self, semparser, error_detector, question_generator, user_simulator, num_options,
                      bool_structure_rev=bool_structure_rev, num_passes=num_passes, dropout_rate=dropout_rate)

    def decode_per_pass(self, input_item, dec_beam_size=1, dec_prefix=None, stop_step=None,
                        avoid_items=None, confirmed_items=None, bool_verbal=False, dropout_rate=0.0):
        q_seq, col_seq, raw_q_seq, raw_col_seq, col_num = input_item

        if dropout_rate > 0.0:
            self.semparser.train()

        hypotheses = self.semparser.interaction_beam_forward(
            q_seq, col_seq, raw_q_seq, raw_col_seq, col_num,
            None if dec_beam_size == np.inf else dec_beam_size,
            [] if dec_prefix is None else dec_prefix,
            stop_step=stop_step, avoid_items=avoid_items, confirmed_items=confirmed_items,
            bool_verbal=bool_verbal) # , dropout_rate=dropout_rate

        if stop_step is None and bool_verbal:
            for output_idx, hyp in enumerate(hypotheses):
                print("Predicted {}-th SQL: {}".format(output_idx, hyp.sql))

        self.semparser.eval()

        return hypotheses

    def apply_pos_feedback(self, seg):
        seg_id = seg[0][0].split('-')[0]
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
            str_idxes = seg[0][3][0]
            self.user_sim.avoid_items[dec_seq_idx].add(tuple(str_idxes))
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

    def evaluation(self, raw_data, pred_queries, query_gt, table_ids, engine):
        exe_tot_acc_num = 0.
        qm_one_acc_num = 0.
        qm_tot_acc_num = 0.

        one_err, tot_err = self.semparser.check_acc(raw_data, pred_queries, query_gt, (True, True, True))

        for idx, (sql_gt, sql_pred, tid) in enumerate(
                zip(query_gt, pred_queries, table_ids)):
            ret_gt = engine.execute(tid,
                                    sql_gt['sel'], sql_gt['agg'], sql_gt['conds'])
            try:
                ret_pred = engine.execute(tid,
                                          sql_pred['sel'], sql_pred['agg'], sql_pred['conds'])
            except:
                ret_pred = None
            exe_tot_acc_num += (ret_gt == ret_pred)

        qm_one_acc_num += (len(raw_data) - one_err) #ed - st
        qm_tot_acc_num += (len(raw_data) - tot_err)

        if self.user_sim.user_type == "sim":
            print("qm correct: {}, exe correct: {}".format(qm_tot_acc_num, exe_tot_acc_num))

        return qm_one_acc_num, qm_tot_acc_num, exe_tot_acc_num
