# Interactive SQL generator
import numpy as np
import traceback

from interaction_framework.ISQL import ISQL
from supermodel import *
from evaluation import build_foreign_key_map_from_json, evaluate_match_per_example
from preprocess_train_dev_data import get_table_dict
import pdb


class ISQLSyntaxSQL(ISQL):
    def __init__(self, semparser, error_detector, question_generator, user_simulator,
                 num_options, length_penalty=0.0, bool_structure_rev=False,
                 num_passes=1, dropout_rate=0.0):
        ISQL.__init__(self, semparser, error_detector, question_generator, user_simulator,
                      num_options, length_penalty=length_penalty, bool_structure_rev=bool_structure_rev,
                      num_passes=num_passes, dropout_rate=dropout_rate)

        self.table_dict = get_table_dict("syntaxSQL/data/tables.json")
        self.kmaps = build_foreign_key_map_from_json("syntaxSQL/data/tables.json")
        self.etype = "match"
        self.db_dir = "syntaxSQL/data/database/"

    def decode_per_pass(self, input_item, dec_beam_size=1, dec_prefix=None, stop_step=None,
                        avoid_items=None, confirmed_items=None, bool_verbal=False, dropout_rate=0.0):
        db_id = input_item["db_id"]
        if db_id not in self.table_dict: print "Error %s not in table_dict" % db_id

        if dropout_rate > 0.0:
            self.semparser.train()

        hypotheses = self.semparser.beam_search([input_item["question_toks"]] * 2,
                                                [] if dec_prefix is None else dec_prefix,
                                                self.table_dict[db_id],
                                                None if dec_beam_size == np.inf else dec_beam_size,
                                                length_penalty_factor=self.length_penalty,
                                                bool_verbal=bool_verbal, stop_step=stop_step,
                                                avoid_items=avoid_items, confirmed_items=confirmed_items)
        if stop_step is None:
            for output_idx, hyp in enumerate(hypotheses):
                hyp.sql = self.semparser.gen_sql(hyp.current_sql, self.table_dict[db_id])
                if bool_verbal: print("Predicted {}-th SQL: {}".format(output_idx, hyp.sql))
            if bool_verbal: print("-" * 50 + "\n")

        self.semparser.eval()

        return hypotheses

    def apply_pos_feedback(self, seg):
        seg_id = seg[0][0].split('-')[0]
        dec_seq_idx = seg[0][-1]
        # confirmed answer
        if seg_id in {SELECT_COL, WHERE_COL, GROUP_COL, HAV_COL, ORDER_COL}:
            confirm_idx = seg[0][1][-1]
        elif seg_id in {SELECT_AGG}:
            if seg[0][2] == 'none_agg':
                confirm_idx = 'none_agg'
            else:
                confirm_idx = AGG_OPS.index(seg[0][2])
        elif seg_id in {WHERE_OP, HAV_OP}:
            confirm_idx = NEW_WHERE_OPS.index(seg[0][2])
        else:
            return
        self.user_sim.confirmed_items[dec_seq_idx].add(confirm_idx)

    def apply_neg_feedback(self, seg, dec_seq, BIO_history, pointer, fix_id, input_item,
                           bool_verbal=False):
        dec_seq_idx = seg[0][-1]
        dec_seq_item = dec_seq[dec_seq_idx]
        seg_id = seg[0][0].split('-')[0]
        if fix_id == 0: # drop
            if seg_id in {SELECT_COL, WHERE_COL, GROUP_COL, HAV_COL, ORDER_COL}:
                drop_idx = seg[0][1][-1]
            elif seg_id in {SELECT_AGG}:
                if seg[0][2] == 'none_agg':
                    drop_idx = 'none_agg'
                else:
                    drop_idx = AGG_OPS.index(seg[0][2])
            elif seg_id in {WHERE_OP, HAV_OP}:
                drop_idx = NEW_WHERE_OPS.index(seg[0][2])
            else:
                raise Exception("Invalid seg_id %s with fix_id 0 (seg %s)" % (seg_id, seg))
            self.user_sim.avoid_items[dec_seq_idx].add(drop_idx)
            try:
                new_hyp = self.decode(input_item, dec_beam_size=1, dec_prefix=dec_seq[:dec_seq_idx],
                                      avoid_items=self.user_sim.avoid_items,
                                      confirmed_items=self.user_sim.confirmed_items, bool_verbal=bool_verbal)[0]
            except Exception:
                print("## WARNING: out of option for %s" % seg_id)
                if self.bool_structure_rev:
                    # assert seg_id in {WHERE_COL, GROUP_COL, ORDER_COL}
                    print("## WARNING: %s structure changes!" % seg_id)
                    if seg_id == WHERE_COL:
                        assert BIO_history[pointer - 1][1] == "where"
                        kw_partial_idx = BIO_history[pointer - 1][-1]
                        kw_item = list(dec_seq[kw_partial_idx])
                        kw_item[-1].remove(KW_OPS.index("where"))
                        kw_item[-2] -= 1
                        dec_seq[kw_partial_idx] = tuple(kw_item)
                    elif seg_id == GROUP_COL:
                        assert BIO_history[pointer - 1][1] == "groupBy"
                        kw_partial_idx = BIO_history[pointer - 1][-1]
                        kw_item = list(dec_seq[kw_partial_idx])
                        kw_item[-1].remove(KW_OPS.index("groupBy"))
                        kw_item[-2] -= 1
                        dec_seq[kw_partial_idx] = tuple(kw_item)
                    elif seg_id == ORDER_COL:
                        # order col
                        assert BIO_history[pointer - 1][1] == "orderBy"
                        kw_partial_idx = BIO_history[pointer - 1][-1]
                        kw_item = list(dec_seq[kw_partial_idx])
                        kw_item[-1].remove(KW_OPS.index("orderBy"))
                        kw_item[-2] -= 1
                        dec_seq[kw_partial_idx] = tuple(kw_item)
                    else:
                        print(traceback.print_exc())
                        return None

                    _ = self.user_sim.avoid_items.pop(dec_seq_idx)
                    new_hyp = self.decode(input_item, dec_beam_size=1, dec_prefix=dec_seq[:dec_seq_idx],
                                          avoid_items=self.user_sim.avoid_items,
                                          confirmed_items=self.user_sim.confirmed_items,
                                          bool_verbal=bool_verbal)[0]
                else:
                    print(traceback.print_exc())
                    new_hyp = None
            return new_hyp

        elif fix_id == 1: #flip
            assert seg_id in {ANDOR, WHERE_ROOT_TERM, HAV_ROOT_TERM, GROUP_NHAV}
            vet, bool_val = dec_seq_item
            dec_seq[dec_seq_idx] = (vet, 1 - bool_val)
            new_hyp = self.decode(input_item, dec_beam_size=1, dec_prefix=dec_seq[:(dec_seq_idx + 1)],
                                  avoid_items=self.user_sim.avoid_items,
                                  confirmed_items=self.user_sim.confirmed_items, bool_verbal=bool_verbal)[0]
            return new_hyp

        else:
            assert fix_id == 2 # re-decode
            if seg_id in {HAV_AGG, ORDER_AGG}:
                if seg[0][2] == 'none_agg':
                    self.user_sim.avoid_items[dec_seq_idx].add('none_agg')
                else:
                    self.user_sim.avoid_items[dec_seq_idx].add(AGG_OPS.index(seg[0][2]))
            elif seg_id == ORDER_DESC_ASC_LIMIT:
                self.user_sim.avoid_items[dec_seq_idx].add(DEC_ASC_OPS.index(seg[0][2]))
            else:
                assert seg_id == IUEN
                self.user_sim.avoid_items[dec_seq_idx].add(SQL_OPS.index(seg[0][1]))
            try:
                cand_hypotheses = self.decode(input_item, dec_beam_size=self.num_options, dec_prefix=dec_seq[:dec_seq_idx],
                                              avoid_items=self.user_sim.avoid_items,
                                              confirmed_items=self.user_sim.confirmed_items, stop_step=dec_seq_idx,
                                              bool_verbal=bool_verbal)
            except Exception:
                print(traceback.print_exc())
                if self.bool_structure_rev:
                    print("## WARNING: with revised structure but cannot generate valid candidate!")
                cand_hypotheses = None

            return cand_hypotheses

    def evaluation(self, input_item, hyp):
        hardness, bool_err, exact_score, partial_scores, _, _ = evaluate_match_per_example(
            input_item['query'], hyp.sql, input_item['db_id'], self.db_dir, self.kmaps)
        print("(Hardness: {}) bool_err {}, exact_score {}, partial_scores {}".format(
            hardness, bool_err, exact_score, partial_scores))

        return hardness, bool_err, exact_score, partial_scores