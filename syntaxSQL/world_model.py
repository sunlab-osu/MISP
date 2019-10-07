from MISP_SQL.world_model import WorldModel as BaseWorldModel
from .supermodel import *
from .preprocess_train_dev_data import get_table_dict


class WorldModel(BaseWorldModel):
    def __init__(self, semparser, num_options, bool_str_revisable, bool_seek,
                 num_passes=1, dropout_rate=0.0):
        BaseWorldModel.__init__(self, semparser, num_options, bool_str_revisable, bool_seek,
                            num_passes=num_passes, dropout_rate=dropout_rate)

        self.table_dict = get_table_dict("syntaxSQL/data/tables.json")
        self.etype = "match"

    def decode_per_pass(self, input_item, dec_beam_size=1, dec_prefix=None, stop_step=None,
                        avoid_items=None, confirmed_items=None, dropout_rate=0.0,
                        bool_collect_choices=False, bool_verbal=False):
        db_id = input_item["db_id"]
        if db_id not in self.table_dict: print "Error %s not in table_dict" % db_id

        if dropout_rate > 0.0:
            self.semparser.train()

        hypotheses = self.semparser.beam_search([input_item["question_toks"]] * 2,
                                                [] if dec_prefix is None else dec_prefix,
                                                self.table_dict[db_id],
                                                None if dec_beam_size == np.inf else dec_beam_size,
                                                bool_verbal=bool_verbal, stop_step=stop_step,
                                                avoid_items=avoid_items, confirmed_items=confirmed_items,
                                                bool_collect_choices=bool_collect_choices)
        if stop_step is None:
            for output_idx, hyp in enumerate(hypotheses):
                hyp.sql = self.semparser.gen_sql(hyp.current_sql, self.table_dict[db_id])
                if bool_verbal: print("Predicted {}-th SQL: {}".format(output_idx, hyp.sql))
            if bool_verbal: print("-" * 50 + "\n")

        if dropout_rate > 0.0:
            self.semparser.eval()

        return hypotheses

    def apply_pos_feedback(self, seg, dec_seq):
        seg_id = seg[0][0].split('-')[0]
        dec_seq_idx = seg[0][-1]
        # confirmed answer
        if seg_id in {SELECT_COL, WHERE_COL, GROUP_COL, HAV_COL, ORDER_COL}:
            confirm_idx = seg[0][1][-1]
        elif seg_id in {SELECT_AGG}:
            confirm_idx = seg[0][2][-1]
        elif seg_id in {WHERE_OP, HAV_OP}:
            confirm_idx = seg[0][2][-1]
        else: # WHERE_ROOTTERM, GROUP_NHAV, HAV_AGG, HAV_ROOTTERM, ORDER_AGG, ORDER_DESC_ASC_LIMIT
            return dec_seq[:(dec_seq_idx + 1)]

        self.confirmed_items[dec_seq_idx].add(confirm_idx)
        return dec_seq[:dec_seq_idx]

    def apply_neg_feedback(self, unit_type, seg, dec_seq):
        dec_seq_idx = seg[0][-1]
        dec_seq_item = dec_seq[dec_seq_idx]
        seg_id = seg[0][0].split('-')[0]
        if unit_type == 0: # drop
            if seg_id in {SELECT_COL, WHERE_COL, GROUP_COL, HAV_COL, ORDER_COL}:
                drop_idx = seg[0][1][-1]
            elif seg_id in {SELECT_AGG}:
                drop_idx = seg[0][2][-1]
            elif seg_id in {WHERE_OP, HAV_OP}:
                drop_idx = seg[0][2][-1]
            else:
                raise Exception("Invalid seg_id %s with fix_id 0 (seg %s)" % (seg_id, seg))
            self.avoid_items[dec_seq_idx].add(drop_idx)
            return dec_seq[:dec_seq_idx]

        elif unit_type == 1: #flip
            assert seg_id in {ANDOR, WHERE_ROOT_TERM, HAV_ROOT_TERM, GROUP_NHAV}
            vet, bool_val = dec_seq_item
            dec_seq[dec_seq_idx] = (vet, 1 - bool_val)
            return dec_seq[:(dec_seq_idx + 1)]

        else:
            assert unit_type == 2 # re-decode
            if seg_id in {HAV_AGG, ORDER_AGG}:
                self.avoid_items[dec_seq_idx].add(seg[0][2][-1])
            elif seg_id == ORDER_DESC_ASC_LIMIT:
                self.avoid_items[dec_seq_idx].add(DEC_ASC_OPS.index(seg[0][2]))
            else:
                assert seg_id == IUEN
                self.avoid_items[dec_seq_idx].add(seg[0][1][1])
            return dec_seq[:dec_seq_idx]

    def decode_revised_structure(self, seg, pointer, hyp, input_item, bool_verbal=False):
        seg_id = seg[0][0]
        dec_seq_idx = seg[0][-1]
        assert seg_id != SELECT_COL, "Error: Cannot remove all SELECT_COL!"

        if seg_id == WHERE_COL:
            print("## WARNING: %s structure changes!" % seg_id)
            assert hyp.tag_seq[pointer - 1][1] == "where"
            kw_partial_idx = hyp.tag_seq[pointer - 1][-1]
            kw_item = list(hyp.dec_seq[kw_partial_idx])
            kw_item[-1].remove(KW_OPS.index("where"))
            kw_item[-2] -= 1
            hyp.dec_seq[kw_partial_idx] = tuple(kw_item)
        elif seg_id == GROUP_COL:
            print("## WARNING: %s structure changes!" % seg_id)
            assert hyp.tag_seq[pointer - 1][1] == "groupBy"
            kw_partial_idx = hyp.tag_seq[pointer - 1][-1]
            kw_item = list(hyp.dec_seq[kw_partial_idx])
            kw_item[-1].remove(KW_OPS.index("groupBy"))
            kw_item[-2] -= 1
            hyp.dec_seq[kw_partial_idx] = tuple(kw_item)
        elif seg_id == HAV_COL:
            print("## WARNING: %s structure changes!" % seg_id)
            assert hyp.tag_seq[pointer - 1][1] == "having"
            kw_partial_idx = hyp.tag_seq[pointer - 1][-1]
            hyp.dec_seq[kw_partial_idx] = (('col', 'groupBy'), 0)
        elif seg_id == ORDER_COL:
            print("## WARNING: %s structure changes!" % seg_id)
            assert hyp.tag_seq[pointer - 1][1] == "orderBy"
            kw_partial_idx = hyp.tag_seq[pointer - 1][-1]
            kw_item = list(hyp.dec_seq[kw_partial_idx])
            kw_item[-1].remove(KW_OPS.index("orderBy"))
            kw_item[-2] -= 1
            hyp.dec_seq[kw_partial_idx] = tuple(kw_item)
        else:
            return pointer + 1, hyp

        self.avoid_items.pop(dec_seq_idx)
        hyp = self.decode(input_item, dec_beam_size=1, dec_prefix=hyp.dec_seq[:dec_seq_idx],
                          avoid_items=self.avoid_items,
                          confirmed_items=self.confirmed_items,
                          bool_verbal=bool_verbal)[0]

        return pointer, hyp
