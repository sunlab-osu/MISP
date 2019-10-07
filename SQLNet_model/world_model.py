from MISP_SQL.world_model import WorldModel as BaseWorldModel
from MISP_SQL.utils import SELECT_COL, SELECT_AGG, WHERE_COL, WHERE_OP, WHERE_VAL, np

AGG_OPS = ['None', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
COND_OPS = ['=', '>', '<', 'OP']


class WorldModel(BaseWorldModel):
    def __init__(self, semparser, num_options, bool_str_revisable, bool_seek,
                 num_passes=1, dropout_rate=0.0):
        BaseWorldModel.__init__(self, semparser, num_options, bool_str_revisable, bool_seek,
                            num_passes=num_passes, dropout_rate=dropout_rate)

    def decode_per_pass(self, input_item, dec_beam_size=1, dec_prefix=None, stop_step=None,
                        avoid_items=None, confirmed_items=None, dropout_rate=0.0,
                        bool_collect_choices=False, bool_verbal=False):
        q_seq, col_seq, raw_q_seq, raw_col_seq, col_num = input_item

        if dropout_rate > 0.0:
            self.semparser.train()

        hypotheses = self.semparser.interaction_beam_forward(
            q_seq, col_seq, raw_q_seq, raw_col_seq, col_num,
            None if dec_beam_size == np.inf else dec_beam_size,
            [] if dec_prefix is None else dec_prefix,
            stop_step=stop_step, avoid_items=avoid_items, confirmed_items=confirmed_items,
            bool_collect_choices=bool_collect_choices, bool_verbal=bool_verbal)

        if stop_step is None and bool_verbal:
            for output_idx, hyp in enumerate(hypotheses):
                print("Predicted {}-th SQL: {}".format(output_idx, hyp.sql))

        if dropout_rate > 0.0:
            self.semparser.eval()

        return hypotheses

    def apply_pos_feedback(self, seg, dec_seq):
        seg_id = seg[0][0].split('-')[0]
        dec_seq_idx = seg[0][-1]
        # confirmed answer
        if seg_id in {WHERE_COL}:
            confirm_idx = seg[0][1][-1]
            self.confirmed_items[dec_seq_idx].add(confirm_idx)
            # in case of conflict
            if confirm_idx in self.avoid_items[dec_seq_idx]:
                self.avoid_items[dec_seq_idx].remove(confirm_idx)

            return dec_seq[:dec_seq_idx]
        else:
            # for SELECT_COL/AGG, WHERE_OP, WHERE_VAL, finalize the verified value
            return dec_seq[:(dec_seq_idx + 1)]

    def apply_neg_feedback(self, unit_type, seg, dec_seq):
        dec_seq_idx = seg[0][-1]
        seg_id = seg[0][0]
        if unit_type == 0:  # drop
            if seg_id in {SELECT_COL, WHERE_COL}:
                drop_idx = seg[0][1][-1]
            elif seg_id == SELECT_AGG:
                drop_idx = seg[0][2][1]
            elif seg_id == WHERE_OP:
                drop_idx = seg[0][2][1]
            else:
                raise Exception("Invalid seg_id %s with fix_id 0 (seg %s)" % (seg_id, seg))
            self.avoid_items[dec_seq_idx].add(drop_idx)
        else:
            assert unit_type == 2 and seg_id == WHERE_VAL  # re-decode
            str_idxes = seg[0][3][0]
            self.avoid_items[dec_seq_idx].add(tuple(str_idxes))

        return dec_seq[:dec_seq_idx]

    def decode_revised_structure(self, seg, pointer, hyp, input_item, bool_verbal=False):
        seg_id = seg[0][0]
        assert seg_id != SELECT_COL, "Error: Cannot remove all SELECT_COL!"

        if seg_id == WHERE_COL:
            print("## WARNING: %s structure changes!" % seg_id)
            dec_seq_idx = seg[0][-1]
            dec_seq_item = list(hyp.dec_seq[dec_seq_idx])
            hyp.dec_seq[dec_seq_idx] = (dec_seq_item[0], 0, [])
            hyp = self.decode(input_item, dec_beam_size=1, dec_prefix=hyp.dec_seq[:(dec_seq_idx + 1)],
                              avoid_items=self.avoid_items,
                              confirmed_items=self.confirmed_items,
                              bool_verbal=bool_verbal)[0]
            return pointer, hyp
        else:
            return pointer + 1, hyp
