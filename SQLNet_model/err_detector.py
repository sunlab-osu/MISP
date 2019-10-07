# error detector

from interaction_framework.err_detector import ErrorDetectorEvaluator, ErrorDetectorProbability, ErrorDetectorBayDropout
from interaction_framework.question_gen import OUTSIDE, SELECT_COL, SELECT_AGG, WHERE_COL, WHERE_OP, WHERE_VAL


AGG_OPS = ['None', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
COND_OPS = ['=', '>', '<', 'OP']


class ErrorDetectorEvaluatorSQLNet(ErrorDetectorEvaluator):
    def __init__(self):
        ErrorDetectorEvaluator.__init__(self)

    def compare(self, g_sql, start_idx, BIO_history):
        # g_sql should look like {"{"sel":3,"conds":[[5,0,"Butler CC (KS)"]],"agg":0}
        lower_cased_conds = []
        for (g_col_idx, g_op_idx, g_val_str) in g_sql['conds']:
            g_val_str = str(g_val_str).lower()
            lower_cased_conds.append((g_col_idx, g_op_idx, g_val_str))

        eval_output = []
        idx = start_idx
        while idx < len(BIO_history):
            seg_id = BIO_history[idx][0]
            if seg_id == OUTSIDE:
                eval_output.append(None)
                idx += 1

            elif seg_id == SELECT_COL:
                eval_output.append(BIO_history[idx][1][-1] == g_sql["sel"])
                idx += 1

            elif seg_id == SELECT_AGG:
                col_item, agg_item = BIO_history[idx][1:3]
                col_idx = col_item[-1]
                if agg_item == 'none_agg':
                    agg_idx = 0
                else:
                    agg_idx = AGG_OPS.index(agg_item)

                eval_output.append(agg_idx == g_sql['agg']) # TODO: associate with sel?
                idx += 1

            elif seg_id == WHERE_COL:
                col_idx = BIO_history[idx][1][-1]
                eval_output.append(col_idx in set([col for col, _, _ in g_sql['conds']]))
                idx += 1

            elif seg_id == WHERE_OP:
                (col_item,), op_item = BIO_history[idx][1:3]
                col_idx = col_item[-1]
                op_idx = COND_OPS.index(op_item)
                eval_output.append((col_idx, op_idx) in set([(col, op) for col, op, _ in g_sql['conds']]))
                idx += 1

            elif seg_id == WHERE_VAL:
                (col_item,), op_item, val_item = BIO_history[idx][1:4]
                col_idx = col_item[-1]
                op_idx = COND_OPS.index(op_item)
                val_str = val_item[-1].lower()
                eval_output.append((col_idx, op_idx, val_str) in lower_cased_conds)
                idx += 1

            else:
                raise Exception("Invalid seg_id {} in seg {}".format(seg_id, BIO_history[idx]))

        return idx, eval_output