from MISP_SQL.environment import ErrorEvaluator as BaseErrorEvaluator, UserSim, RealUser as BaseRealUser
from MISP_SQL.utils import *


class RealUser(BaseRealUser):
    def __init__(self, err_evaluator, tables):
        BaseRealUser.__init__(self, err_evaluator)

        self.tables = tables

    def show_table(self, table_id):
        table = self.tables[table_id]

        # # basic information
        # for key, key_alias in zip(["page_title", "section_title", "caption"],
        #                           ["Page Title", "Section Title", "Table Caption"]):
        #     if key in table:
        #         print("{}: {}".format(key_alias, table[key]))

        # print("\n")
        # x = PrettyTable()
        # x.field_names = table['header']
        # for row in table['rows']:
        #     x.add_row(row)
        # print(x)

        print(bcolors.BLUE + bcolors.BOLD + "{}".format(table['header']) + bcolors.ENDC)


class ErrorEvaluator(BaseErrorEvaluator):
    def __init__(self):
        BaseErrorEvaluator.__init__(self)

    def compare(self, g_sql, start_idx, tag_seq, bool_return_truths=False):
        # g_sql should look like {"{"sel":3,"conds":[[5,0,"Butler CC (KS)"]],"agg":0}
        lower_cased_conds = []
        for (g_col_idx, g_op_idx, g_val_str) in g_sql['conds']:
            g_val_str = str(g_val_str).lower()
            lower_cased_conds.append((g_col_idx, g_op_idx, g_val_str))

        eval_output, true_tag_seq = [], []
        idx = start_idx
        while idx < len(tag_seq):
            seg_id = tag_seq[idx][0]
            if seg_id == OUTSIDE:
                eval_output.append(None)
                true_tag_seq.append(None)
                idx += 1

            elif seg_id == SELECT_COL:
                eval_output.append(tag_seq[idx][1][-1] == g_sql["sel"])
                true_tag_seq.append([g_sql["sel"]])
                idx += 1

            elif seg_id == SELECT_AGG:
                col_item, agg_item = tag_seq[idx][1:3]
                col_idx = col_item[-1]
                agg_idx = agg_item[-1]

                eval_output.append(agg_idx == g_sql['agg']) # TODO: associate with sel?
                true_tag_seq.append([(col_idx, g_sql['agg'])])
                idx += 1

            elif seg_id == WHERE_COL:
                col_idx = tag_seq[idx][1][-1]
                eval_output.append(col_idx in set([col for col, _, _ in g_sql['conds']]))
                true_tag_seq.append([col for col, _, _ in g_sql['conds']])
                idx += 1

            elif seg_id == WHERE_OP:
                (col_item,), op_item = tag_seq[idx][1:3]
                col_idx = col_item[-1]
                op_idx = op_item[-1]
                eval_output.append((col_idx, op_idx) in set([(col, op) for col, op, _ in g_sql['conds']]))
                true_tag_seq.append([(col, op) for col, op, _ in g_sql['conds']])
                idx += 1

            elif seg_id == WHERE_VAL:
                (col_item,), op_item, val_item = tag_seq[idx][1:4]
                col_idx = col_item[-1]
                op_idx = op_item[-1]
                val_str = val_item[-1].lower()
                eval_output.append((col_idx, op_idx, val_str) in lower_cased_conds)
                true_tag_seq.append(lower_cased_conds)
                idx += 1

            else:
                raise Exception("Invalid seg_id {} in seg {}".format(seg_id, tag_seq[idx]))

        if bool_return_truths:
            return idx, eval_output, true_tag_seq
        
        return idx, eval_output


