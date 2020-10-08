from prettytable import PrettyTable
from MISP_SQL.environment import UserSim, RealUser as BaseRealUser, ErrorEvaluator as BaseErrorEvaluator, \
    GoldUserSim as BaseGoldUserSim
from MISP_SQL.utils import *


class RealUser(BaseRealUser):
    def __init__(self, error_evaluator, tables, bool_undo=True):
        BaseRealUser.__init__(self, error_evaluator, bool_undo=bool_undo)

        self.tables = tables

    def show_table(self, table_id):
        table = self.tables[table_id]
        print(bcolors.BLUE + bcolors.BOLD + "{}".format(table['header']) + bcolors.ENDC)

        # basic information
        '''
        print(bcolors.BOLD + "Example rows:" + bcolors.ENDC)
        x = PrettyTable()
        x.field_names = table['header']
        for row in table['rows'][:3]:
            x.add_row(row)
        print(x)

        print("\n" + bcolors.BOLD + "Additional info. about this table: " + bcolors.ENDC)
        for key, key_alias in zip(["page_title", "section_title", "caption"],
                                  ["Page Title", "Section Title", "Table Caption"]):
            if key in table:
                print("{}: {}".format(key_alias, table[key]))
        print("")
        '''
        # for key, key_alias in zip(["page_title", "section_title", "caption"],
        #                           ["Page Title", "Section Title", "Table Caption"]):
        #     if key in table:
        #         print("{}: {}".format(key_alias, table[key]))
        #
        # print("\n")
        #x = PrettyTable()
        #x.field_names = table['header']
        #print(bcolors.BLUE + bcolors.BOLD + "{}".format(table['header']) + bcolors.ENDC)
        for c,row in enumerate(table['rows']):
            #x.add_row(row)
            if c == 1 or c == 2:
                print(row)
            if c > 2:
                break

        #print(x)

        #print(bcolors.BLUE + bcolors.BOLD + "{}".format(table['header']) + bcolors.ENDC)


class ErrorEvaluator(BaseErrorEvaluator):
    def __init__(self):
        BaseErrorEvaluator.__init__(self)

    def compare(self, g_sql, start_idx, tag_seq, bool_return_true_selections=False,
                bool_return_true_semantic_units=False):
        # g_sql should look like {"{"sel":3,"conds":[[5,0,"Butler CC (KS)"]],"agg":0,
        # "g_wvi": [[st1,end1],[st2, end2],..]} # g_wvi is added from raw data
        lower_cased_conds = []
        for (g_col_idx, g_op_idx, g_val_str), st_end in zip(g_sql['conds'], g_sql["g_wvi"]):
            g_val_str = str(g_val_str).lower()
            lower_cased_conds.append((g_col_idx, g_op_idx, g_val_str, st_end))

        eval_output, true_selections, true_semantic_units = [], [], []
        idx = start_idx
        while idx < len(tag_seq):
            semantic_tag = tag_seq[idx][0]
            if semantic_tag == OUTSIDE:
                eval_output.append(None)
                true_selections.append(None)
                true_semantic_units.append(None)
                idx += 1

            elif semantic_tag == SELECT_COL:
                eval_output.append(tag_seq[idx][1][-1] == g_sql["sel"])
                true_selections.append([g_sql["sel"]])

                new_su = list(tag_seq[idx])
                new_su[1] = (None, None, g_sql["sel"])
                true_semantic_units.append([tuple(new_su)])

                idx += 1

            elif semantic_tag == SELECT_AGG:
                col_item, agg_item = tag_seq[idx][1:3]
                col_idx = col_item[-1]
                agg_idx = agg_item[-1]

                eval_output.append(agg_idx == g_sql['agg']) # TODO: associate with sel?
                true_selections.append([(col_idx, g_sql['agg'])])

                new_su = list(tag_seq[idx])
                new_su[2] = (None, g_sql['agg'])
                true_semantic_units.append([tuple(new_su)])

                idx += 1

            elif semantic_tag == WHERE_COL:
                col_idx = tag_seq[idx][1][-1]
                eval_output.append(col_idx in set([col for col, _, _ in g_sql['conds']]))
                true_selections.append([col for col, _, _ in g_sql['conds']])

                _true_semantic_units = []
                for true_col_idx in true_selections[-1]:
                    new_su = list(tag_seq[idx])
                    new_su[1] = (None, None, true_col_idx)
                    _true_semantic_units.append(tuple(new_su))
                true_semantic_units.append(_true_semantic_units)

                idx += 1

            elif semantic_tag == WHERE_OP:
                (col_item,), op_item = tag_seq[idx][1:3]
                col_idx = col_item[-1]
                op_idx = op_item[-1]
                true_col_op = [(col, op) for col, op, _ in g_sql['conds']]
                eval_output.append((col_idx, op_idx) in set(true_col_op))
                true_selections.append(true_col_op)

                bool_found_col = False
                for true_col_idx, true_op_idx in true_col_op:
                    if col_idx == true_col_idx:
                        new_su = list(tag_seq[idx])
                        new_su[2] = (None, true_op_idx)
                        true_semantic_units.append([tuple(new_su)])
                        bool_found_col = True
                        break
                if not bool_found_col:
                    true_semantic_units.append(None)

                idx += 1

            elif semantic_tag == WHERE_VAL:
                (col_item,), op_item, val_item = tag_seq[idx][1:4]
                col_idx = col_item[-1]
                op_idx = op_item[-1]
                val_str = val_item[-1].lower()
                lower_cased_conds_str = [(true_col, true_op, true_val_str)
                                          for (true_col, true_op, true_val_str, _) in lower_cased_conds]
                eval_output.append((col_idx, op_idx, val_str) in lower_cased_conds_str)
                true_selections.append(lower_cased_conds_str)

                bool_found_col = False
                for true_col_idx, true_op_idx, true_val_str, true_val_st_end in lower_cased_conds:
                    if true_col_idx == col_idx and true_op_idx == op_idx:
                        new_su = list(tag_seq[idx])
                        new_su[3] = (true_val_st_end[0], true_val_st_end[1], true_val_str)
                        true_semantic_units.append([tuple(new_su)])
                        bool_found_col = True
                        break
                if not bool_found_col:
                    true_semantic_units.append(None)

                idx += 1
            else:
                raise Exception("Invalid semantic_tag {} in semantic unit {}".format(semantic_tag, tag_seq[idx]))

        return_items = [idx, eval_output]
        if bool_return_true_selections:
            return_items.append(true_selections)
        if bool_return_true_semantic_units:
            return_items.append(true_semantic_units)

        return tuple(return_items)


class GoldUserSim(BaseGoldUserSim):
    def __init__(self, error_evaluator, bool_structure_question=False):
        BaseGoldUserSim.__init__(self, error_evaluator)
        self.bool_structure_question = bool_structure_question

    def get_gold_selection(self, pointer):
        pointer_truth = self.true_semantic_units[pointer]  # ground-truth decision
        old_su = self.tag_seq[pointer]
        semantic_tag = old_su[0]
        old_dec_item = self.dec_seq[old_su[-1]]
        gold_semantic_units, gold_dec_items = [], []

        if pointer_truth is not None:
            gold_semantic_units.extend(pointer_truth)
            for su in gold_semantic_units:
                if semantic_tag == SELECT_COL:
                    new_dec_item = list(old_dec_item)
                    new_dec_item[-1] = su[1][-1]
                    gold_dec_items.append(tuple(new_dec_item))
                elif semantic_tag == SELECT_AGG:
                    new_dec_item = list(old_dec_item)
                    new_dec_item[-1] = su[2][-1]
                    gold_dec_items.append(tuple(new_dec_item))
                elif semantic_tag == WHERE_COL:
                    gold_dec_items.append(None)
                elif semantic_tag == WHERE_OP:
                    new_dec_item = list(old_dec_item)
                    new_dec_item[-1] = su[2][-1]
                    gold_dec_items.append(tuple(new_dec_item))
                else:
                    new_dec_item = list(old_dec_item)
                    new_dec_item[-2] = su[3][0]
                    new_dec_item[-1] = su[3][1]
                    gold_dec_items.append(tuple(new_dec_item))

        print("Gold semantic units: %s." % str(gold_semantic_units))
        print("Gold dec_items: %s." % str(gold_dec_items))

        if len(gold_semantic_units):
            selections = [choice + 1 for choice in range(len(gold_semantic_units))]
            sel_none_of_above = len(gold_semantic_units) + 1
        elif self.bool_structure_question and semantic_tag == WHERE_COL:
            sel_none_of_above = 1
            selections = [sel_none_of_above + 1] # invalid structure
        else:
            sel_none_of_above = 1
            selections = [sel_none_of_above]
        print("Gold user selections ('none of above' = %d): %s.\n" % (sel_none_of_above, str(selections)))

        return gold_semantic_units, gold_dec_items, sel_none_of_above, selections

