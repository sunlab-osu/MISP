from MISP_SQL.utils import SELECT_COL, SELECT_AGG, WHERE_COL, WHERE_OP, WHERE_ROOT_TERM, ANDOR, \
    GROUP_COL, GROUP_NHAV, HAV_COL, HAV_AGG, HAV_OP, HAV_ROOT_TERM, ORDER_COL, ORDER_AGG, ORDER_DESC_ASC_LIMIT, \
    IUEN, OUTSIDE, END_NESTED, bcolors
from MISP_SQL.environment import ErrorEvaluator as BaseErrorEvaluator, UserSim, RealUser as BaseRealUser
import evaluation
from .supermodel import SQL_OPS, NEW_WHERE_OPS
from collections import defaultdict


class RealUser(BaseRealUser):
    def __init__(self, error_evaluator, tables):
        BaseRealUser.__init__(self, error_evaluator)

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

        table2columns = defaultdict(list)
        for tab_id, column in table['column_names_original']:
            if tab_id >= 0:
                table2columns[tab_id].append(column)

        for tab_id in range(len(table2columns)):
            print(bcolors.BOLD + "Table %d " % (tab_id + 1) + bcolors.YELLOW + table['table_names_original'][tab_id] + bcolors.ENDC)
            print(bcolors.BLUE + bcolors.BOLD + "{}\n".format(table2columns[tab_id]) + bcolors.ENDC)


class ErrorEvaluator(BaseErrorEvaluator):
    def __init__(self):
        BaseErrorEvaluator.__init__(self)

    def helper_find_closest_fw(self, history, start_idx, tgt_name=None, tgt_id=None):
        skip_root = []
        idx = start_idx
        while idx < len(history):
            if len(skip_root) > 0:
                if END_NESTED in history[idx]:
                    _ = skip_root.pop()
                idx += 1
            else:
                if (tgt_name is not None and tgt_name in history[idx]) or \
                        (tgt_id is not None and history[idx][0] == tgt_id):
                    return idx
                elif "root" in history[idx]:
                    skip_root.append(idx)
                    idx += 1
                else:
                    idx += 1

        return -1

    def parse_select(self, sql_select):
        col_agg_pairs = []
        for (agg_id, val_unit) in sql_select[1]:
            unit_op, col_unit1, col_unit2 = val_unit
            assert unit_op==0 and col_unit2 is None # syntaxSQL does not support "col1-col2"
            _, col_idx, _ = col_unit1
            col_agg_pairs.append((col_idx, agg_id))
        return col_agg_pairs

    def parse_where_having(self, sql_where_having):
        col_op_root = []
        and_or = None
        for item in sql_where_having:
            if isinstance(item, str) or isinstance(item, unicode):#"and"/"or"
                if and_or is None: # we take the first and/or decision
                    and_or = item
            else:
                not_op, op_idx, val_unit, val1, val2 = item
                op_name = evaluation.WHERE_OPS[op_idx]
                if not_op:
                    op_name = 'not ' + op_name
                _, col_unit1, _ = val_unit
                agg_idx, col_idx, _ = col_unit1
                col_op_root.append(((col_idx, agg_idx), NEW_WHERE_OPS.index(op_name), (val1, val2)))

        return col_op_root, and_or

    def parse_group(self, sql_groupBy):
        cols = []
        for col_unit in sql_groupBy:
            _, col_idx, _ = col_unit
            cols.append(col_idx)
        return cols

    def parse_orderBy_limit(self, sql_orderBy, sql_limit):
        bool_limit = sql_limit is not None
        asc_desc = sql_orderBy[0]
        col_agg_other = []
        for val_unit in sql_orderBy[1]:
            _, col_unit, _ = val_unit
            agg_idx, col_idx, _ = col_unit
            col_agg_other.append(((col_idx, agg_idx), (asc_desc, bool_limit)))
        return col_agg_other

    def compare(self, g_sql, start_idx, tag_seq, bool_return_truths=False):
        eval_output, true_tag_seq = [], []
        idx = start_idx
        while idx < len(tag_seq):
            seg_id = tag_seq[idx][0]
            if seg_id == OUTSIDE:
                eval_output.append(None)
                true_tag_seq.append(None)
                idx += 1

            elif seg_id == IUEN:
                truth = 'none'
                for cand in ['intersect', 'union', 'except']:
                    if g_sql[cand] is not None:
                        truth = cand
                        break

                true_tag_seq.append([SQL_OPS.index(truth)])  # IUEN id
                if truth == 'none':
                    if tag_seq[idx][1][0] == 'none':
                        eval_output.append(True)
                        idx += 1
                    else:
                        eval_output.append(False)
                        idx += 1
                        if idx == len(tag_seq): break # check partial end

                        assert "root" in tag_seq[idx] and tag_seq[idx][0] == OUTSIDE
                        eval_output.append(None)
                        true_tag_seq.append(None)
                        idx += 1
                        if idx == len(tag_seq): break  # check partial end

                        end_nested_idx = self.helper_find_closest_fw(tag_seq, idx, tgt_name=END_NESTED)
                        # assert end_nested_idx != -1
                        if end_nested_idx == -1:
                            end_nested_idx = None
                        idx, main_sql_eval, main_sql_true_tag_seq = self.compare(
                            g_sql, idx, tag_seq[:end_nested_idx], bool_return_truths=True)
                        # assert idx == end_nested_idx
                        eval_output.extend(main_sql_eval)
                        true_tag_seq.extend(main_sql_true_tag_seq)

                        # for remaining part (nested sql), eval to False
                        while idx < len(tag_seq):
                            eval_output.append(False if tag_seq[idx][0] != OUTSIDE else None)
                            true_tag_seq.append(None)
                            idx += 1
                else:
                    if tag_seq[idx][1][0] == 'none':
                        eval_output.append(False)
                        idx += 1

                    elif tag_seq[idx][1][0] == truth:
                        eval_output.append(True)
                        idx += 1
                        if idx == len(tag_seq): break  # check partial end

                        assert "root" in tag_seq[idx] and tag_seq[idx][0] == OUTSIDE
                        eval_output.append(None)
                        true_tag_seq.append(None)
                        idx += 1
                        if idx == len(tag_seq): break  # check partial end

                        end_nested_idx = self.helper_find_closest_fw(tag_seq, idx, tgt_name=END_NESTED)
                        # assert end_nested_idx != -1
                        if end_nested_idx == -1:
                            end_nested_idx = None
                        idx, main_sql_eval, main_sql_true_tag_seq = self.compare(
                            g_sql, idx, tag_seq[:end_nested_idx], bool_return_truths=True)
                        # assert idx == end_nested_idx
                        eval_output.extend(main_sql_eval)
                        true_tag_seq.extend(main_sql_true_tag_seq)
                        if idx == len(tag_seq): break  # check partial end

                        idx, nested_sql_eval, nested_sql_true_tag_seq = self.compare(
                            g_sql[truth], idx, tag_seq, bool_return_truths=True)
                        eval_output.extend(nested_sql_eval)
                        true_tag_seq.extend(nested_sql_true_tag_seq)

                    else:
                        eval_output.append(False)
                        idx += 1
                        if idx == len(tag_seq): break  # check partial end

                        assert "root" in tag_seq[idx] and tag_seq[idx][0] == OUTSIDE
                        eval_output.append(None)
                        true_tag_seq.append(None)
                        idx += 1
                        if idx == len(tag_seq): break  # check partial end

                        end_nested_idx = self.helper_find_closest_fw(tag_seq, idx, tgt_name=END_NESTED)
                        # assert end_nested_idx != -1
                        if end_nested_idx == -1:
                            end_nested_idx = None
                        idx, main_sql_eval, main_sql_true_tag_seq = self.compare(
                            g_sql, idx, tag_seq[:end_nested_idx], bool_return_truths=True)
                        # assert idx == end_nested_idx
                        eval_output.extend(main_sql_eval)
                        true_tag_seq.extend(main_sql_true_tag_seq)

                        while idx < len(tag_seq):
                            eval_output.append(False if tag_seq[idx][0] != OUTSIDE else None)
                            true_tag_seq.append(None)
                            idx += 1

            elif seg_id == SELECT_COL:
                select_col_agg_pairs = self.parse_select(g_sql['select'])
                while idx < len(tag_seq) and tag_seq[idx][0] == SELECT_COL:
                    col_idx = tag_seq[idx][1][-1]
                    eval_output.append(col_idx in set([col for col,_ in select_col_agg_pairs]))
                    true_tag_seq.append([col for col,_ in select_col_agg_pairs])
                    idx += 1
                if idx == len(tag_seq): break  # check partial end

                assert tag_seq[idx][0] == SELECT_AGG
                while idx < len(tag_seq) and tag_seq[idx][0] == SELECT_AGG:
                    col_idx = tag_seq[idx][1][-1]
                    agg_idx = tag_seq[idx][2][-1]
                    eval_output.append((col_idx, agg_idx) in select_col_agg_pairs) #evaluation.AGG_OPS.index(agg_name)
                    true_tag_seq.append(select_col_agg_pairs)
                    idx += 1

            elif seg_id == WHERE_COL:
                if len(g_sql['where']) == 0:
                    while idx < len(tag_seq) and tag_seq[idx][0] in {WHERE_COL, WHERE_OP, WHERE_ROOT_TERM, ANDOR}:
                        if tag_seq[idx][0] == WHERE_ROOT_TERM and tag_seq[idx][3] == 'root':
                            eval_output.append(False)
                            true_tag_seq.append(None)
                            idx += 1
                            end_nested_idx = self.helper_find_closest_fw(tag_seq, idx, tgt_name=END_NESTED)
                            if end_nested_idx == -1: end_nested_idx = len(tag_seq) - 1
                            # eval.extend([False] * (end_nested_idx - idx + 1))
                            eval_output.extend([False if tag_seq[ii][0] != OUTSIDE else None
                                                for ii in range(idx, (end_nested_idx + 1))])
                            true_tag_seq.extend([None] * (end_nested_idx + 1 - idx))
                            idx = end_nested_idx + 1
                        else:
                            eval_output.append(False)
                            true_tag_seq.append(None)
                            idx += 1

                else:
                    col_op_root, and_or = self.parse_where_having(g_sql['where'])

                    while idx < len(tag_seq) and tag_seq[idx][0] == WHERE_COL:
                        col_idx = tag_seq[idx][1][-1]
                        eval_output.append(col_idx in set([col for (col,_),_,_ in col_op_root]))
                        true_tag_seq.append([col for (col,_),_,_ in col_op_root])
                        idx += 1
                    if idx == len(tag_seq): break  # check partial end

                    if tag_seq[idx][0] == ANDOR:
                        eval_output.append(tag_seq[idx][1] == and_or)
                        true_tag_seq.append([and_or])
                        idx += 1
                    if idx == len(tag_seq): break  # check partial end

                    assert tag_seq[idx][0] == WHERE_OP

                    while idx < len(tag_seq) and tag_seq[idx][0] == WHERE_OP: # for all WHERE_COLs
                        while idx < len(tag_seq) and tag_seq[idx][0] == WHERE_OP: # for each col_idx
                           col_idx = tag_seq[idx][1][0][-1]
                           op_name, op_idx = tag_seq[idx][2]
                           eval_output.append((col_idx, op_idx) in set([(col, op) for (col,_),op,_ in col_op_root]))
                           true_tag_seq.append([(col, op) for (col,_),op,_ in col_op_root])
                           idx += 1
                        while idx < len(tag_seq) and tag_seq[idx][0] == WHERE_ROOT_TERM:
                            col_idx = tag_seq[idx][1][0][-1]
                            op_name, op_idx = tag_seq[idx][2]
                            root_term = tag_seq[idx][3]
                            bool_matched_col_op = False
                            for (col,_), op, (val1, val2) in col_op_root:
                                if col == col_idx and op == op_idx:
                                    if not isinstance(val1, dict) and (val2 is None or not isinstance(val2, dict)):
                                        truth = 'terminal'
                                        true_tag_seq.append([truth])
                                        if root_term == truth:
                                            eval_output.append(True)
                                            idx += 1
                                        else:
                                            eval_output.append(False)
                                            idx += 1
                                            # consider terms associated with the nested root as wrong
                                            end_nested_idx = self.helper_find_closest_fw(tag_seq, idx, tgt_name=END_NESTED)
                                            if end_nested_idx == -1: end_nested_idx = len(tag_seq) - 1
                                            # eval.extend([False] * (end_nested_idx - idx + 1))
                                            eval_output.extend([False if tag_seq[ii][0] != OUTSIDE else None for ii in
                                                                range(idx, (end_nested_idx + 1))])
                                            true_tag_seq.extend([None] * (end_nested_idx + 1 - idx))
                                            idx = end_nested_idx + 1
                                    else:
                                        assert isinstance(val1, dict) and val2 is None
                                        truth = 'root'
                                        true_tag_seq.append([truth])
                                        if root_term != truth: #root_term = terminal
                                            eval_output.append(False)
                                            idx += 1
                                        else:
                                            eval_output.append(True)
                                            idx += 1
                                            # evaluate nested component
                                            end_nested_idx = self.helper_find_closest_fw(tag_seq, idx, tgt_name=END_NESTED)
                                            if end_nested_idx == -1: end_nested_idx = len(tag_seq) - 1
                                            idx, nested_eval, nested_true_tag_seq = self.compare(
                                                val1, idx, tag_seq[:(end_nested_idx + 1)], bool_return_truths=True)
                                            eval_output.extend(nested_eval)
                                            true_tag_seq.extend(nested_true_tag_seq)
                                    bool_matched_col_op = True
                                    break

                            if not bool_matched_col_op:
                                eval_output.append(False)
                                true_tag_seq.append(None)
                                idx += 1
                                if root_term == 'root':
                                    # consider terms associated with the nested root as wrong
                                    end_nested_idx = self.helper_find_closest_fw(tag_seq, idx, tgt_name=END_NESTED)
                                    if end_nested_idx == -1: end_nested_idx = len(tag_seq) - 1
                                    # eval.extend([False] * (end_nested_idx - idx + 1))
                                    eval_output.extend([False if tag_seq[ii][0] != OUTSIDE else None for ii in
                                                        range(idx, (end_nested_idx + 1))])
                                    true_tag_seq.extend([None] * (end_nested_idx + 1 - idx))
                                    idx = end_nested_idx + 1

            elif seg_id == GROUP_COL:
                if len(g_sql['groupBy']) == 0:
                    groupBy_cols = []
                else:
                    groupBy_cols = set(self.parse_group(g_sql['groupBy']))
                while idx < len(tag_seq) and tag_seq[idx][0] == GROUP_COL:
                    col_idx = tag_seq[idx][1][-1]
                    eval_output.append(col_idx in groupBy_cols)
                    true_tag_seq.append(groupBy_cols)
                    idx += 1

            elif seg_id == GROUP_NHAV:
                eval_output.append(len(g_sql['having']) == 0)
                true_tag_seq.append(None)
                idx += 1

            elif seg_id == HAV_COL:
                if len(g_sql['having']) == 0:
                    eval_output.append(False)
                    true_tag_seq.append(None)
                    idx += 1
                    while idx < len(tag_seq) and tag_seq[idx][0] in {HAV_COL, HAV_AGG, HAV_OP, HAV_ROOT_TERM}:
                        eval_output.append(False)
                        true_tag_seq.append(None)
                        idx += 1

                else:
                    col_op_root, and_or = self.parse_where_having(g_sql['having'])

                    while idx < len(tag_seq) and tag_seq[idx][0] == HAV_COL:
                        col_idx = tag_seq[idx][1][-1]
                        eval_output.append(col_idx in set([col for (col,_), _, _ in col_op_root]))
                        true_tag_seq.append([col for (col,_), _, _ in col_op_root])
                        idx += 1
                    if idx == len(tag_seq): break  # check partial end

                    assert tag_seq[idx][0] == HAV_AGG

                    while idx < len(tag_seq) and tag_seq[idx][0] == HAV_AGG:  # for all HAV_COLs
                        tab_col_item, (agg_name, agg_idx) = tag_seq[idx][1:3]
                        col_idx = tab_col_item[-1]
                        eval_output.append((col_idx, agg_idx) in set([col_agg for col_agg, _, _ in col_op_root]))
                        true_tag_seq.append([col_agg for col_agg, _, _ in col_op_root])
                        idx += 1

                        while idx < len(tag_seq) and tag_seq[idx][0] == HAV_OP:  # for each col_idx
                            col_idx = tag_seq[idx][1][0][-1]
                            agg_name, agg_idx = tag_seq[idx][1][1]
                            op_name, op_idx = tag_seq[idx][2]
                            eval_output.append(((col_idx, agg_idx), op_idx) in
                                               set([(col_agg, op) for col_agg, op, _ in col_op_root]))
                            true_tag_seq.append([(col_agg, op) for col_agg, op, _ in col_op_root])
                            idx += 1

                        while idx < len(tag_seq) and tag_seq[idx][0] == HAV_ROOT_TERM:
                            col_idx = tag_seq[idx][1][0][-1]
                            agg_name, agg_idx = tag_seq[idx][1][1]
                            op_name, op_idx = tag_seq[idx][2]
                            root_term = tag_seq[idx][3]
                            bool_matched_col_op = False
                            for col_agg, op, (val1, val2) in col_op_root:
                                if (col_idx, agg_idx) == col_agg and op == op_idx:
                                    if not isinstance(val1, dict) and (val2 is None or not isinstance(val2, dict)):
                                        truth = 'terminal'
                                        true_tag_seq.append(['terminal'])
                                        if root_term == truth:
                                            eval_output.append(True)
                                            idx += 1
                                        else:
                                            eval_output.append(False)
                                            idx += 1
                                            # consider terms associated with the nested root as wrong
                                            end_nested_idx = self.helper_find_closest_fw(tag_seq, idx, tgt_name=END_NESTED)
                                            if end_nested_idx == -1: end_nested_idx = len(tag_seq) - 1
                                            # eval.extend([False] * (end_nested_idx - idx + 1))
                                            eval_output.extend([False if tag_seq[ii][0] != OUTSIDE else None for ii in
                                                                range(idx, (end_nested_idx + 1))])
                                            true_tag_seq.extend([None] * (end_nested_idx + 1 - idx))
                                            idx = end_nested_idx + 1
                                    else:
                                        assert isinstance(val1, dict) and val2 is None
                                        truth = 'root'
                                        true_tag_seq.append(['root'])
                                        if root_term != truth:  # root_term = terminal
                                            eval_output.append(False)
                                            idx += 1
                                        else:
                                            eval_output.append(True)
                                            idx += 1
                                            # evaluate nested component
                                            end_nested_idx = self.helper_find_closest_fw(tag_seq, idx,
                                                                                         tgt_name=END_NESTED)
                                            if end_nested_idx == -1: end_nested_idx = len(tag_seq) - 1
                                            idx, nested_eval, nested_true_tag_seq = self.compare(
                                                val1, idx, tag_seq[:(end_nested_idx + 1)], bool_return_truths=True)
                                            eval_output.extend(nested_eval)
                                            true_tag_seq.extend(nested_true_tag_seq)
                                    bool_matched_col_op = True
                                    break

                            if not bool_matched_col_op: # cannot find matched (col_agg, op)
                                eval_output.append(False)
                                true_tag_seq.append(None)
                                idx += 1
                                if root_term == 'root':
                                    # consider terms associated with the nested root as wrong
                                    end_nested_idx = self.helper_find_closest_fw(tag_seq, idx, tgt_name=END_NESTED)
                                    if end_nested_idx == -1: end_nested_idx = len(tag_seq) - 1
                                    # eval.extend([False] * (end_nested_idx - idx + 1))
                                    eval_output.extend([False if tag_seq[ii][0] != OUTSIDE else None for ii in
                                                        range(idx, (end_nested_idx + 1))])
                                    true_tag_seq.extend([None] * (end_nested_idx + 1 - idx))
                                    idx = end_nested_idx + 1

            elif seg_id == ORDER_COL:
                if len(g_sql['orderBy']) == 0:
                    while idx < len(tag_seq) and tag_seq[idx][0] in {ORDER_COL, ORDER_AGG, ORDER_DESC_ASC_LIMIT}:
                        eval_output.append(False)
                        true_tag_seq.append(None)
                        idx += 1

                else:
                    col_agg_other = self.parse_orderBy_limit(g_sql['orderBy'], g_sql['limit'])
                    while idx < len(tag_seq) and tag_seq[idx][0] == ORDER_COL:
                        col_idx = tag_seq[idx][1][-1]
                        eval_output.append(col_idx in set([col for (col,_), _ in col_agg_other]))
                        true_tag_seq.append([col for (col,_), _ in col_agg_other])
                        idx += 1
                    if idx == len(tag_seq): break  # check partial end

                    assert tag_seq[idx][0] == ORDER_AGG
                    while idx < len(tag_seq) and tag_seq[idx][0] == ORDER_AGG:
                        # ORDER_AGG
                        col_idx = tag_seq[idx][1][-1]
                        agg_name, agg_idx = tag_seq[idx][2]
                        eval_output.append((col_idx, agg_idx) in set([col_agg for col_agg, _ in col_agg_other]))
                        true_tag_seq.append([col_agg for col_agg, _ in col_agg_other])
                        idx += 1
                        if idx == len(tag_seq): break  # check partial end

                        # ORDER_DESC_ASC_LIMIT
                        assert tag_seq[idx][0] == ORDER_DESC_ASC_LIMIT
                        col_idx = tag_seq[idx][1][0][-1]
                        agg_name, agg_idx = tag_seq[idx][1][1]
                        desc_asc_limit = tag_seq[idx][2]
                        eval_output.append(((col_idx, agg_idx), desc_asc_limit) in col_agg_other)
                        true_tag_seq.append(col_agg_other)
                        idx += 1

            else:
                raise Exception("Invalid id {} at idx {} in history:\n{}".format(seg_id, idx, tag_seq))

        if bool_return_truths:
            return idx, eval_output, true_tag_seq

        return idx, eval_output

