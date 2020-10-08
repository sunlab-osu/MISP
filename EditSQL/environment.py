from MISP_SQL.environment import ErrorEvaluator as BaseErrorEvaluator, UserSim as BaseUserSim, \
    RealUser as BaseRealUser, GoldUserSim as BaseGoldUserSim
from MISP_SQL.utils import SELECT_AGG_v2, WHERE_COL, WHERE_OP, WHERE_ROOT_TERM, GROUP_COL, HAV_AGG_v2, \
    HAV_OP_v2, HAV_ROOT_TERM_v2, ORDER_AGG_v2, ORDER_DESC_ASC, ORDER_LIMIT, IUEN_v2, OUTSIDE, END_NESTED
from EditSQL.eval_scripts.evaluation import WHERE_OPS, AGG_OPS
from user_study_utils import bcolors
from collections import defaultdict
import sqlite3

NEW_WHERE_OPS = ('=','>','<','>=','<=','!=','like','not in','in','between', 'not like')
NEW_SQL_OPS = ('none','intersect', 'union', 'except')


class RealUser(BaseRealUser):
    def __init__(self, error_evaluator, tables, db_path,
                 bool_structure_question=False):
        BaseRealUser.__init__(self, error_evaluator)

        self.tables = tables
        self.db_path = db_path
        self.bool_structure_question = bool_structure_question

    def show_table(self, table_id):
        schema = {} # a dict of {table_name: [column names, row1, row2, row3]}

        # load table content
        conn = sqlite3.connect(self.db_path + '{}/{}.sqlite'.format(table_id, table_id))
        cursor = conn.cursor()

        # fetch table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [str(table_name[0].lower()) for table_name in cursor.fetchall()]

        # fetch table info
        for table_name in tables:
            cursor.execute("PRAGMA table_info({})".format(table_name))
            schema[table_name] = [[str(col[1].lower()) for col in cursor.fetchall()]]
            count_row = 0
            for row in cursor.execute("SELECT * FROM {};".format(table_name)):
                schema[table_name].append(row)
                count_row += 1
                if count_row == 3:
                    break

        table = self.tables[table_id]

        table2columns = defaultdict(list)
        for tab_id, column in table['column_names_original']:
            if tab_id >= 0:
                table2columns[tab_id].append(column)

        for tab_id in range(len(table2columns)):
            print(bcolors.BOLD + "Table %d " % (tab_id + 1) + bcolors.YELLOW + table['table_names_original'][tab_id] + bcolors.ENDC)
            print(bcolors.BLUE + bcolors.BOLD + "{}\n".format(table2columns[tab_id]) + bcolors.ENDC)
            for i,row in enumerate(schema[table['table_names_original'][tab_id].lower()]):
                if i == 0:
                    continue
                print(row)
            print('\n')

    def get_selection(self, pointer, answer_sheet, sel_none_of_above):
        def answer_parsing(answer_str):
            selections = answer_str.split(", ")
            try:
                selections = [int(sel) for sel in selections]
            except:
                return None
            else:
                assert len(selections)
                if sel_none_of_above in selections:
                    assert len(selections) == 1 # mutual exclusive "none of the above"
                if self.bool_structure_question and (sel_none_of_above + 1) in selections:
                    assert len(selections) == 1 # mutual exclusive "invalid structure"
                return selections

        answer = input("Please enter the option id(s) delimited by comma ', ': ")
        selections = answer_parsing(answer)
        while selections is None:
            answer = input("Please enter the option id(s) delimited by comma ', ': ")
            selections = answer_parsing(answer)

        return selections


class ErrorEvaluator(BaseErrorEvaluator):
    def __init__(self):
        BaseErrorEvaluator.__init__(self)
        self.kw2asterisk = None
        self.base_vocab = None
        self.column_names_surface_form_to_id = None

    def _clear_meta(self):
        self.kw2asterisk = None
        self.base_vocab = None
        self.column_names_surface_form_to_id = None

    def _set_global_clause_asterisk(self, kw2asterisk):
        self.kw2asterisk = kw2asterisk

    def _set_global_vocab(self, base_vocab, column_names_surface_form_to_id):
        self.base_vocab = base_vocab # fix for all queries
        self.column_names_surface_form_to_id = column_names_surface_form_to_id # dynamic for each db

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
            # assert unit_op==0 and col_unit2 is None # we do not validate "col1-col2"
            _, col_idx, bool_distinct = col_unit1
            if col_idx == 0: #asterisk
                col_name = self.kw2asterisk['select']
                col_idx = self.column_names_surface_form_to_id[col_name] + len(self.base_vocab)
            else:
                col_idx += len(self.base_vocab)
            col_agg_pairs.append((col_idx, agg_id, bool_distinct))
        return col_agg_pairs

    def parse_where_having(self, sql_where_having):
        col_op_root = []
        and_or = None
        for item in sql_where_having:
            if isinstance(item, str):#"and"/"or" or isinstance(item, unicode)
                if and_or is None: # we take the first and/or decision
                    and_or = item
            else:
                not_op, op_idx, val_unit, val1, val2 = item
                op_name = WHERE_OPS[op_idx]
                if not_op:
                    op_name = 'not ' + op_name
                _, col_unit1, _ = val_unit
                agg_idx, col_idx, bool_distinct = col_unit1

                if col_idx == 0:  # asterisk
                    col_name = self.kw2asterisk['having'] # will not happen to where clauses
                    col_idx = self.column_names_surface_form_to_id[col_name] + len(self.base_vocab)
                else:
                    col_idx += len(self.base_vocab)

                col_op_root.append(((col_idx, agg_idx, bool_distinct),
                                    NEW_WHERE_OPS.index(op_name), (val1, val2)))

        return col_op_root, and_or

    def parse_group(self, sql_groupBy):
        cols = []
        for col_unit in sql_groupBy:
            _, col_idx, _ = col_unit
            assert col_idx > 0
            col_idx += len(self.base_vocab)
            cols.append(col_idx)
        return cols

    def parse_orderBy_limit(self, sql_orderBy, sql_limit):
        bool_limit = sql_limit is not None
        asc_desc = sql_orderBy[0]
        col_agg_desc_asc, col_agg_limit = [], []
        for val_unit in sql_orderBy[1]:
            _, col_unit, _ = val_unit
            agg_idx, col_idx, bool_distinct = col_unit

            if col_idx == 0: #asterisk
                col_name = self.kw2asterisk['order_by']
                col_idx = self.column_names_surface_form_to_id[col_name] + len(self.base_vocab)
            else:
                col_idx += len(self.base_vocab)

            col_agg_desc_asc.append(((col_idx, agg_idx, bool_distinct), asc_desc))
            col_agg_limit.append(((col_idx, agg_idx, bool_distinct), bool_limit))
        return col_agg_desc_asc, col_agg_limit

    def compare(self, g_sql, start_idx, tag_seq, bool_return_true_selections=False,
                bool_return_true_semantic_units=False):
        # Change log:
        # 01/28: Add table name check for * case. Associated semantic tags: SELECT/HAV/ORDER_AGG_v2,
        # HAV_OP_v2, HAV_ROOT_ITEM_v2, ORDER_DESC_ASC, ORDER_LIMIT.

        if "extracted_clause_asterisk" in g_sql:
            self._set_global_clause_asterisk(g_sql['extracted_clause_asterisk'])
        if "base_vocab" in g_sql:
            self._set_global_vocab(g_sql['base_vocab'], g_sql['column_names_surface_form_to_id'])

        eval_output, true_selections, true_semantic_units = [], [], []
        idx = start_idx
        while idx < len(tag_seq):
            semantic_tag = tag_seq[idx][0]
            if semantic_tag == OUTSIDE:
                eval_output.append(None)
                true_selections.append(None)
                true_semantic_units.append(None)
                idx += 1

            elif semantic_tag == IUEN_v2:
                truth = 'none'
                for cand in ['intersect', 'union', 'except']:
                    if g_sql[cand] is not None:
                        truth = cand
                        break

                true_selections.append([NEW_SQL_OPS.index(truth)])  # IUEN id
                eval_output.append(truth == tag_seq[idx][1][0])
                if truth == 'none':
                    true_semantic_units.append(None) # none of the above -> 'none'
                else:
                    new_su = list(tag_seq[idx])
                    new_su[1] = (truth, NEW_SQL_OPS.index(truth))
                    true_semantic_units.append([tuple(new_su)])
                idx += 1

                if truth == 'none' or (not eval_output[-1]):
                    # for remaining part, eval to False
                    while idx < len(tag_seq):
                        eval_output.append(False if tag_seq[idx][0] != OUTSIDE else None)
                        true_selections.append(None)
                        true_semantic_units.append(None)
                        idx += 1

                else: # truth == tag_seq[idx][1][0] and truth != 'none'
                    idx, main_sql_eval, main_sql_true_selections, main_sql_true_su = self.compare(
                        g_sql[truth], idx, tag_seq, bool_return_true_selections=True,
                        bool_return_true_semantic_units=True)
                    eval_output.extend(main_sql_eval)
                    true_selections.extend(main_sql_true_selections)
                    true_semantic_units.extend(main_sql_true_su)

            elif semantic_tag == SELECT_AGG_v2:
                select_col_agg_pairs = self.parse_select(g_sql['select'])
                while idx < len(tag_seq) and tag_seq[idx][0] == SELECT_AGG_v2:
                    col_idx = tag_seq[idx][1][-1]
                    agg_idx = tag_seq[idx][2][-1]
                    bool_distinct = tag_seq[idx][3]
                    eval_output.append((col_idx, agg_idx, bool_distinct) in select_col_agg_pairs)
                    true_selections.append(select_col_agg_pairs)

                    _true_semantic_units = []
                    for true_col_idx, true_agg_idx, true_bool_distinct in select_col_agg_pairs:
                        new_su = list(tag_seq[idx])
                        new_su[1] = (None, None, true_col_idx)
                        new_su[2] = (None, true_agg_idx)
                        new_su[3] = true_bool_distinct
                        _true_semantic_units.append(tuple(new_su))
                    true_semantic_units.append(_true_semantic_units)
                    idx += 1

            elif semantic_tag == WHERE_COL:
                if len(g_sql['where']) == 0:
                    while idx < len(tag_seq) and tag_seq[idx][0] in {WHERE_COL, WHERE_OP, WHERE_ROOT_TERM}:
                        if tag_seq[idx][0] == WHERE_ROOT_TERM and tag_seq[idx][3] == 'root':
                            eval_output.append(False)
                            true_selections.append(None)
                            true_semantic_units.append(None)
                            idx += 1
                            end_nested_idx = self.helper_find_closest_fw(tag_seq, idx, tgt_name=END_NESTED)
                            if end_nested_idx == -1: end_nested_idx = len(tag_seq) - 1
                            eval_output.extend([False if tag_seq[ii][0] != OUTSIDE else None
                                                for ii in range(idx, (end_nested_idx + 1))])
                            true_selections.extend([None] * (end_nested_idx + 1 - idx))
                            true_semantic_units.extend([None] * (end_nested_idx + 1 - idx))
                            idx = end_nested_idx + 1
                        else:
                            eval_output.append(False)
                            true_selections.append(None)
                            true_semantic_units.append(None)
                            idx += 1

                else:
                    col_op_root, and_or = self.parse_where_having(g_sql['where'])

                    while idx < len(tag_seq) and tag_seq[idx][0] == WHERE_COL:
                        col_idx = tag_seq[idx][1][-1]
                        eval_output.append(col_idx in set([col for (col,_,_),_,_ in col_op_root]))
                        true_selections.append([col for (col,_,_),_,_ in col_op_root])

                        _true_semantic_units = []
                        for true_col_idx in true_selections[-1]:
                            new_su = list(tag_seq[idx])
                            new_su[1] = (None, None, true_col_idx)
                            _true_semantic_units.append(tuple(new_su))
                        true_semantic_units.append(_true_semantic_units)
                        idx += 1
                    if idx == len(tag_seq): break  # check partial end

                    assert tag_seq[idx][0] == WHERE_OP

                    while idx < len(tag_seq) and tag_seq[idx][0] == WHERE_OP: # for all WHERE_COLs
                        while idx < len(tag_seq) and tag_seq[idx][0] == WHERE_OP: # for each col_idx
                            col_idx = tag_seq[idx][1][0][-1]
                            op_name, op_idx = tag_seq[idx][2]
                            true_col_op_pairs = [(col, op) for (col,_,_),op,_ in col_op_root]
                            eval_output.append((col_idx, op_idx) in true_col_op_pairs)
                            # true_selections.append([(col, op) for (col,_,_),op,_ in col_op_root])

                            _true_semantic_units, _true_selections = [], []
                            for true_col_idx, true_op_idx in true_col_op_pairs:
                                if true_col_idx == col_idx:
                                    new_su = list(tag_seq[idx])
                                    new_su[2] = (None, true_op_idx)
                                    _true_semantic_units.append(tuple(new_su))
                                    _true_selections.append((true_col_idx, true_op_idx))
                            if len(_true_selections) == 0:
                                true_selections.append(None)
                                true_semantic_units.append(None)
                            else:
                                true_selections.append(_true_selections)
                                true_semantic_units.append(_true_semantic_units)
                            idx += 1
                        while idx < len(tag_seq) and tag_seq[idx][0] == WHERE_ROOT_TERM:
                            col_idx = tag_seq[idx][1][0][-1]
                            op_name, op_idx = tag_seq[idx][2]
                            root_term = tag_seq[idx][3]
                            bool_matched_col_op = False
                            for (col,_,_), op, (val1, val2) in col_op_root:
                                if col == col_idx and op == op_idx:
                                    if not isinstance(val1, dict) and (val2 is None or not isinstance(val2, dict)):
                                        truth = 'terminal'
                                        true_selections.append([truth]) # dummy; no multi-choice for ROOT_TERM
                                        true_semantic_units.append(None)
                                        if root_term == truth:
                                            eval_output.append(True)
                                            idx += 1
                                        else:
                                            eval_output.append(False)
                                            idx += 1
                                            # consider terms associated with the nested root as wrong
                                            end_nested_idx = self.helper_find_closest_fw(tag_seq, idx, tgt_name=END_NESTED)
                                            if end_nested_idx == -1: end_nested_idx = len(tag_seq) - 1
                                            eval_output.extend([False if tag_seq[ii][0] != OUTSIDE else None for ii in
                                                                range(idx, (end_nested_idx + 1))])
                                            true_selections.extend([None] * (end_nested_idx + 1 - idx))
                                            true_semantic_units.extend([None] * (end_nested_idx + 1 - idx))
                                            idx = end_nested_idx + 1
                                    else:
                                        assert isinstance(val1, dict) and val2 is None
                                        truth = 'root'
                                        true_selections.append([truth])
                                        true_semantic_units.append([truth])
                                        if root_term != truth: #root_term = terminal
                                            eval_output.append(False)
                                            idx += 1
                                        else:
                                            eval_output.append(True)
                                            idx += 1
                                            # evaluate nested component
                                            end_nested_idx = self.helper_find_closest_fw(tag_seq, idx, tgt_name=END_NESTED)
                                            if end_nested_idx == -1: end_nested_idx = len(tag_seq) - 1
                                            idx, nested_eval, nested_true_selections, nested_true_su = self.compare(
                                                val1, idx, tag_seq[:(end_nested_idx + 1)],
                                                bool_return_true_selections=True, bool_return_true_semantic_units=True)
                                            eval_output.extend(nested_eval)
                                            true_selections.extend(nested_true_selections)
                                            true_semantic_units.extend(nested_true_su)
                                    bool_matched_col_op = True
                                    break

                            if not bool_matched_col_op:
                                eval_output.append(False)
                                true_selections.append(None)
                                true_semantic_units.append(None)
                                idx += 1
                                if root_term == 'root':
                                    # consider terms associated with the nested root as wrong
                                    end_nested_idx = self.helper_find_closest_fw(tag_seq, idx, tgt_name=END_NESTED)
                                    if end_nested_idx == -1: end_nested_idx = len(tag_seq) - 1
                                    # eval.extend([False] * (end_nested_idx - idx + 1))
                                    eval_output.extend([False if tag_seq[ii][0] != OUTSIDE else None for ii in
                                                        range(idx, (end_nested_idx + 1))])
                                    true_selections.extend([None] * (end_nested_idx + 1 - idx))
                                    true_semantic_units.extend([None] * (end_nested_idx + 1 - idx))
                                    idx = end_nested_idx + 1

            elif semantic_tag == GROUP_COL:
                if len(g_sql['groupBy']) == 0:
                    while idx < len(tag_seq) and tag_seq[idx][0] == GROUP_COL:
                        eval_output.append(False)
                        true_selections.append(None)
                        true_semantic_units.append(None)
                        idx += 1
                else:
                    groupBy_cols = self.parse_group(g_sql['groupBy'])
                    while idx < len(tag_seq) and tag_seq[idx][0] == GROUP_COL:
                        col_idx = tag_seq[idx][1][-1]
                        eval_output.append(col_idx in groupBy_cols)
                        true_selections.append(groupBy_cols)

                        _true_semantic_units = []
                        for col_idx in groupBy_cols:
                            new_su = list(tag_seq[idx])
                            new_su[1] = (None, None, col_idx)
                            _true_semantic_units.append(tuple(new_su))
                        true_semantic_units.append(_true_semantic_units)
                        idx += 1

            elif semantic_tag == HAV_AGG_v2:
                if len(g_sql['having']) == 0:
                    eval_output.append(False)
                    true_selections.append(None)
                    true_semantic_units.append(None)
                    idx += 1
                    while idx < len(tag_seq) and tag_seq[idx][0] in {HAV_AGG_v2, HAV_OP_v2, HAV_ROOT_TERM_v2}:
                        eval_output.append(False)
                        true_selections.append(None)
                        true_semantic_units.append(None)
                        idx += 1

                else:
                    col_op_root, and_or = self.parse_where_having(g_sql['having'])

                    while idx < len(tag_seq) and tag_seq[idx][0] == HAV_AGG_v2:  # for all HAV_COLs
                        tab_col_item, (agg_name, agg_idx), bool_distinct = tag_seq[idx][1:4]
                        col_idx = tab_col_item[-1]
                        eval_output.append((col_idx, agg_idx, bool_distinct) in
                                           set([col_agg for col_agg, _, _ in col_op_root]))
                        true_selections.append([col_agg for col_agg, _, _ in col_op_root])

                        _true_semantic_units = []
                        for true_col_idx, true_agg_idx, true_bool_distinct in true_selections[-1]:
                            new_su = list(tag_seq[idx])
                            new_su[1] = (None, None, true_col_idx)
                            new_su[2] = (None, true_agg_idx)
                            new_su[3] = true_bool_distinct
                            _true_semantic_units.append(tuple(new_su))
                        true_semantic_units.append(_true_semantic_units)
                        idx += 1

                        while idx < len(tag_seq) and tag_seq[idx][0] == HAV_OP_v2:  # for each col_idx
                            col_idx = tag_seq[idx][1][0][-1]
                            agg_name, agg_idx = tag_seq[idx][1][1]
                            bool_distinct = tag_seq[idx][1][2]
                            op_name, op_idx = tag_seq[idx][2]
                            true_col_agg_op_pairs = [(col_agg, op) for col_agg, op, _ in col_op_root]
                            eval_output.append(((col_idx, agg_idx, bool_distinct), op_idx) in true_col_agg_op_pairs)
                            # true_selections.append([(col_agg, op) for col_agg, op, _ in col_op_root])

                            _true_semantic_units, _true_selections = [], []
                            for (true_col_idx, true_agg_idx, true_bool_distinct), true_op_idx in true_col_agg_op_pairs:
                                if true_col_idx == col_idx and true_agg_idx == agg_idx and \
                                  true_bool_distinct == bool_distinct:
                                    new_su = list(tag_seq[idx])
                                    new_su[2] = (None, true_op_idx)
                                    _true_semantic_units.append(tuple(new_su))
                                    _true_selections.append(((true_col_idx, true_agg_idx, true_bool_distinct), true_op_idx))
                            if len(_true_selections) == 0:
                                true_selections.append(None)
                                true_semantic_units.append(None)
                            else:
                                true_selections.append(_true_selections)
                                true_semantic_units.append(_true_semantic_units)
                            idx += 1

                        while idx < len(tag_seq) and tag_seq[idx][0] == HAV_ROOT_TERM_v2:
                            col_idx = tag_seq[idx][1][0][-1]
                            agg_name, agg_idx = tag_seq[idx][1][1]
                            bool_distinct = tag_seq[idx][1][2]
                            op_name, op_idx = tag_seq[idx][2]
                            root_term = tag_seq[idx][3]
                            bool_matched_col_op = False
                            for col_agg, op, (val1, val2) in col_op_root:
                                if (col_idx, agg_idx, bool_distinct) == col_agg and op == op_idx:
                                    if not isinstance(val1, dict) and (val2 is None or not isinstance(val2, dict)):
                                        truth = 'terminal'
                                        true_selections.append(['terminal'])
                                        true_semantic_units.append(['terminal'])
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
                                            true_selections.extend([None] * (end_nested_idx + 1 - idx))
                                            true_semantic_units.extend([None] * (end_nested_idx + 1 - idx))
                                            idx = end_nested_idx + 1
                                    else:
                                        assert isinstance(val1, dict) and val2 is None
                                        truth = 'root'
                                        true_selections.append(['root'])
                                        true_semantic_units.append(['root'])
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
                                            idx, nested_eval, nested_true_selections, nested_true_su = self.compare(
                                                val1, idx, tag_seq[:(end_nested_idx + 1)],
                                                bool_return_true_selections=True, bool_return_true_semantic_units=True)
                                            eval_output.extend(nested_eval)
                                            true_selections.extend(nested_true_selections)
                                            true_semantic_units.extend(nested_true_su)
                                    bool_matched_col_op = True
                                    break

                            if not bool_matched_col_op: # cannot find matched (col_agg, op)
                                eval_output.append(False)
                                true_selections.append(None)
                                true_semantic_units.append(None)
                                idx += 1
                                if root_term == 'root':
                                    # consider terms associated with the nested root as wrong
                                    end_nested_idx = self.helper_find_closest_fw(tag_seq, idx, tgt_name=END_NESTED)
                                    if end_nested_idx == -1: end_nested_idx = len(tag_seq) - 1
                                    eval_output.extend([False if tag_seq[ii][0] != OUTSIDE else None for ii in
                                                        range(idx, (end_nested_idx + 1))])
                                    true_selections.extend([None] * (end_nested_idx + 1 - idx))
                                    true_semantic_units.extend([None] * (end_nested_idx + 1 - idx))
                                    idx = end_nested_idx + 1

            elif semantic_tag == ORDER_AGG_v2:
                if len(g_sql['orderBy']) == 0:
                    while idx < len(tag_seq) and tag_seq[idx][0] in {ORDER_AGG_v2, ORDER_DESC_ASC, ORDER_LIMIT}:
                        eval_output.append(False)
                        true_selections.append(None)
                        true_semantic_units.append(None)
                        idx += 1

                else:
                    col_agg_desc_asc, col_agg_limit = self.parse_orderBy_limit(g_sql['orderBy'], g_sql['limit'])

                    while idx < len(tag_seq) and tag_seq[idx][0] == ORDER_AGG_v2:
                        # ORDER_AGG
                        col_idx = tag_seq[idx][1][-1]
                        agg_name, agg_idx = tag_seq[idx][2]
                        bool_distinct = tag_seq[idx][3]
                        eval_output.append((col_idx, agg_idx, bool_distinct) in
                                           set([col_agg for col_agg, _ in col_agg_desc_asc]))
                        true_selections.append([col_agg for col_agg, _ in col_agg_desc_asc])

                        _true_semantic_units = []
                        for true_col_idx, true_agg_idx, true_bool_distinct in true_selections[-1]:
                            new_su = list(tag_seq[idx])
                            new_su[1] = (None, None, true_col_idx)
                            new_su[2] = (None, true_agg_idx)
                            new_su[3] = true_bool_distinct
                            _true_semantic_units.append(tuple(new_su))
                        true_semantic_units.append(_true_semantic_units)
                        idx += 1
                        if idx == len(tag_seq): break  # check partial end

                        # ORDER_DESC_ASC
                        if tag_seq[idx][0] == ORDER_DESC_ASC:
                            col_idx = tag_seq[idx][1][0][-1]
                            agg_name, agg_idx = tag_seq[idx][1][1]
                            bool_distinct = tag_seq[idx][1][2]
                            desc_asc = tag_seq[idx][2]
                            eval_output.append(((col_idx, agg_idx, bool_distinct), desc_asc) in col_agg_desc_asc)
                            # true_selections.append(col_agg_desc_asc)
                            true_selections.append(None) # no option selection for ORDER_DESC_ASC
                            true_semantic_units.append(None)
                            idx += 1

                        if idx == len(tag_seq): break  # check partial end

                        # ORDER_LIMIT
                        if tag_seq[idx][0] == ORDER_LIMIT:
                            col_idx = tag_seq[idx][1][0][-1]
                            agg_name, agg_idx = tag_seq[idx][1][1]
                            bool_distinct = tag_seq[idx][1][2]
                            eval_output.append(((col_idx, agg_idx, bool_distinct), True) in col_agg_limit)
                            # true_selections.append(col_agg_limit)
                            true_selections.append(None) # no option selection for ORDER_LIMIT
                            true_semantic_units.append(None)
                            idx += 1

            else:
                raise Exception("Invalid id {} at idx {} in history:\n{}".format(semantic_tag, idx, tag_seq))

        return_items = [idx, eval_output]
        if bool_return_true_selections:
            return_items.append(true_selections)

        if bool_return_true_semantic_units:
            return_items.append(true_semantic_units)

        return tuple(return_items)


class UserSim(BaseUserSim):
    def __init__(self, error_evaluator, bool_structure_question=False):
        """
        Constructor of UserSim.
        :param error_evaluator: an instance of ErrorEvaluator.
        :param bool_structure_question: set to True if SQL structure (WHERE/GROUP_COL, ORDER/HAV_AGG_v2) is
            allow to question.
        """
        BaseUserSim.__init__(self, error_evaluator)
        self.bool_structure_question = bool_structure_question

    def get_selection(self, pointer, answer_sheet, sel_none_of_above):
        pointer_truth = self.true_selections[pointer]  # ground-truth decision
        old_su = self.tag_seq[pointer]
        semantic_tag = old_su[0]
        selections = []

        # if the prefix query is correct, possible true decisions exist
        if pointer_truth is not None:
            for select_id, select_val in answer_sheet.items():
                if len(pointer_truth) and select_val in pointer_truth:
                    selections.append(select_id)
                elif len(pointer_truth) == 0 and select_val is None:
                    selections.append(select_id)

            if len(selections) == 0:  # none of the above
                selections.append(sel_none_of_above)
        elif self.bool_structure_question and semantic_tag in {WHERE_COL, GROUP_COL, ORDER_AGG_v2, HAV_AGG_v2}:
            selections.append(sel_none_of_above + 1) # sel_invalid_structure
        else:
            selections.append(sel_none_of_above)

        print("User answer: %s.\n" % str(selections))

        return selections


class GoldUserSim(BaseGoldUserSim):
    def __init__(self, error_evaluator, bool_structure_question=False):
        BaseGoldUserSim.__init__(self, error_evaluator)
        self.kw2asterisk = None
        self.base_vocab = None
        self.column_names_surface_form_to_id = None
        self.complete_vocab = None
        self.bool_structure_question = bool_structure_question

    def update_truth(self, groud_truth):
        self.ground_truth = groud_truth
        self.kw2asterisk = groud_truth['extracted_clause_asterisk']
        self.base_vocab = groud_truth['base_vocab']
        self.column_names_surface_form_to_id = groud_truth['column_names_surface_form_to_id']
        id2col_name = {v:k for k,v in self.column_names_surface_form_to_id.items()}

        self.complete_vocab = []
        for id in range(len(self.base_vocab)):
            self.complete_vocab.append(self.base_vocab.id_to_token(id))
        for id in range(len(self.column_names_surface_form_to_id)):
            self.complete_vocab.append(id2col_name[id])

    def get_gold_selection(self, pointer):
        pointer_truth = self.true_semantic_units[pointer]  # ground-truth decision
        old_su = self.tag_seq[pointer]
        semantic_tag = old_su[0]
        old_dec_item = self.dec_seq[old_su[-1]]
        gold_semantic_units, gold_dec_items = [], []

        if pointer_truth is not None:
            gold_semantic_units.extend(pointer_truth)
            for su in gold_semantic_units:
                if semantic_tag in {SELECT_AGG_v2, HAV_AGG_v2, ORDER_AGG_v2}:
                    new_decision_item = []
                    col, agg, bool_distinct = su[1:4]
                    if agg[-1] > 0:
                        agg_name = AGG_OPS[agg[-1]]
                        new_decision_item.append(self.complete_vocab.index(agg_name))
                        new_decision_item.append(self.complete_vocab.index('('))
                        if bool_distinct:
                            new_decision_item.append(self.complete_vocab.index('distinct'))
                        new_decision_item.append(col[-1])
                        new_decision_item.append(self.complete_vocab.index(')'))
                    else:
                        if bool_distinct:
                            new_decision_item.append(self.complete_vocab.index('distinct'))
                        new_decision_item.append(col[-1])
                    gold_dec_items.append(new_decision_item)

                elif semantic_tag in {WHERE_COL, GROUP_COL}:
                    gold_dec_items.append(su[1][-1])

                elif semantic_tag in {WHERE_OP, HAV_OP_v2}:
                    op_name = NEW_WHERE_OPS[su[2][-1]]
                    if op_name in self.complete_vocab:
                        gold_dec_items.append([self.complete_vocab.index(op_name)])
                    else:
                        assert op_name in {'>=', '<=', 'not like', 'not in'}
                        new_decision_item = []
                        if op_name in {'>=', '<='}:
                            new_decision_item.append(self.complete_vocab.index(op_name[0]))
                            new_decision_item.append(self.complete_vocab.index(op_name[1]))
                        else:
                            op_name1, op_name2 = op_name.split(" ")
                            new_decision_item.append(self.complete_vocab.index(op_name1))
                            new_decision_item.append(self.complete_vocab.index(op_name2))
                        gold_dec_items.append(new_decision_item)

                elif semantic_tag == IUEN_v2:
                    iuen_name = NEW_SQL_OPS[su[1][-1]]
                    assert iuen_name != 'none'
                    gold_dec_items.append(self.complete_vocab.index(iuen_name))

                else:
                    raise Exception("Invalid semantic_tag: {}!".format(semantic_tag))

        print("Gold semantic units: %s." % str(gold_semantic_units))
        print("Gold dec_items: %s." % str(gold_dec_items))

        if len(gold_semantic_units):
            selections = [choice + 1 for choice in range(len(gold_semantic_units))]
            sel_none_of_above = len(gold_semantic_units) + 1
        elif self.bool_structure_question and semantic_tag in {WHERE_COL, GROUP_COL, ORDER_AGG_v2, HAV_AGG_v2}:
            sel_none_of_above = 1
            selections = [sel_none_of_above + 1] # invalid structure
        else:
            sel_none_of_above = 1
            selections = [sel_none_of_above]
        print("Gold user selections ('none of above' = %d): %s.\n" % (sel_none_of_above, str(selections)))

        return gold_semantic_units, gold_dec_items, sel_none_of_above, selections
