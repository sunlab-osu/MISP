# heuristic baselines for error detection

from interaction_framework.question_gen import QuestionGenerator, SELECT_COL, SELECT_AGG, WHERE_COL, WHERE_OP, WHERE_ROOT_TERM, ANDOR, \
    GROUP_COL, GROUP_NHAV, HAV_COL, HAV_AGG, HAV_OP, HAV_ROOT_TERM, ORDER_COL, ORDER_AGG, ORDER_DESC_ASC_LIMIT, \
    IUEN, OUTSIDE, END_NESTED
from interaction_framework.err_detector import ErrorDetectorEvaluator, ErrorDetectorProbability, ErrorDetectorBayDropout
import evaluation


class ErrorDetectorEvaluatorSyntaxSQL(ErrorDetectorEvaluator):
    def __init__(self):
        ErrorDetectorEvaluator.__init__(self)

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
            agg_name = 'none_agg' if agg_id == 0 else evaluation.AGG_OPS[agg_id]
            unit_op, col_unit1, col_unit2 = val_unit
            assert unit_op==0 and col_unit2 is None # syntaxSQL does not support "col1-col2"
            _, col_idx, _ = col_unit1
            col_agg_pairs.append((col_idx, agg_name))
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
                agg_name = 'none_agg' if agg_idx == 0 else evaluation.AGG_OPS[agg_idx]
                col_op_root.append(((col_idx, agg_name), op_name, (val1, val2)))

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
            agg_name = 'none_agg' if agg_idx == 0 else evaluation.AGG_OPS[agg_idx]
            col_agg_other.append(((col_idx, agg_name), (asc_desc, bool_limit)))
        return col_agg_other

    def compare_whole(self, g_sql, start_idx, BIO_history):
        eval_output = []
        idx = start_idx
        while idx < len(BIO_history):
            seg_id = BIO_history[idx][0]
            if seg_id == OUTSIDE:
                eval_output.append(None)
                idx += 1

            elif seg_id == IUEN:
                truth = 'none'
                for cand in ['intersect', 'union', 'except']:
                    if g_sql[cand] is not None:
                        truth = cand
                        break

                if truth == 'none':
                    if BIO_history[idx][1] == 'none':
                        eval_output.append(True)
                        idx += 1
                    else:
                        eval_output.append(False)
                        idx += 1

                        assert "root" in BIO_history[idx] and BIO_history[idx][0] == OUTSIDE
                        eval_output.append(None)
                        idx += 1

                        end_nested_idx = self.helper_find_closest_fw(BIO_history, idx, tgt_name=END_NESTED)
                        assert end_nested_idx != -1
                        self.helper_find_closest_fw(BIO_history, idx, tgt_name=END_NESTED)
                        idx, main_sql_eval = self.compare(g_sql, idx, BIO_history[:end_nested_idx])
                        assert idx == end_nested_idx
                        eval_output.extend(main_sql_eval)

                        # for remaining part (nested sql), eval to False
                        while idx < len(BIO_history):
                            eval_output.append(False if BIO_history[idx][0] != OUTSIDE else None)
                            idx += 1
                else:
                    if BIO_history[idx][1] == 'none':
                        eval_output.append(False)
                        idx += 1

                    elif BIO_history[idx][1] == truth:
                        eval_output.append(True)
                        idx += 1

                        assert "root" in BIO_history[idx] and BIO_history[idx][0] == OUTSIDE
                        eval_output.append(None)
                        idx += 1

                        end_nested_idx = self.helper_find_closest_fw(BIO_history, idx, tgt_name=END_NESTED)
                        assert end_nested_idx != -1
                        idx, main_sql_eval = self.compare(g_sql, idx, BIO_history[:end_nested_idx])
                        assert idx == end_nested_idx
                        eval_output.extend(main_sql_eval)

                        idx, nested_sql_eval = self.compare(g_sql[truth], idx, BIO_history)
                        eval_output.extend(nested_sql_eval)

                    else:
                        eval_output.append(False)
                        idx += 1

                        assert "root" in BIO_history[idx] and BIO_history[idx][0] == OUTSIDE
                        eval_output.append(None)
                        idx += 1

                        end_nested_idx = self.helper_find_closest_fw(BIO_history, idx, tgt_name=END_NESTED)
                        assert end_nested_idx != -1
                        idx, main_sql_eval = self.compare(g_sql, idx, BIO_history[:end_nested_idx])
                        assert idx == end_nested_idx
                        eval_output.extend(main_sql_eval)

                        while idx < len(BIO_history):
                            eval_output.append(False if BIO_history[idx][0] != OUTSIDE else None)
                            idx += 1

            elif seg_id == SELECT_COL:
                select_col_agg_pairs = self.parse_select(g_sql['select'])
                while BIO_history[idx][0] == SELECT_COL:
                    col_idx = BIO_history[idx][1][-1]
                    eval_output.append(col_idx in set([col for col,agg in select_col_agg_pairs]))
                    idx += 1
                assert BIO_history[idx][0] == SELECT_AGG
                while idx < len(BIO_history) and BIO_history[idx][0] == SELECT_AGG:
                    col_idx = BIO_history[idx][1][-1]
                    agg_name = BIO_history[idx][2]
                    eval_output.append((col_idx, agg_name) in select_col_agg_pairs)
                    idx += 1

            elif seg_id == WHERE_COL:
                if len(g_sql['where']) == 0:
                    while idx < len(BIO_history) and BIO_history[idx][0] in {WHERE_COL, WHERE_OP, WHERE_ROOT_TERM, ANDOR}:
                        if BIO_history[idx][0] == WHERE_ROOT_TERM and BIO_history[idx][3] == 'root':
                            eval_output.append(False)
                            idx += 1
                            end_nested_idx = self.helper_find_closest_fw(BIO_history, idx, tgt_name=END_NESTED)
                            if end_nested_idx == -1: end_nested_idx = len(BIO_history) - 1
                            # eval.extend([False] * (end_nested_idx - idx + 1))
                            eval_output.extend([False if BIO_history[idx][0] != OUTSIDE else None for idx in range(idx, (end_nested_idx + 1))])
                            idx = end_nested_idx + 1
                        else:
                            eval_output.append(False)
                            idx += 1

                else:
                    col_op_root, and_or = self.parse_where_having(g_sql['where'])

                    while BIO_history[idx][0] == WHERE_COL:
                        col_idx = BIO_history[idx][1][-1]
                        eval_output.append(col_idx in set([col for (col,_),_,_ in col_op_root]))
                        idx += 1
                    if BIO_history[idx][0] == ANDOR:
                        eval_output.append(BIO_history[idx][1] == and_or)
                        idx += 1

                    assert BIO_history[idx][0] == WHERE_OP

                    while idx < len(BIO_history) and BIO_history[idx][0] == WHERE_OP: # for all WHERE_COLs
                        while BIO_history[idx][0] == WHERE_OP: # for each col_idx
                           col_idx = BIO_history[idx][1][0][-1]
                           op_name = BIO_history[idx][2]
                           eval_output.append((col_idx, op_name) in set([(col, op) for (col,_),op,_ in col_op_root]))
                           idx += 1
                        while idx < len(BIO_history) and BIO_history[idx][0] == WHERE_ROOT_TERM:
                            col_idx = BIO_history[idx][1][0][-1]
                            op_name = BIO_history[idx][2]
                            root_term = BIO_history[idx][3]
                            bool_matched_col_op = False
                            for (col,_), op, (val1, val2) in col_op_root:
                                if col == col_idx and op == op_name:
                                    if not isinstance(val1, dict) and (val2 is None or not isinstance(val2, dict)):
                                        truth = 'terminal'
                                        if root_term == truth:
                                            eval_output.append(True)
                                            idx += 1
                                        else:
                                            eval_output.append(False)
                                            idx += 1
                                            # consider terms associated with the nested root as wrong
                                            end_nested_idx = self.helper_find_closest_fw(BIO_history, idx, tgt_name=END_NESTED)
                                            if end_nested_idx == -1: end_nested_idx = len(BIO_history) - 1
                                            # eval.extend([False] * (end_nested_idx - idx + 1))
                                            eval_output.extend([False if BIO_history[idx][0] != OUTSIDE else None for idx in
                                                         range(idx, (end_nested_idx + 1))])
                                            idx = end_nested_idx + 1
                                    else:
                                        assert isinstance(val1, dict) and val2 is None
                                        truth = 'root'
                                        if root_term != truth: #root_term = terminal
                                            eval_output.append(False)
                                            idx += 1
                                        else:
                                            eval_output.append(True)
                                            idx += 1
                                            # evaluate nested component
                                            end_nested_idx = self.helper_find_closest_fw(BIO_history, idx, tgt_name=END_NESTED)
                                            if end_nested_idx == -1: end_nested_idx = len(BIO_history) - 1
                                            idx, nested_eval = self.compare(val1, idx, BIO_history[:(end_nested_idx + 1)])
                                            eval_output.extend(nested_eval)
                                    bool_matched_col_op = True
                                    break

                            if not bool_matched_col_op:
                                eval_output.append(False)
                                idx += 1
                                if root_term == 'root':
                                    # consider terms associated with the nested root as wrong
                                    end_nested_idx = self.helper_find_closest_fw(BIO_history, idx, tgt_name=END_NESTED)
                                    if end_nested_idx == -1: end_nested_idx = len(BIO_history) - 1
                                    # eval.extend([False] * (end_nested_idx - idx + 1))
                                    eval_output.extend([False if BIO_history[idx][0] != OUTSIDE else None for idx in
                                                 range(idx, (end_nested_idx + 1))])
                                    idx = end_nested_idx + 1

            elif seg_id == GROUP_COL:
                if len(g_sql['groupBy']) == 0:
                    groupBy_cols = []
                else:
                    groupBy_cols = set(self.parse_group(g_sql['groupBy']))
                while BIO_history[idx][0] == GROUP_COL:
                    col_idx = BIO_history[idx][1][-1]
                    eval_output.append(col_idx in groupBy_cols)
                    idx += 1

            elif seg_id == GROUP_NHAV:
                eval_output.append(len(g_sql['having']) == 0)
                idx += 1

            elif seg_id == HAV_COL:
                if len(g_sql['having']) == 0:
                    eval_output.append(False)
                    idx += 1
                    while idx < len(BIO_history) and BIO_history[idx][0] in {HAV_COL, HAV_AGG, HAV_OP, HAV_ROOT_TERM}:
                        eval_output.append(False)
                        idx += 1

                else:
                    col_op_root, and_or = self.parse_where_having(g_sql['having'])

                    while BIO_history[idx][0] == HAV_COL:
                        col_idx = BIO_history[idx][1][-1]
                        eval_output.append(col_idx in set([col for (col,_), _, _ in col_op_root]))
                        idx += 1

                    assert BIO_history[idx][0] == HAV_AGG

                    while idx < len(BIO_history) and BIO_history[idx][0] == HAV_AGG:  # for all HAV_COLs
                        col_idx, agg_name = BIO_history[idx][1:3]
                        eval_output.append((col_idx, agg_name) in set([col_agg for col_agg, _, _ in col_op_root]))
                        idx += 1

                        while BIO_history[idx][0] == HAV_OP:  # for each col_idx
                            col_idx = BIO_history[idx][1][0][-1]
                            agg_name = BIO_history[idx][1][1]
                            op_name = BIO_history[idx][2]
                            eval_output.append(((col_idx, agg_name), op_name) in set([(col_agg, op) for col_agg, op, _ in col_op_root]))
                            idx += 1
                        while idx < len(BIO_history) and BIO_history[idx][0] == HAV_ROOT_TERM:
                            col_idx = BIO_history[idx][1][0][-1]
                            agg_name = BIO_history[idx][1][1]
                            op_name = BIO_history[idx][2]
                            root_term = BIO_history[idx][3]
                            bool_matched_col_op = False
                            for col_agg, op, (val1, val2) in col_op_root:
                                if (col_idx, agg_name) == col_agg and op == op_name:
                                    if not isinstance(val1, dict) and (val2 is None or not isinstance(val2, dict)):
                                        truth = 'terminal'
                                        if root_term == truth:
                                            eval_output.append(True)
                                            idx += 1
                                        else:
                                            eval_output.append(False)
                                            idx += 1
                                            # consider terms associated with the nested root as wrong
                                            end_nested_idx = self.helper_find_closest_fw(BIO_history, idx, tgt_name=END_NESTED)
                                            if end_nested_idx == -1: end_nested_idx = len(BIO_history) - 1
                                            # eval.extend([False] * (end_nested_idx - idx + 1))
                                            eval_output.extend([False if BIO_history[idx][0] != OUTSIDE else None for idx in
                                                         range(idx, (end_nested_idx + 1))])
                                            idx = end_nested_idx + 1
                                    else:
                                        assert isinstance(val1, dict) and val2 is None
                                        truth = 'root'
                                        if root_term != truth:  # root_term = terminal
                                            eval_output.append(False)
                                            idx += 1
                                        else:
                                            eval_output.append(True)
                                            idx += 1
                                            # evaluate nested component
                                            end_nested_idx = self.helper_find_closest_fw(BIO_history, idx,
                                                                                         tgt_name=END_NESTED)
                                            if end_nested_idx == -1: end_nested_idx = len(BIO_history) - 1
                                            idx, nested_eval = self.compare(val1, idx, BIO_history[:(end_nested_idx + 1)])
                                            eval_output.extend(nested_eval)
                                    bool_matched_col_op = True
                                    break

                            if not bool_matched_col_op: # cannot find matched (col_agg, op)
                                eval_output.append(False)
                                idx += 1
                                if root_term == 'root':
                                    # consider terms associated with the nested root as wrong
                                    end_nested_idx = self.helper_find_closest_fw(BIO_history, idx, tgt_name=END_NESTED)
                                    if end_nested_idx == -1: end_nested_idx = len(BIO_history) - 1
                                    # eval.extend([False] * (end_nested_idx - idx + 1))
                                    eval_output.extend([False if BIO_history[idx][0] != OUTSIDE else None for idx in
                                                 range(idx, (end_nested_idx + 1))])
                                    idx = end_nested_idx + 1

            elif seg_id == ORDER_COL:
                if len(g_sql['orderBy']) == 0:
                    while idx < len(BIO_history) and BIO_history[idx][0] in {ORDER_COL, ORDER_AGG, ORDER_DESC_ASC_LIMIT}:
                        eval_output.append(False)
                        idx += 1

                else:
                    col_agg_other = self.parse_orderBy_limit(g_sql['orderBy'], g_sql['limit'])
                    while BIO_history[idx][0] == ORDER_COL:
                        col_idx = BIO_history[idx][1][-1]
                        eval_output.append(col_idx in set([col for (col,_), _ in col_agg_other]))
                        idx += 1

                    assert BIO_history[idx][0] == ORDER_AGG
                    while idx < len(BIO_history) and BIO_history[idx][0] == ORDER_AGG:
                        # ORDER_AGG
                        col_idx = BIO_history[idx][1][-1]
                        agg_name = BIO_history[idx][2]
                        eval_output.append((col_idx, agg_name) in set([col_agg for col_agg, _ in col_agg_other]))
                        idx += 1

                        # ORDER_DESC_ASC_LIMIT
                        assert BIO_history[idx][0] == ORDER_DESC_ASC_LIMIT
                        col_idx = BIO_history[idx][1][0][-1]
                        agg_name = BIO_history[idx][1][1]
                        desc_asc_limit = BIO_history[idx][2]
                        eval_output.append(((col_idx, agg_name), desc_asc_limit) in col_agg_other)
                        idx += 1

            else:
                raise Exception("Invalid id {} at idx {} in history:\n{}".format(seg_id, idx, BIO_history))

        return idx, eval_output

    def compare(self, g_sql, start_idx, BIO_history):
        eval_output = []
        idx = start_idx
        while idx < len(BIO_history):
            seg_id = BIO_history[idx][0]
            if seg_id == OUTSIDE:
                eval_output.append(None)
                idx += 1

            elif seg_id == IUEN:
                truth = 'none'
                for cand in ['intersect', 'union', 'except']:
                    if g_sql[cand] is not None:
                        truth = cand
                        break

                if truth == 'none':
                    if BIO_history[idx][1] == 'none':
                        eval_output.append(True)
                        idx += 1
                    else:
                        eval_output.append(False)
                        idx += 1
                        if idx == len(BIO_history): break # check partial end

                        assert "root" in BIO_history[idx] and BIO_history[idx][0] == OUTSIDE
                        eval_output.append(None)
                        idx += 1
                        if idx == len(BIO_history): break  # check partial end

                        end_nested_idx = self.helper_find_closest_fw(BIO_history, idx, tgt_name=END_NESTED)
                        # assert end_nested_idx != -1
                        if end_nested_idx == -1:
                            end_nested_idx = None
                        idx, main_sql_eval = self.compare(g_sql, idx, BIO_history[:end_nested_idx])
                        # assert idx == end_nested_idx
                        eval_output.extend(main_sql_eval)

                        # for remaining part (nested sql), eval to False
                        while idx < len(BIO_history):
                            eval_output.append(False if BIO_history[idx][0] != OUTSIDE else None)
                            idx += 1
                else:
                    if BIO_history[idx][1] == 'none':
                        eval_output.append(False)
                        idx += 1

                    elif BIO_history[idx][1] == truth:
                        eval_output.append(True)
                        idx += 1
                        if idx == len(BIO_history): break  # check partial end

                        assert "root" in BIO_history[idx] and BIO_history[idx][0] == OUTSIDE
                        eval_output.append(None)
                        idx += 1
                        if idx == len(BIO_history): break  # check partial end

                        end_nested_idx = self.helper_find_closest_fw(BIO_history, idx, tgt_name=END_NESTED)
                        # assert end_nested_idx != -1
                        if end_nested_idx == -1:
                            end_nested_idx = None
                        idx, main_sql_eval = self.compare(g_sql, idx, BIO_history[:end_nested_idx])
                        # assert idx == end_nested_idx
                        eval_output.extend(main_sql_eval)
                        if idx == len(BIO_history): break  # check partial end

                        idx, nested_sql_eval = self.compare(g_sql[truth], idx, BIO_history)
                        eval_output.extend(nested_sql_eval)

                    else:
                        eval_output.append(False)
                        idx += 1
                        if idx == len(BIO_history): break  # check partial end

                        assert "root" in BIO_history[idx] and BIO_history[idx][0] == OUTSIDE
                        eval_output.append(None)
                        idx += 1
                        if idx == len(BIO_history): break  # check partial end

                        end_nested_idx = self.helper_find_closest_fw(BIO_history, idx, tgt_name=END_NESTED)
                        # assert end_nested_idx != -1
                        if end_nested_idx == -1:
                            end_nested_idx = None
                        idx, main_sql_eval = self.compare(g_sql, idx, BIO_history[:end_nested_idx])
                        # assert idx == end_nested_idx
                        eval_output.extend(main_sql_eval)

                        while idx < len(BIO_history):
                            eval_output.append(False if BIO_history[idx][0] != OUTSIDE else None)
                            idx += 1

            elif seg_id == SELECT_COL:
                select_col_agg_pairs = self.parse_select(g_sql['select'])
                while idx < len(BIO_history) and BIO_history[idx][0] == SELECT_COL:
                    col_idx = BIO_history[idx][1][-1]
                    eval_output.append(col_idx in set([col for col,agg in select_col_agg_pairs]))
                    idx += 1
                if idx == len(BIO_history): break  # check partial end

                assert BIO_history[idx][0] == SELECT_AGG
                while idx < len(BIO_history) and BIO_history[idx][0] == SELECT_AGG:
                    col_idx = BIO_history[idx][1][-1]
                    agg_name = BIO_history[idx][2]
                    eval_output.append((col_idx, agg_name) in select_col_agg_pairs)
                    idx += 1

            elif seg_id == WHERE_COL:
                if len(g_sql['where']) == 0:
                    while idx < len(BIO_history) and BIO_history[idx][0] in {WHERE_COL, WHERE_OP, WHERE_ROOT_TERM, ANDOR}:
                        if BIO_history[idx][0] == WHERE_ROOT_TERM and BIO_history[idx][3] == 'root':
                            eval_output.append(False)
                            idx += 1
                            end_nested_idx = self.helper_find_closest_fw(BIO_history, idx, tgt_name=END_NESTED)
                            if end_nested_idx == -1: end_nested_idx = len(BIO_history) - 1
                            # eval.extend([False] * (end_nested_idx - idx + 1))
                            eval_output.extend([False if BIO_history[idx][0] != OUTSIDE else None for idx in range(idx, (end_nested_idx + 1))])
                            idx = end_nested_idx + 1
                        else:
                            eval_output.append(False)
                            idx += 1

                else:
                    col_op_root, and_or = self.parse_where_having(g_sql['where'])

                    while idx < len(BIO_history) and BIO_history[idx][0] == WHERE_COL:
                        col_idx = BIO_history[idx][1][-1]
                        eval_output.append(col_idx in set([col for (col,_),_,_ in col_op_root]))
                        idx += 1
                    if idx == len(BIO_history): break  # check partial end

                    if BIO_history[idx][0] == ANDOR:
                        eval_output.append(BIO_history[idx][1] == and_or)
                        idx += 1
                    if idx == len(BIO_history): break  # check partial end

                    assert BIO_history[idx][0] == WHERE_OP

                    while idx < len(BIO_history) and BIO_history[idx][0] == WHERE_OP: # for all WHERE_COLs
                        while idx < len(BIO_history) and BIO_history[idx][0] == WHERE_OP: # for each col_idx
                           col_idx = BIO_history[idx][1][0][-1]
                           op_name = BIO_history[idx][2]
                           eval_output.append((col_idx, op_name) in set([(col, op) for (col,_),op,_ in col_op_root]))
                           idx += 1
                        while idx < len(BIO_history) and BIO_history[idx][0] == WHERE_ROOT_TERM:
                            col_idx = BIO_history[idx][1][0][-1]
                            op_name = BIO_history[idx][2]
                            root_term = BIO_history[idx][3]
                            bool_matched_col_op = False
                            for (col,_), op, (val1, val2) in col_op_root:
                                if col == col_idx and op == op_name:
                                    if not isinstance(val1, dict) and (val2 is None or not isinstance(val2, dict)):
                                        truth = 'terminal'
                                        if root_term == truth:
                                            eval_output.append(True)
                                            idx += 1
                                        else:
                                            eval_output.append(False)
                                            idx += 1
                                            # consider terms associated with the nested root as wrong
                                            end_nested_idx = self.helper_find_closest_fw(BIO_history, idx, tgt_name=END_NESTED)
                                            if end_nested_idx == -1: end_nested_idx = len(BIO_history) - 1
                                            # eval.extend([False] * (end_nested_idx - idx + 1))
                                            eval_output.extend([False if BIO_history[idx][0] != OUTSIDE else None for idx in
                                                         range(idx, (end_nested_idx + 1))])
                                            idx = end_nested_idx + 1
                                    else:
                                        assert isinstance(val1, dict) and val2 is None
                                        truth = 'root'
                                        if root_term != truth: #root_term = terminal
                                            eval_output.append(False)
                                            idx += 1
                                        else:
                                            eval_output.append(True)
                                            idx += 1
                                            # evaluate nested component
                                            end_nested_idx = self.helper_find_closest_fw(BIO_history, idx, tgt_name=END_NESTED)
                                            if end_nested_idx == -1: end_nested_idx = len(BIO_history) - 1
                                            idx, nested_eval = self.compare(val1, idx, BIO_history[:(end_nested_idx + 1)])
                                            eval_output.extend(nested_eval)
                                    bool_matched_col_op = True
                                    break

                            if not bool_matched_col_op:
                                eval_output.append(False)
                                idx += 1
                                if root_term == 'root':
                                    # consider terms associated with the nested root as wrong
                                    end_nested_idx = self.helper_find_closest_fw(BIO_history, idx, tgt_name=END_NESTED)
                                    if end_nested_idx == -1: end_nested_idx = len(BIO_history) - 1
                                    # eval.extend([False] * (end_nested_idx - idx + 1))
                                    eval_output.extend([False if BIO_history[idx][0] != OUTSIDE else None for idx in
                                                 range(idx, (end_nested_idx + 1))])
                                    idx = end_nested_idx + 1

            elif seg_id == GROUP_COL:
                if len(g_sql['groupBy']) == 0:
                    groupBy_cols = []
                else:
                    groupBy_cols = set(self.parse_group(g_sql['groupBy']))
                while idx < len(BIO_history) and BIO_history[idx][0] == GROUP_COL:
                    col_idx = BIO_history[idx][1][-1]
                    eval_output.append(col_idx in groupBy_cols)
                    idx += 1

            elif seg_id == GROUP_NHAV:
                eval_output.append(len(g_sql['having']) == 0)
                idx += 1

            elif seg_id == HAV_COL:
                if len(g_sql['having']) == 0:
                    eval_output.append(False)
                    idx += 1
                    while idx < len(BIO_history) and BIO_history[idx][0] in {HAV_COL, HAV_AGG, HAV_OP, HAV_ROOT_TERM}:
                        eval_output.append(False)
                        idx += 1

                else:
                    col_op_root, and_or = self.parse_where_having(g_sql['having'])

                    while idx < len(BIO_history) and BIO_history[idx][0] == HAV_COL:
                        col_idx = BIO_history[idx][1][-1]
                        eval_output.append(col_idx in set([col for (col,_), _, _ in col_op_root]))
                        idx += 1
                    if idx == len(BIO_history): break  # check partial end

                    assert BIO_history[idx][0] == HAV_AGG

                    while idx < len(BIO_history) and BIO_history[idx][0] == HAV_AGG:  # for all HAV_COLs
                        col_idx, agg_name = BIO_history[idx][1:3]
                        eval_output.append((col_idx, agg_name) in set([col_agg for col_agg, _, _ in col_op_root]))
                        idx += 1

                        while idx < len(BIO_history) and BIO_history[idx][0] == HAV_OP:  # for each col_idx
                            col_idx = BIO_history[idx][1][0][-1]
                            agg_name = BIO_history[idx][1][1]
                            op_name = BIO_history[idx][2]
                            eval_output.append(((col_idx, agg_name), op_name) in set([(col_agg, op) for col_agg, op, _ in col_op_root]))
                            idx += 1

                        while idx < len(BIO_history) and BIO_history[idx][0] == HAV_ROOT_TERM:
                            col_idx = BIO_history[idx][1][0][-1]
                            agg_name = BIO_history[idx][1][1]
                            op_name = BIO_history[idx][2]
                            root_term = BIO_history[idx][3]
                            bool_matched_col_op = False
                            for col_agg, op, (val1, val2) in col_op_root:
                                if (col_idx, agg_name) == col_agg and op == op_name:
                                    if not isinstance(val1, dict) and (val2 is None or not isinstance(val2, dict)):
                                        truth = 'terminal'
                                        if root_term == truth:
                                            eval_output.append(True)
                                            idx += 1
                                        else:
                                            eval_output.append(False)
                                            idx += 1
                                            # consider terms associated with the nested root as wrong
                                            end_nested_idx = self.helper_find_closest_fw(BIO_history, idx, tgt_name=END_NESTED)
                                            if end_nested_idx == -1: end_nested_idx = len(BIO_history) - 1
                                            # eval.extend([False] * (end_nested_idx - idx + 1))
                                            eval_output.extend([False if BIO_history[idx][0] != OUTSIDE else None for idx in
                                                         range(idx, (end_nested_idx + 1))])
                                            idx = end_nested_idx + 1
                                    else:
                                        assert isinstance(val1, dict) and val2 is None
                                        truth = 'root'
                                        if root_term != truth:  # root_term = terminal
                                            eval_output.append(False)
                                            idx += 1
                                        else:
                                            eval_output.append(True)
                                            idx += 1
                                            # evaluate nested component
                                            end_nested_idx = self.helper_find_closest_fw(BIO_history, idx,
                                                                                         tgt_name=END_NESTED)
                                            if end_nested_idx == -1: end_nested_idx = len(BIO_history) - 1
                                            idx, nested_eval = self.compare(val1, idx, BIO_history[:(end_nested_idx + 1)])
                                            eval_output.extend(nested_eval)
                                    bool_matched_col_op = True
                                    break

                            if not bool_matched_col_op: # cannot find matched (col_agg, op)
                                eval_output.append(False)
                                idx += 1
                                if root_term == 'root':
                                    # consider terms associated with the nested root as wrong
                                    end_nested_idx = self.helper_find_closest_fw(BIO_history, idx, tgt_name=END_NESTED)
                                    if end_nested_idx == -1: end_nested_idx = len(BIO_history) - 1
                                    # eval.extend([False] * (end_nested_idx - idx + 1))
                                    eval_output.extend([False if BIO_history[idx][0] != OUTSIDE else None for idx in
                                                 range(idx, (end_nested_idx + 1))])
                                    idx = end_nested_idx + 1

            elif seg_id == ORDER_COL:
                if len(g_sql['orderBy']) == 0:
                    while idx < len(BIO_history) and BIO_history[idx][0] in {ORDER_COL, ORDER_AGG, ORDER_DESC_ASC_LIMIT}:
                        eval_output.append(False)
                        idx += 1

                else:
                    col_agg_other = self.parse_orderBy_limit(g_sql['orderBy'], g_sql['limit'])
                    while idx < len(BIO_history) and BIO_history[idx][0] == ORDER_COL:
                        col_idx = BIO_history[idx][1][-1]
                        eval_output.append(col_idx in set([col for (col,_), _ in col_agg_other]))
                        idx += 1
                    if idx == len(BIO_history): break  # check partial end

                    assert BIO_history[idx][0] == ORDER_AGG
                    while idx < len(BIO_history) and BIO_history[idx][0] == ORDER_AGG:
                        # ORDER_AGG
                        col_idx = BIO_history[idx][1][-1]
                        agg_name = BIO_history[idx][2]
                        eval_output.append((col_idx, agg_name) in set([col_agg for col_agg, _ in col_agg_other]))
                        idx += 1
                        if idx == len(BIO_history): break  # check partial end

                        # ORDER_DESC_ASC_LIMIT
                        assert BIO_history[idx][0] == ORDER_DESC_ASC_LIMIT
                        col_idx = BIO_history[idx][1][0][-1]
                        agg_name = BIO_history[idx][1][1]
                        desc_asc_limit = BIO_history[idx][2]
                        eval_output.append(((col_idx, agg_name), desc_asc_limit) in col_agg_other)
                        idx += 1

            else:
                raise Exception("Invalid id {} at idx {} in history:\n{}".format(seg_id, idx, BIO_history))

        return idx, eval_output

