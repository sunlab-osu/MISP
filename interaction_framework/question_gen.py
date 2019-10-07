# question generation for interactive text2SQL
from user_study_utils import *

# from supermodel
KW_OPS = ('where','groupBy','orderBy')

# For interaction semantic unit tags
SELECT_COL = 'SELECT_COL'
SELECT_AGG = 'SELECT_AGG'
WHERE_COL = 'WHERE_COL'
WHERE_OP = 'WHERE_OP'
WHERE_ROOT_TERM = 'WHERE_ROOT_TERM'
WHERE_VAL = 'WHERE_VAL' # for models with value prediction
ANDOR = 'ANDOR'
GROUP_COL = 'GROUP_COL'
GROUP_NHAV = 'GROUP_NHAV'
HAV_COL = 'HAV_COL'
HAV_AGG = 'HAV_AGG'
HAV_OP = 'HAV_OP'
HAV_ROOT_TERM = 'HAV_ROOT_TERM'
ORDER_COL = 'ORDER_COL'
ORDER_AGG = 'ORDER_AGG'
ORDER_DESC_ASC_LIMIT = 'DESC_ASC_LIMIT'
IUEN = 'IUEN'
MISSING_KW = 'MISSING_KW'
MISSING_COL = 'MISSING_COL'
REDUNDANT_COL = 'REDUNDANT_COL'
MISSING_AGG = 'MISSING_AGG'
REDUNDANT_AGG = 'REDUNDANT_AGG'
MISSING_OP = 'MISSING_OP'
REDUNDANT_OP = 'REDUNDANT_OP'
OUTSIDE = "O"
END_NESTED = "##END_NESTED##"


# Tagging logic
# Each col is a tuple of (tab_name, col_name, col_idx)
#
# ('O', ('root', None), 1.0,), ('IUEN', 'none', prob,)
#
# When it is 'intersect'/'union'/'except' in IUEN:
# ('IUEN', 'intersect'/'union'/'except', prob,), <- here's the main sql-> ('O', '##END_NESTED##', 1.0, None)
# <-followed by the nested sql to intersect/union/except with->
#
# SELECT COL:
# (O, "select",), (SELECT_COL, col1,), (SELECT_COL, col2,), .., (MISSING_COL, "select", col_num, cols, col_num prob, [col probs],)
# For each col:
# (SELECT_AGG, col1, agg1,), (SELECT_AGG, col1, agg2,), .., (MISSING_AGG, "select", col1, agg_num, aggs, agg_num prob, [agg probs],)#note for SELECT_AGG, agg being "none_agg" will not be shown
#
# WHERE:
# (O, "where",), (WHERE_COL, col1,), (WHERE_COL, col2,) .., (MISSING_COL, "where", col_num, cols, col_num prob, [col probs],), (ANDOR, "and"/"or", col_names, andor_prob,)#when multiple cols selected
# For each col:
# (WHERE_OP, (col1,), op1,), (WHERE_OP, (col1,), op2,), (MISSING_OP, "where", (col1,), op_num, ops, op_num prob, [op probs],)
# For each (col, op):
# (WHERE_ROOTTERM, (col,), op, root/terminal,) for Spider or (WHERE_VAL, (col,), op, val_str) for WikiSQL
#
# GROUP:
# (O, "groupBy",), (GROUP_COL, col1,), (GROUP_COL, col2,), .., (MISSING_COL, "groupBy", col_num, cols, col_num prob, [col probs],)
# (GROUP_NHAV, "none_having",) #end of groupBy
# or (O, "having",), (HAV_COL, col1), (HAV_COL, col2), .., (MISSING_COL, "having", col_num, cols, col_num_prob, [col probs],)
# For each col:
# (HAV_AGG, col, agg/none_agg,) (HAV_OP, (col, agg), op1,), (HAV_OP, (col, agg), op2,), (MISSING_OP, "having", (col, agg), op_num, ops, op_num prob, [op probs],)
# For each op:
# (HAV_ROOTTERM, (col, agg), op, root/terminal,)
#
# ORDER:
# (O, "orderBy",), (ORDER_COL, col1,), (ORDER_COL, col2,), .., (MISSING_COL, "orderBy", col_num, cols, col_num prob, [col probs],)
# For each col:
# (ORDER_AGG, col, agg/none_agg), (ORDER_DESC_ASC_LIMIT, (col, agg), desc_asc_limit, prob)
#
# (MISSING_KW, kw_num, kws, kw_num_prob,)


class QuestionGenerator:
    def __init__(self):
        self.agg = {"avg": "average value in", "count": "number of", "sum": "sum of values in",
                    "min": "minimum value in", "max": "maximum value in"}
        self.where_op = {"like": "follows a pattern like", "not in": "is NOT IN", ">": "is greater than",
                         "<": "is less than", "=": "equals to", ">=": "is greater than or equivalent to",
                         "<=": "is less than or equivalent to", "!=": "does not equal to",
                         "in": "is IN", "between": "is between"}
        self.desc_asc_limit = {("desc", False): "in descending order", ("asc", False): "in ascending order",
                               ("desc", True): "in descending order and limited to top N",
                               ("asc", True): "in ascending order and limited to top N"}
        # self.kw2descrip = {"where": "considers conditions in query (e.g., some attributes =/>/< some values)",
        #                    "orderBy": "needs to order (intermediate) results",
        #                    "groupBy": "needs to group table items by (calculations of) some attributes"}

    def agg_col_tab_description(self, col_name, tab_name, agg=None, bool_having=False):
        if agg is not None:
            agg = agg.lower()

        if col_name == "*":
            if bool_having:
                col_descrip = "grouped items"
            else:
                col_descrip = "all items" #"all " + tab_name
            if agg is None:
                return col_descrip #"(max/min/sum/count/.. of) " + col_descrip
            else:
                return self.agg.get(agg, agg) + " " + col_descrip
        else:
            if tab_name is None:
                col_descrip = bcolors.BLUE + "\"" + col_name + "\"" + bcolors.ENDC
            else:
                col_descrip = bcolors.BLUE + "\"" + col_name + "\"" + bcolors.ENDC + " in the table " +\
                              bcolors.YELLOW + "\"" + tab_name + "\"" + bcolors.ENDC
            # agg_descrip = (self.agg.get(agg, agg)) if (agg is not None and len(agg)) else ""
            # return agg_descrip + col_descrip
            if agg is not None and len(agg):
                return self.agg.get(agg, agg) + " " + col_descrip
            else:
                return "the table attribute " + col_descrip

    def group_by_agg_col_tab_description(self, col_name, tab_name):
        return "items in the table " + bcolors.YELLOW + "\"" + tab_name + "\"" + bcolors.ENDC +\
               " based on the table attribute " + bcolors.BLUE + "\"" + col_name + "\"" + bcolors.ENDC

    def where_cond_descrip(self, col_name, tab_name, where_op, root_or_terminal, agg=None, bool_having=False):
        agg_col_tab = self.agg_col_tab_description(col_name, tab_name, agg=agg, bool_having=bool_having)
        where_op_descrip = self.where_op.get(where_op, where_op)

        if root_or_terminal == "terminal":
            prefix = agg_col_tab + " " + where_op_descrip + " "
            if where_op in {"in", "not in"}:
                return prefix + bcolors.UNDERLINE + "a set of given literal values" + bcolors.ENDC
            elif where_op == "between":
                return prefix + bcolors.UNDERLINE + "two given literal values" + bcolors.ENDC
            else:
                return prefix + bcolors.UNDERLINE + "a given literal value (e.g., number 5, string \"France\")" + bcolors.ENDC
        elif root_or_terminal == "root":
            prefix = agg_col_tab + " " + where_op_descrip + " "
            if where_op in {"in", "not in"}:
                return prefix + bcolors.UNDERLINE + "a set of values to be calculated" + bcolors.ENDC
            else:
                return prefix + bcolors.UNDERLINE + "a value to be calculated" + bcolors.ENDC
        else:
            raise ValueError("Invalid root_or_terminal=%s!" % root_or_terminal)

    def select_col_question(self, col_name, tab_name):
        return "Does the system need to return information about %s?" % self.agg_col_tab_description(col_name, tab_name)

    def select_agg_question(self, col_name, tab_name, src_agg):
        if src_agg == "none_agg":
            # agg_info = "pure value (in contrast to calculations) of"
            return "Does the system need to return a value " + bcolors.UNDERLINE + "after" + bcolors.ENDC + " any mathematical calculations " \
                   "(e.g., maximum, minimum, sum, average, count) on %s?" % self.agg_col_tab_description(col_name, tab_name)
        else:
            src_agg = src_agg.lower()
            return "Does the system need to return %s?" % self.agg_col_tab_description(col_name, tab_name, agg=src_agg)

    def where_col_question(self, col_name, tab_name):
        return "Does the system need to consider any conditions about %s?" %\
               self.agg_col_tab_description(col_name, tab_name)

    def andor_question(self, and_or, selected_cols_info):
        if and_or == "and":
            return "Do the conditions about %s hold at the same time?" % selected_cols_info
        elif and_or == "or":
            return "Do the conditions about %s hold alternatively?" % selected_cols_info
        else:
            raise ValueError("Invalid and_or=%s!" % and_or)

    def where_op_question(self, agg_col_tab_name, op):
        return "The system considers the following condition: %s %s a value. " % (
            agg_col_tab_name, self.where_op.get(op, op)) + "Is the condition correct?"
        # return "Does the system need to consider the condition \"%s %s a value\"?" % (
        #     agg_col_tab_name, self.where_op.get(op, op))

    def root_terminal_question(self, col_name, tab_name, where_op, root_or_terminal,
                               bool_having=False, agg=None, group_cols_info=None):
        agg_col_tab = self.agg_col_tab_description(col_name, tab_name, agg=agg, bool_having=bool_having)
        where_op_descrip = self.where_op.get(where_op, where_op)
        if bool_having:
            question = "Given that the system groups %s before doing any mathematical calculations (e.g., maximum, minimum, sum, average, count), " \
                       "the system considers the following condition: " % group_cols_info +\
                       agg_col_tab + " " + where_op_descrip
        else:
            question = "The system considers the following condition: " + agg_col_tab + " " + where_op_descrip

        if root_or_terminal == "terminal":
            if where_op in {"in", "not in"}:
                question += " a set of given literal values. "
            elif where_op == "between":
                question += " two given literal values. "
            else:
                question += " a given literal value (e.g., number 5, string \"France\"). "
        elif root_or_terminal == "root":
            if where_op in {"in", "not in"}:
                question += " a set of values to be calculated. "
            else:
                question += " a value to be calculated. "

        return question + "Is this condition correct?"

    def where_val_question(self, col_name, tab_name, where_op, val_str):
        question = "The system considers the following condition: %s %s \"%s\". " % (
            self.agg_col_tab_description(col_name, tab_name), self.where_op.get(where_op, where_op), val_str) +\
                   "Is the condition correct?"
        # question = "Does the system need to consider the condition \"%s %s \"%s\"\"?" % (
        #     self.agg_col_tab_description(col_name, tab_name), self.where_op.get(where_op, where_op), val_str)
        return question

    def group_col_question(self, col_name, tab_name):
        assert tab_name is not None
        return "Does the system need to group %s" % self.group_by_agg_col_tab_description(col_name, tab_name) +\
               " before doing any mathematical calculations (e.g., maximum, minimum, sum, average, count)?"

    def group_none_having_question(self, group_by_cols_info):
        # trigger for "none_having"
        return "Given that the system groups %s before doing any mathematical calculations (e.g., maximum, minimum, sum, average, count), " \
               "does the system need to consider any conditions?" % group_by_cols_info

    def have_col_question(self, group_by_cols_info, col_name, tab_name):
        return "Given that the system groups %s before doing any mathematical calculations (e.g., maximum, minimum, sum, average, count), " \
               "does the system need to consider any conditions about %s" % (
               group_by_cols_info, self.agg_col_tab_description(col_name, tab_name, bool_having=True))

    def have_agg_question(self, group_by_cols_info, col_name, tab_name, src_agg):
        src_agg = src_agg.lower()
        if src_agg == "none_agg":
            "Given that the system groups %s before doing any mathematical calculations (e.g., maximum, minimum, sum, average, count), " % group_by_cols_info +\
            "does the system need to return a value " + bcolors.UNDERLINE + "after" + bcolors.ENDC + " any mathematical calculations on %s?" % (
                self.agg_col_tab_description(col_name, tab_name, bool_having=True))
        else:
            return "Given that the system groups %s before doing any mathematical calculations (e.g., maximum, minimum, sum, average, count), " \
                   "Does the system need to consider any conditions about %s?" % (
                group_by_cols_info, self.agg_col_tab_description(col_name, tab_name, agg=src_agg, bool_having=True))

    def have_op_question(self, group_by_cols_info, col_name, tab_name, op, agg=None):
        return "The system groups %s before doing any mathematical calculations (e.g., maximum, minimum, sum, average, count), " \
               "then considers the following condition: %s %s a value. " \
               "Is the condition correct?" % (
            group_by_cols_info, self.agg_col_tab_description(col_name, tab_name, agg=agg, bool_having=True),
            self.where_op.get(op, op))

    def order_col_question(self, col_name, tab_name):
        return "Does the system need to order results based on %s?" % self.agg_col_tab_description(col_name, tab_name)

    def order_agg_question(self, col_name, tab_name, src_agg):
        src_agg = src_agg.lower()

        if src_agg == "none_agg":
            return "Does the system need to order results based on a value " + bcolors.UNDERLINE + "after" + bcolors.ENDC +\
                   " any mathematical calculations (e.g., maximum, minimum, sum, average, count) on %s?" %\
                   self.agg_col_tab_description(col_name, tab_name)
        else:
            return "Does the system need to order results based on %s?" % self.agg_col_tab_description(col_name, tab_name, agg=src_agg)

    def order_desc_asc_limit_question(self, col_name, tab_name, desc_asc_limit, agg=None):
        return "Given that the system orders the results based on %s, does it need to be %s?" % (
            self.agg_col_tab_description(col_name, tab_name, agg=agg), self.desc_asc_limit[desc_asc_limit])

    def iuen_question(self, iuen):
        if iuen == "except":
            return "Does the system need to return information satisfying some cases BUT NOT others?\n" \
                   "e.g., Find all airlines that have flights from airport 'CVO' BUT NOT from 'APG'."
        elif iuen == "union":
            return "Does the system need to return information satisfying either some cases OR others?\n" \
                   "e.g., What are the id and names of the countries which have more than 3 car makers OR produce the 'fiat' model?"
        elif iuen == "intersect":
            return "Does the system need to return information satisfying BOTH some cases AND the others AT THE SAME TIME?\n" \
                   "e.g., Which district has BOTH stores with less than 3000 products AND stores with more than 10000 products?"
        else:
            return "Does the system need to return information that meets one of the three situations: \n" \
                   "(1) satisfying some cases BUT NOT others, e.g., Find all airlines that have flights from airport 'CVO' BUT NOT from 'APG'.\n" \
                   "(2) satisfying either some cases OR others, e.g., What are the id and names of the countries which have more than 3 car makers OR produce the 'fiat' model?\n" \
                   "(3) satisfying BOTH some cases AND the others AT THE SAME TIME, e.g., Which district has BOTH stores with less than 3000 products AND stores with more than 10000 products?\n" \
                   "(Note: your situation is very likely to fall into NONE of the above - suggest to answer 'no')"

    def where_having_nested_question(self, col_name, tab_name, where_op, right_question, agg=None, bool_having=False):
        revised_right_question = right_question[:-1] + " for this calculation?"
        return "Given that %s, answer the following question to help the system to calculate " % self.where_cond_descrip(
            col_name, tab_name, where_op, "root", agg=agg, bool_having=bool_having) +\
               bcolors.UNDERLINE + "the value(s)" + bcolors.ENDC + ":\n%s" % revised_right_question

    def helper_find_closest_bw(self, history, start_idx, tgt_name=None, tgt_id=None):
        skip_nested = []
        idx = start_idx
        while idx > 0:
            if len(skip_nested) > 0:
                if "root" in history[idx]:
                    _ = skip_nested.pop()
                idx -= 1
            else:
                if (tgt_name is not None and tgt_name in history[idx]) or\
                        (tgt_id is not None and history[idx][0] == tgt_id): #include tgt_name == END_NESTED
                    return idx
                elif END_NESTED in history[idx]:
                    skip_nested.append(idx)
                    idx -= 1
                else:
                    idx -= 1

        return -1 # not found

    def helper_find_indices(self, BIO_history, tgt_names):
        return [idx for idx in range(len(BIO_history)) if BIO_history[idx][1] in tgt_names]

    @staticmethod
    def semantic_unit_segment(BIO_history):
        BIO_lists, seg_pointers = [], []
        for idx, BIO_item in enumerate(BIO_history):
            if BIO_item[0] != OUTSIDE:
                BIO_lists.append([BIO_item])
                seg_pointers.append(idx)
        return BIO_lists, seg_pointers

    def question_generation(self, BIO_list, BIO_history, pointer):
        """
        Generating questions.
        :param BIO_list:
        :param BIO_history:
        :param pointer:
        :return: question and answer_sheet = {'yes/no': (True for continue decoding,
        0->drop, 1-> flip, 2->re-decode)}
        """
        assert BIO_history[pointer] == BIO_list[0]
        option = None

        template_id = BIO_list[0][0].split('-')[0]
        if template_id == SELECT_COL:
            tab_col_item, = BIO_list
            question = self.select_col_question(tab_col_item[1][1], tab_col_item[1][0])
            answer_sheet = {'yes': (True, None), 'no': (False, 0)} #no->drop

        elif template_id == SELECT_AGG:
            col, agg = BIO_list[0][1:3]
            # assert agg != "none_agg"
            question = self.select_agg_question(col[1], col[0], agg)
            if agg == "none_agg":
                answer_sheet = {'no': (True, None), 'yes': (False, 0)}
            else:
                answer_sheet = {'yes': (True, None), 'no': (False, 0)}

        elif template_id == WHERE_COL:
            tab_col_item, = BIO_list
            question = self.where_col_question(tab_col_item[1][1], tab_col_item[1][0])
            answer_sheet = {'yes': (True, None), 'no': (False, 0)}

        elif template_id == ANDOR:
            and_or, cols = BIO_list[0][1:3]
            cols_info = [self.agg_col_tab_description(col[1], col[0]) for col in cols]
            question = self.andor_question(and_or, ", ".join(cols_info))
            answer_sheet = {'yes': (True, None), 'no': (False, 1)} #flip

        elif template_id == WHERE_OP:
            ((col,), op_name) = BIO_list[0][1:3]
            question = self.where_op_question(self.agg_col_tab_description(col[1], col[0]), op_name)
            answer_sheet = {'yes': (True, None), 'no': (False, 0)}

        elif template_id == WHERE_ROOT_TERM:
            ((col,), op_name, root_term) = BIO_list[0][1:4]
            question = self.root_terminal_question(col[1], col[0], op_name, root_term)
            answer_sheet = {'yes': (True, None), 'no': (False, 1)} #no->flip

        elif template_id == WHERE_VAL:
            ((col,), op_name, val_item) = BIO_list[0][1:4]
            question = self.where_val_question(col[1], col[0], op_name, val_item[-1])
            answer_sheet = {'yes': (True, None), 'no': (False, 2)}

        elif template_id == GROUP_COL:
            tab_col_item, = BIO_list
            question = self.group_col_question(tab_col_item[1][1], tab_col_item[1][0])
            answer_sheet = {'yes': (True, None), 'no': (False, 0)}  # no->drop

        elif template_id == GROUP_NHAV:
            groupBy_cols = []
            # idx = pointer - 2
            idx = self.helper_find_closest_bw(BIO_history, pointer - 1, tgt_id=GROUP_COL)
            while idx > 0:
                if BIO_history[idx][0] == GROUP_COL:
                    # groupBy_cols.append(self.agg_col_tab_description(BIO_history[idx][1][1], BIO_history[idx][1][0]))
                    groupBy_cols.append(self.group_by_agg_col_tab_description(BIO_history[idx][1][1], BIO_history[idx][1][0]))
                else:
                    break
                idx -= 1
            question = self.group_none_having_question(", ".join(groupBy_cols))
            answer_sheet = {'yes': (False, 1), 'no': (True, None)}

        elif template_id == HAV_COL:
            tab_col_item, = BIO_list

            closest_group_col_idx = self.helper_find_closest_bw(BIO_history, pointer - 1, tgt_id=GROUP_COL)
            groupBy_cols = []
            idx = closest_group_col_idx
            while idx > 0:
                if BIO_history[idx][0] == GROUP_COL:
                    groupBy_cols.append(self.group_by_agg_col_tab_description(BIO_history[idx][1][1], BIO_history[idx][1][0]))
                else:
                    break
                idx -= 1
            question = self.have_col_question(", ".join(groupBy_cols), tab_col_item[1][1], tab_col_item[1][0])
            answer_sheet = {'yes': (True, None), 'no':(False, 0)}

        elif template_id == HAV_AGG:
            col, agg = BIO_list[0][1:3]

            closest_group_col_idx = self.helper_find_closest_bw(BIO_history, pointer - 1, tgt_id=GROUP_COL)
            groupBy_cols = []
            idx = closest_group_col_idx
            while idx > 0:
                if BIO_history[idx][0] == GROUP_COL:
                    groupBy_cols.append(self.group_by_agg_col_tab_description(BIO_history[idx][1][1], BIO_history[idx][1][0]))
                else:
                    break
                idx -= 1

            question = self.have_agg_question(", ".join(groupBy_cols), col[1], col[0], agg)

            if agg == "none_agg":
                answer_sheet = {'no': (True, None), 'yes': (False, 2)}
            else:
                answer_sheet = {'yes': (True, None), 'no': (False, 2)}

        elif template_id == HAV_OP:
            (col, agg), op = BIO_list[0][1:3]

            closest_group_col_idx = self.helper_find_closest_bw(BIO_history, pointer - 1, tgt_id=GROUP_COL)
            groupBy_cols = []
            idx = closest_group_col_idx
            while idx > 0:
                if BIO_history[idx][0] == GROUP_COL:
                    groupBy_cols.append(self.group_by_agg_col_tab_description(BIO_history[idx][1][1], BIO_history[idx][1][0]))
                else:
                    break
                idx -= 1

            question = self.have_op_question(", ".join(groupBy_cols), col[1], col[0], op,
                                             agg=None if agg == "none_agg" else agg)
            answer_sheet = {'yes': (True, None), 'no': (False, 0)}

        elif template_id == HAV_ROOT_TERM:
            (col, agg), op, root_term = BIO_list[0][1:4]

            closest_group_col_idx = self.helper_find_closest_bw(BIO_history, pointer - 1, tgt_id=GROUP_COL)
            groupBy_cols = []
            idx = closest_group_col_idx
            while idx > 0:
                if BIO_history[idx][0] == GROUP_COL:
                    groupBy_cols.append(self.group_by_agg_col_tab_description(BIO_history[idx][1][1], BIO_history[idx][1][0]))
                else:
                    break
                idx -= 1

            question = self.root_terminal_question(col[1], col[0], op, root_term, bool_having=True,
                                                   agg=None if agg == "none_agg" else agg,
                                                   group_cols_info=", ".join(groupBy_cols))
            answer_sheet = {'yes': (True, None), 'no': (False, 1)}

        elif template_id == ORDER_COL:
            tab_col_item, = BIO_list
            question = self.order_col_question(tab_col_item[1][1], tab_col_item[1][0])
            answer_sheet = {'yes': (True, None), 'no': (False, 0)}

        elif template_id == ORDER_AGG:
            col, agg = BIO_list[0][1:3]
            question = self.order_agg_question(col[1], col[0], agg)
            if agg == "none_agg":
                answer_sheet = {'no': (True, None), 'yes': (False, 2)}
            else:
                answer_sheet = {'yes': (True, None), 'no': (False, 2)}

        elif template_id == ORDER_DESC_ASC_LIMIT:
            (col, agg), desc_asc_limit = BIO_list[0][1:3]
            question = self.order_desc_asc_limit_question(col[1], col[0], desc_asc_limit,
                                                          agg=None if agg == "none_agg" else agg)
            answer_sheet = {'yes': (True, None), 'no': (False, 2)}

        # elif template_id == REDUNDANT_COL:
        #     kw, col_num, cols = BIO_list[0][1:4]
        #     cols_info = [self.agg_col_tab_descrip(col[1], col[0], bool_having=(kw == "having")) for col in cols]
        #     if kw == "having":
        #         closest_group_col_idx = self.helper_find_closest_bw(BIO_history, pointer - 1, tgt_id=GROUP_COL)
        #         groupBy_cols = []
        #         idx = closest_group_col_idx
        #         while idx > 0:
        #             if BIO_history[idx][0] == GROUP_COL:
        #                 groupBy_cols.append(self.agg_col_tab_descrip(BIO_history[idx][1][1], BIO_history[idx][1][0]))
        #             else:
        #                 break
        #             idx -= 1
        #         groupBy_cols_info = ", ".join(groupBy_cols)
        #     else:
        #         groupBy_cols_info = None
        #     question = self.missing_col_question("remove", kw, ", ".join(cols_info), context_group_col=groupBy_cols_info)
        #     answer_sheet = {'no': (True, None), 'yes': (False, 0)}
        #     option = cols_info
        #
        # elif template_id == MISSING_COL:
        #     kw, col_num, cols = BIO_list[0][1:4]
        #     cols_info = [self.agg_col_tab_descrip(col[1], col[0], bool_having=(kw == "having")) for col in cols]
        #     if kw == "having":
        #         closest_group_col_idx = self.helper_find_closest_bw(BIO_history, pointer - 1, tgt_id=GROUP_COL)
        #         groupBy_cols = []
        #         idx = closest_group_col_idx
        #         while idx > 0:
        #             if BIO_history[idx][0] == GROUP_COL:
        #                 groupBy_cols.append(self.agg_col_tab_descrip(BIO_history[idx][1][1], BIO_history[idx][1][0]))
        #             else:
        #                 break
        #             idx -= 1
        #         groupBy_cols_info = ", ".join(groupBy_cols)
        #     else:
        #         groupBy_cols_info = None
        #     question = self.missing_col_question("add", kw, ", ".join(cols_info), context_group_col=groupBy_cols_info)
        #     answer_sheet = {'no': (True, None), 'yes': (False, 2)}
        #
        # elif template_id == REDUNDANT_AGG:
        #     kw, col, agg_num, aggs = BIO_list[0][1:5]
        #     assert kw == "select"
        #     if len(aggs) == 0:
        #         aggs_info = None
        #     else:
        #         aggs_info = [self.agg.get(agg, agg) for agg in aggs]
        #     question = self.missing_agg_question("remove", self.agg_col_tab_descrip(col[1], col[0]),
        #                                           ", ".join(aggs_info) if aggs_info is not None else None)
        #     answer_sheet = {'no': (True, None), 'yes': (False, 0)}
        #     option = aggs_info
        #
        # elif template_id == MISSING_AGG:
        #     kw, col, agg_num, aggs = BIO_list[0][1:5]
        #     assert kw == "select"
        #     if len(aggs) == 0:
        #         aggs_info = None
        #     else:
        #         aggs_info = [self.agg.get(agg, agg) for agg in aggs]
        #
        #     question = self.missing_agg_question("add", self.agg_col_tab_descrip(col[1], col[0]),
        #                                           ", ".join(aggs_info) if aggs_info is not None else None)
        #     answer_sheet = {'no': (True, None), 'yes': (False, 2)}
        #
        # elif template_id == REDUNDANT_OP:
        #     kw, col_agg, op_num, ops = BIO_list[0][1:5]
        #
        #     if kw == "where":
        #         col, = col_agg
        #         agg = None
        #         context_group_col = None
        #     else:
        #         col, agg = col_agg
        #         closest_group_col_idx = self.helper_find_closest_bw(BIO_history, pointer - 1, tgt_id=GROUP_COL)
        #         groupBy_cols = []
        #         idx = closest_group_col_idx
        #         while idx > 0:
        #             if BIO_history[idx][0] == GROUP_COL:
        #                 groupBy_cols.append(self.agg_col_tab_descrip(BIO_history[idx][1][1], BIO_history[idx][1][0]))
        #             else:
        #                 break
        #             idx -= 1
        #         context_group_col = ", ".join(groupBy_cols)
        #
        #     ops_info = [self.where_op.get(op, op) for op in ops]
        #     question = self.missing_where_having_op_question("remove", kw, self.agg_col_tab_descrip(
        #         col[1], col[0], agg=agg, bool_having=(kw == "having")), " | ".join(ops_info), context_group_col=context_group_col)
        #     answer_sheet = {'no': (True, None), 'yes': (False, 0)}
        #     option = ops_info
        #
        # elif template_id == MISSING_OP:
        #     kw, col_agg, op_num, ops = BIO_list[0][1:5]
        #
        #     if kw == "where":
        #         col, = col_agg
        #         agg = None
        #         context_group_col = None
        #     else:
        #         col, agg = col_agg
        #         closest_group_col_idx = self.helper_find_closest_bw(BIO_history, pointer - 1, tgt_id=GROUP_COL)
        #         groupBy_cols = []
        #         idx = closest_group_col_idx
        #         while idx > 0:
        #             if BIO_history[idx][0] == GROUP_COL:
        #                 groupBy_cols.append(self.agg_col_tab_descrip(BIO_history[idx][1][1], BIO_history[idx][1][0]))
        #             else:
        #                 break
        #             idx -= 1
        #         context_group_col = ", ".join(groupBy_cols)
        #
        #     ops_info = [self.where_op.get(op, op) for op in ops]
        #
        #     question = self.missing_where_having_op_question("add", kw, self.agg_col_tab_descrip(
        #         col[1], col[0], agg=agg, bool_having=(kw == "having")), " | ".join(ops_info), context_group_col=context_group_col)
        #     answer_sheet = {'no': (True, None), 'yes': (False, 2)}
        #
        # elif template_id == MISSING_KW:
        #     kw_num, kws = BIO_list[0][1:3]
        #     remaining_kws = set(KW_OPS) - set(kws)
        #     if len(remaining_kws) == 0:
        #         question = ""
        #         answer_sheet = None
        #     else:
        #         question = self.missing_kw_question(kws)
        #         answer_sheet = {'no': (True, None), 'yes': (False, 2)}
        #         option = [("The system " + self.kw2descrip[kw]) for kw in remaining_kws]

        elif template_id == IUEN:
            iuen = BIO_list[0][1]
            question = self.iuen_question(iuen)
            if iuen == "none":
                answer_sheet = {'no': (True, None), 'yes': (False, 2)}
            else:
                answer_sheet = {'yes': (True, None), 'no': (False, 2)}

        else:
            print("WARNING: Unknown template_id %s" % template_id)
            question = ""
            answer_sheet = None

        # check nested WHERE/HAVING condition
        closest_root_idx = self.helper_find_closest_bw(BIO_history, pointer - 1, tgt_name="root")
        if closest_root_idx == -1: # not found, not nested
            return question, answer_sheet, option
        else:
            root_tag = BIO_history[closest_root_idx][0]
            if root_tag == OUTSIDE: # IUEN != none
                return question, answer_sheet, option
            else:
                closest_end_nested_idx = self.helper_find_closest_bw(BIO_history, pointer - 1, tgt_name=END_NESTED)
                if closest_end_nested_idx != -1 and closest_end_nested_idx > closest_root_idx:
                    return question, answer_sheet, option
                if root_tag == WHERE_ROOT_TERM:
                    (col,), op = BIO_history[closest_root_idx][1:3]
                    question = self.where_having_nested_question(col[1], col[0], op, question)
                elif root_tag == HAV_ROOT_TERM:
                    (col, agg), op = BIO_history[closest_root_idx][1:3]
                    question = self.where_having_nested_question(col[1], col[0], op, question,
                                                                 agg=agg if agg != 'none_agg' else None,
                                                                 bool_having=True)
                else:
                    raise ValueError("Unexpected nested condition: BIO_history: {}\nPointer: {}, closest root: {}.".format(
                        BIO_history, pointer, BIO_history[closest_root_idx]
                    ))
                return question, answer_sheet, option


if __name__ == "__main__":
    nl = "What are all the song names by singers who are older than average?"
    current_sql = "{'sql': {'where': [(u'singer', u'age', 13), '>', {'sql': {'where': [(u'singer', u'name', 9), '=', 'terminal'], 'select': [(u'singer', u'age', 13), 'avg']}}], 'select': [(u'singer', u'song name', 11), 'none_agg']}}"
    pred_sql = "SELECT Song_Name from Singer WHERE Age > (SELECT AVG(Age) FROM Singer WHERE Name = 'terminal')"
    history = [('O', ('root', None), 1.0, None), ('IUEN', 'none', 1.0, 0), ('O', 'select', 1.0, None), ('SELECT_COL', (u'singer', u'song name', 11), 0.9319478, 2), ('REDUNDANT_COL', 'select', 1, [(u'singer', u'song name', 11)], 1.0, [0.9319478], 2), ('MISSING_COL', 'select', 1, [(u'singer', u'song name', 11)], 1.0, [0.9319478], 2), ('REDUNDANT_AGG', 'select', (u'singer', u'song name', 11), 0, [], 1.0, [], 3), ('MISSING_AGG', 'select', (u'singer', u'song name', 11), 0, [], 1.0, [], 3), ('O', 'where', 0.9999995, 0.9999989, None), ('WHERE_COL', (u'singer', u'age', 13), 0.028576141, 4), ('REDUNDANT_COL', 'where', 1, [(u'singer', u'age', 13)], 0.9999924, [0.028576141], 4), ('MISSING_COL', 'where', 1, [(u'singer', u'age', 13)], 0.9999924, [0.028576141], 4), ('WHERE_OP', ((u'singer', u'age', 13),), '>', 0.9999256, 5), ('REDUNDANT_OP', 'where', ((u'singer', u'age', 13),), 1, ['>'], 1.0, [0.9999256], 5), ('MISSING_OP', 'where', ((u'singer', u'age', 13),), 1, ['>'], 1.0, [0.9999256], 5), ('WHERE_ROOT_TERM', ((u'singer', u'age', 13),), '>', 'root', 0.9999988, 6), ('IUEN', 'none', 1.0, 7), ('O', 'select', 1.0, None), ('SELECT_COL', (u'singer', u'age', 13), 0.75160456, 9), ('REDUNDANT_COL', 'select', 1, [(u'singer', u'age', 13)], 1.0, [0.75160456], 9), ('MISSING_COL', 'select', 1, [(u'singer', u'age', 13)], 1.0, [0.75160456], 9), ('SELECT_AGG', (u'singer', u'age', 13), 'avg', 1.0, 10), ('REDUNDANT_AGG', 'select', (u'singer', u'age', 13), 1, ['avg'], 0.9999993, [1.0], 10), ('MISSING_AGG', 'select', (u'singer', u'age', 13), 1, ['avg'], 0.9999993, [1.0], 10), ('O', 'where', 0.9998784, 0.99961936, None), ('WHERE_COL', (u'singer', u'name', 9), 0.06865124, 11), ('REDUNDANT_COL', 'where', 1, [(u'singer', u'name', 9)], 1.0, [0.06865124], 11), ('MISSING_COL', 'where', 1, [(u'singer', u'name', 9)], 1.0, [0.06865124], 11), ('WHERE_OP', ((u'singer', u'name', 9),), '=', 0.9999987, 12), ('REDUNDANT_OP', 'where', ((u'singer', u'name', 9),), 1, ['='], 1.0, [0.9999987], 12), ('MISSING_OP', 'where', ((u'singer', u'name', 9),), 1, ['='], 1.0, [0.9999987], 12), ('WHERE_ROOT_TERM', ((u'singer', u'name', 9),), '=', 'terminal', 0.999998, 13), ('MISSING_KW', 1, ['where'], 0.9998784, 8), ('O', '##END_NESTED##', 1.0, None), ('MISSING_KW', 1, ['where'], 0.9999995, 1)]
    segs, pointers = QuestionGenerator.semantic_unit_segment(history)
    q_gen = QuestionGenerator()
    for seg, pointer in zip(segs, pointers):
        question, answer_sheet, option = q_gen.question_generation(seg, history, pointer)
        print(pointer, seg)
        print("Question: {}\nAnswer sheet: {}\noption: {}".format(question, answer_sheet, option))
        print("-" * 50 + "\n")
