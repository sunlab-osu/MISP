# question generator
from .utils import *


class QuestionGenerator:
    """
    This is the class for question generation.
    """
    def __init__(self):
        # the seed lexicon
        self.agg_regular = {"avg": "the average value of",
                            "count": "the number of items in",
                            "sum": "the sum of values of",
                            "min": "the minimum value among items of",
                            "max": "the maximum value among items of"}
        self.agg_distinct = {"avg": "the average value of distinct items in",
                             "count": "the number of distinct items in",
                             "sum": "the sum of distinct values of",
                             "min": "the minimum value among distinct items of",
                             "max": "the maximum value among distinct items of"}
        self.agg_asterisk = {"avg": "the average value of items", # warning: this should not be triggered
                             "count": "the number of items",
                             "sum": "the sum of values", # warning: this should not be triggered
                             "min": "the minimum value among items", # warning: this should not be triggered
                             "max": "the maximum value among items"} # warning: this should not be triggered
        self.agg_asterisk_distinct = {"avg": "the average value of distinct items",  # warning: this should not be triggered
                                      "count": "the number of distinct items",
                                      "sum": "the sum of distinct values",  # warning: this should not be triggered
                                      "min": "the minimum value among distinct items",  # warning: this should not be triggered
                                      "max": "the maximum value among distinct items"}  # warning: this should not be triggered
        self.where_op = {"like": "follow", "not in": "be NOT IN", ">": "be greater than",
                         "<": "be less than", "=": "equal to", ">=": "be greater than or equal to",
                         "<=": "be less than or equal to", "!=": "be NOT equal to",
                         "in": "be IN", "between": "be between"}
        self.desc_asc_limit = {("desc", False): "in descending order", ("asc", False): "in ascending order",
                               ("desc", True): "in descending order and limited to top N",
                               ("asc", True): "in ascending order and limited to top N"}

    def agg_col_tab_description(self, col_name, tab_name, agg=None, bool_having=False, bool_distinct=False):
        if agg is not None:
            agg = agg.lower()

        if bool_distinct:
            assert agg is not None

        if col_name == "*":
            agg_descrip = "all items"
            if agg is not None and bool_distinct:
                agg_descrip = self.agg_asterisk_distinct[agg]
            elif agg is not None:
                agg_descrip = self.agg_asterisk[agg]

            tab_descrip = ""
            if bool_having:
                tab_descrip += " in each group"

            if tab_name is not None:
                tab_descrip += " in the table " + bcolors.YELLOW + "\"" + tab_name + "\"" + bcolors.ENDC

            return agg_descrip + tab_descrip
        else:
            agg_descrip = ""
            if agg is not None and bool_distinct:
                agg_descrip = self.agg_distinct[agg] + " "
            elif agg is not None:
                agg_descrip = self.agg_regular[agg] + " "

            col_descrip = "the table attribute " + bcolors.BLUE + "\"" + col_name + "\"" + bcolors.ENDC

            tab_descrip = " in the table"
            if tab_name is not None:
                tab_descrip += " " + bcolors.YELLOW + "\"" + tab_name + "\"" + bcolors.ENDC

            return agg_descrip + col_descrip + tab_descrip

    def group_by_agg_col_tab_description(self, col_name, tab_name):
        if tab_name is None:
            return "items based on the table attribute " + bcolors.BLUE + "\"" + col_name + "\"" + bcolors.ENDC
        else:
            return "items in the table " + bcolors.YELLOW + "\"" + tab_name + "\"" + bcolors.ENDC +\
                   " based on the table attribute " + bcolors.BLUE + "\"" + col_name + "\"" + bcolors.ENDC

    def select_col_question(self, col_name, tab_name):
        return "Does the system need to return any information about %s?" % self.agg_col_tab_description(col_name, tab_name)

    def select_agg_question(self, col_name, tab_name, src_agg, bool_distinct=False):
        if src_agg == "none_agg":
            return "Does the system need to return the value of %s?" % self.agg_col_tab_description(col_name, tab_name)
        else:
            src_agg = src_agg.lower()
            return "Does the system need to return %s?" % self.agg_col_tab_description(
                col_name, tab_name, agg=src_agg, bool_distinct=bool_distinct)

    def where_col_question(self, col_name, tab_name):
        return "Does the system need to consider any specific conditions about %s?" %\
               self.agg_col_tab_description(col_name, tab_name)

    def andor_question(self, and_or, selected_cols_info): # deprecated
        if and_or == "and":
            return "Do the conditions about %s hold at the same time?" % selected_cols_info
        elif and_or == "or":
            return "Do the conditions about %s hold alternatively?" % selected_cols_info
        else:
            raise ValueError("Invalid and_or=%s!" % and_or)

    def where_op_question(self, agg_col_tab_name, op_name):
        value_descrip = "patterns" if op_name == "like" else "values"
        return "The system is enforcing the condition that in the results, %s must %s some specific %s. " % (
            agg_col_tab_name, self.where_op[op_name], value_descrip) + "Is the condition correct?"

    def root_terminal_question(self, col_name, tab_name, op_name, root_or_terminal,
                               bool_having=False, agg=None, group_by_col_info=None,
                               bool_distinct=False):
        root_term_description = self.root_terminal_description(col_name, tab_name, op_name, root_or_terminal,
                                                               bool_having=bool_having, agg=agg,
                                                               bool_distinct=bool_distinct)

        if bool_having:
            question = "The system will first group %s. " \
                       "Does it need to enforce the condition that in the results, %s?" % (
                       group_by_col_info, root_term_description)
        else:
            question = "The system is enforcing the condition that in the results, %s. Is this condition correct?" % (
                root_term_description)

        return question

    def root_terminal_description(self, col_name, tab_name, op_name, root_or_terminal,
                                  bool_having=False, agg=None, bool_distinct=False):
        agg_col_tab_name = self.agg_col_tab_description(col_name, tab_name, agg=agg, bool_having=bool_having,
                                                        bool_distinct=bool_distinct)

        if root_or_terminal == "terminal":
            if op_name in {"in", "not in"}:
                value_descrip = "a set of given literal values (e.g., number 5, string \"France\")"
            elif op_name == "between":
                value_descrip = "two given literal values (e.g., number 5, string \"France\")"
            else:
                value_descrip = "a given literal value (e.g., number 5, string \"France\")"
        else:
            assert root_or_terminal == "root"
            if op_name in {"in", "not in"}:
                value_descrip = "a set of values to be calculated"
            else:
                value_descrip = "a value to be calculated"

        description = "%s must %s %s" % (agg_col_tab_name, self.where_op[op_name], value_descrip)

        return description

    def where_val_question(self, col_name, tab_name, op_name, val_str):
        return "The system is enforcing the condition that in the results, %s must %s %s. " % (
            self.agg_col_tab_description(col_name, tab_name), self.where_op[op_name], val_str) + \
               "Is the condition correct?"

    def group_col_question(self, col_name, tab_name):
        assert tab_name is not None
        return "Does the system need to group %s?" % self.group_by_agg_col_tab_description(col_name, tab_name)

    def group_none_having_question(self, group_by_cols_info): # deprecated
        return "The system decides to group %s, but " % group_by_cols_info + bcolors.UNDERLINE + "without" + bcolors.ENDC + \
               " considering any other conditions. Is this correct?"

    def have_col_question(self, group_by_cols_info, col_name, tab_name):
        question = "The system will first group %s. " \
                   "Does it need to consider any specific conditions about %s?" % (
            group_by_cols_info, self.agg_col_tab_description(col_name, tab_name, bool_having=True))

        return question

    def have_agg_question(self, group_by_cols_info, col_name, tab_name, src_agg, bool_distinct=False):
        src_agg = src_agg.lower()
        if src_agg == "none_agg":
            question = "The system will first group %s. " \
                       "Does it need to consider any specific conditions about the value of %s?" % (
                           group_by_cols_info, self.agg_col_tab_description(col_name, tab_name, bool_having=True))

        else:
            agg = src_agg

            question = "The system will first group %s. " \
                       "Does it need to consider any specific conditions about %s?" % (
                           group_by_cols_info, self.agg_col_tab_description(col_name, tab_name, agg=agg, bool_having=True))

        return question

    def have_op_question(self, group_by_cols_info, col_name, tab_name, op_name, agg=None, bool_distinct=False):
        value_descrip = "patterns" if op_name == "like" else "values"
        question = "The system will first group %s. " \
                   "Does it need to enforce the condition that in the results, " \
                   "%s must % some specific %s?" % (
            group_by_cols_info, self.agg_col_tab_description(col_name, tab_name, agg=agg,
                                                             bool_having=True, bool_distinct=bool_distinct),
            self.where_op[op_name], value_descrip)

        return question

    def order_col_question(self, col_name, tab_name):
        return "Does the system need to order (in ascending or descending order) " \
               "the results based on %s?" % self.agg_col_tab_description(col_name, tab_name)

    def order_agg_question(self, col_name, tab_name, src_agg, bool_distinct=False):
        src_agg = src_agg.lower()
        if src_agg == "none_agg":
            return "Does the system need to order (in ascending or descending order) " \
                   "the results based on the value of %s?" % self.agg_col_tab_description(col_name, tab_name)
        else:
            agg = src_agg
            return "Does the system need to order (in ascending or descending order) " \
                   "the results based on %s?" % self.agg_col_tab_description(col_name, tab_name, agg=agg,
                                                                             bool_distinct=bool_distinct)

    def order_desc_asc_limit_question(self, col_name, tab_name, desc_asc_limit, agg=None):
        return "Assume that the system will sort the results based on %s. \n" \
               "\tIf this assumption is true, do the results need to be %s? \n" \
               "\tIf you think this assumption is incorrect, please select the 'No' option." % (
            self.agg_col_tab_description(col_name, tab_name, agg=agg), self.desc_asc_limit[desc_asc_limit])

    def order_desc_asc_question(self, col_name, tab_name, desc_asc, agg=None, bool_distinct=False):
        return "Assume that the system will sort the results based on %s. \n" \
               "\tIf this assumption is true, do the results need to be %s? \n" \
               "\tIf you think this assumption is incorrect, please select the 'No' option." % (
                   self.agg_col_tab_description(col_name, tab_name, agg=agg, bool_distinct=bool_distinct),
                   self.desc_asc_limit[(desc_asc, False)])

    def order_limit_question(self, col_name, tab_name, agg=None, bool_distinct=False):
        return "Assume that the system will sort the results based on %s (in ascending or descending order). \n" \
               "\tIf this assumption is true, do the results need to be limited to top N? \n" \
               "\tIf you think this assumption is incorrect, please select the 'No' option." % (
            self.agg_col_tab_description(col_name, tab_name, agg=agg, bool_distinct=bool_distinct))

    def iuen_question(self, iuen):
        if iuen == "except":
            return "Does the system need to return information satisfying some cases BUT NOT others?\n" \
                   "e.g., Find all airlines that have flights from airport 'CVO' BUT NOT from 'APG'."
        elif iuen == "union":
            return "Does the system need to return information satisfying either some cases OR others?\n" \
                   "e.g., What are the id and names of the countries which have more than 3 car makers OR " \
                   "produce the 'fiat' model?"
        elif iuen == "intersect":
            return "Does the system need to return information satisfying BOTH some cases AND the others AT THE " \
                   "SAME TIME?\ne.g., Which district has BOTH stores with less than 3000 products AND " \
                   "stores with more than 10000 products?"
        else:
            return "Does the system need to return information that meets one of the three situations: \n" \
                   "(1) satisfying some cases BUT NOT others, e.g., Find all airlines that have flights " \
                   "from airport 'CVO' BUT NOT from 'APG'.\n" \
                   "(2) satisfying either some cases OR others, e.g., What are the id and " \
                   "names of the countries which have more than 3 car makers OR produce the 'fiat' model?\n" \
                   "(3) satisfying BOTH some cases AND the others AT THE SAME TIME, e.g., Which district has BOTH " \
                   "stores with less than 3000 products AND stores with more than 10000 products?\n" \
                   "(Note: your situation is very likely to fall into NONE of the above - suggest to answer 'no')"

    def where_having_nested_question(self, col_name, tab_name, op_name, right_question, agg=None, bool_having=False,
                                     bool_distinct=False):
        revised_right_question = right_question[:-1] + " for this calculation?"
        return "Assume the system will enforce the condition that in the results, %s, " \
               "answer the following question to help the system to calculate " % self.root_terminal_description(
               col_name, tab_name, op_name, "root", agg=agg, bool_having=bool_having) +\
               bcolors.UNDERLINE + "the value(s)" + bcolors.ENDC + ": \n%s" % revised_right_question

    def question_generation(self, semantic_unit, tag_seq, pointer):
        """
        Generating NL questions.
        :param semantic_unit: the questioned semantic unit.
        :param tag_seq: the tag sequence produced by the parser.
        :param pointer: the pointer to tag_item in the tag_seq.
        :return: an NL question and cheat_sheet = {'yes'/'no': (bool_correct, bool_binary_choice_unit)}, where
                 bool_correct is True when the the user response ('yes' or 'no') means the decision is correct, and
                 bool_binary_choice_unit is True when there are only two choices for the decision (e.g., AND/OR); in
                 this case, the agent will switch to the alternative choice when the current one is regared wrong.
                 In general, cheat_sheet is used to simulate user feedback. For example, {'yes': (True, None), 'no': (False, 0)}
                 indicates that, if the user answers 'yes', she affirms the decision; if she answers 'no', she negates it.
        """
        assert tag_seq[pointer] == semantic_unit

        semantic_tag = semantic_unit[0]
        if semantic_tag == SELECT_COL:
            tab_col_item = semantic_unit[1]
            question = self.select_col_question(tab_col_item[1], tab_col_item[0])
            cheat_sheet = {'yes': (True, None), 'no': (False, 0)}

        elif semantic_tag == SELECT_AGG:
            col, agg = semantic_unit[1:3]
            question = self.select_agg_question(col[1], col[0], agg[0])
            cheat_sheet = {'yes': (True, None), 'no': (False, 0)}

        elif semantic_tag == SELECT_AGG_v2:
            col, agg, bool_distinct = semantic_unit[1:4]
            question = self.select_agg_question(col[1], col[0], agg[0], bool_distinct=bool_distinct)
            cheat_sheet = {'yes': (True, None), 'no': (False, 0)}

        elif semantic_tag == WHERE_COL:
            tab_col_item = semantic_unit[1]
            question = self.where_col_question(tab_col_item[1], tab_col_item[0])
            cheat_sheet = {'yes': (True, None), 'no': (False, 0)}

        elif semantic_tag == ANDOR:
            and_or, cols = semantic_unit[1:3]
            cols_info = [self.agg_col_tab_description(col[1], col[0]) for col in cols]
            question = self.andor_question(and_or, ", ".join(cols_info))
            cheat_sheet = {'yes': (True, None), 'no': (False, 1)}

        elif semantic_tag == WHERE_OP:
            ((col,), op) = semantic_unit[1:3]
            question = self.where_op_question(self.agg_col_tab_description(col[1], col[0]), op[0])
            cheat_sheet = {'yes': (True, None), 'no': (False, 0)}

        elif semantic_tag == WHERE_VAL:
            ((col,), op, val_item) = semantic_unit[1:4]
            question = self.where_val_question(col[1], col[0], op[0], val_item[-1])
            cheat_sheet = {'yes': (True, None), 'no': (False, 0)}

        elif semantic_tag == WHERE_ROOT_TERM:
            ((col,), op, root_term) = semantic_unit[1:4]
            question = self.root_terminal_question(col[1], col[0], op[0], root_term)
            cheat_sheet = {'yes': (True, None), 'no': (False, 1)}

        elif semantic_tag == GROUP_COL:
            tab_col_item = semantic_unit[1]
            question = self.group_col_question(tab_col_item[1], tab_col_item[0])
            cheat_sheet = {'yes': (True, None), 'no': (False, 0)}  # no->drop

        elif semantic_tag == GROUP_NHAV:
            groupBy_cols = []
            # idx = pointer - 2
            idx = helper_find_closest_bw(tag_seq, pointer - 1, tgt_id=GROUP_COL)
            while idx > 0:
                if tag_seq[idx][0] == GROUP_COL:
                    groupBy_cols.append(self.group_by_agg_col_tab_description(tag_seq[idx][1][1], tag_seq[idx][1][0]))
                else:
                    break
                idx -= 1
            question = self.group_none_having_question(", ".join(groupBy_cols))
            cheat_sheet = {'yes': (True, None), 'no': (False, 1)}

        elif semantic_tag == HAV_COL:
            tab_col_item = semantic_unit[1]

            closest_group_col_idx = helper_find_closest_bw(tag_seq, pointer - 1, tgt_id=GROUP_COL)
            groupBy_cols = []
            idx = closest_group_col_idx
            while idx > 0:
                if tag_seq[idx][0] == GROUP_COL:
                    groupBy_cols.append(self.group_by_agg_col_tab_description(tag_seq[idx][1][1], tag_seq[idx][1][0]))
                else:
                    break
                idx -= 1

            if len(groupBy_cols) > 1:
                group_by_col_info = ", ".join(groupBy_cols[:-1]) + " and " + groupBy_cols[-1]
            else:
                group_by_col_info = groupBy_cols[0]

            question = self.have_col_question(group_by_col_info, tab_col_item[1], tab_col_item[0])
            cheat_sheet = {'yes': (True, None), 'no':(False, 0)}

        elif semantic_tag == HAV_AGG:
            col, agg = semantic_unit[1:3]

            closest_group_col_idx = helper_find_closest_bw(tag_seq, pointer - 1, tgt_id=GROUP_COL)
            groupBy_cols = []
            idx = closest_group_col_idx
            while idx > 0:
                if tag_seq[idx][0] == GROUP_COL:
                    groupBy_cols.append(self.group_by_agg_col_tab_description(tag_seq[idx][1][1], tag_seq[idx][1][0]))
                else:
                    break
                idx -= 1

            if len(groupBy_cols) > 1:
                group_by_col_info = ", ".join(groupBy_cols[:-1]) + " and " + groupBy_cols[-1]
            else:
                group_by_col_info = groupBy_cols[0]

            question = self.have_agg_question(group_by_col_info, col[1], col[0], agg[0])

            cheat_sheet = {'yes': (True, None), 'no': (False, 0)}

        elif semantic_tag == HAV_AGG_v2:
            col, agg, bool_distinct = semantic_unit[1:4]
            closest_group_col_idx = helper_find_closest_bw(tag_seq, pointer - 1, tgt_id=GROUP_COL)
            groupBy_cols = []
            idx = closest_group_col_idx
            while idx > 0:
                if tag_seq[idx][0] == GROUP_COL:
                    groupBy_cols.append(self.group_by_agg_col_tab_description(tag_seq[idx][1][1], tag_seq[idx][1][0]))
                else:
                    break
                idx -= 1

            if len(groupBy_cols) > 1:
                group_by_col_info = ", ".join(groupBy_cols[:-1]) + " and " + groupBy_cols[-1]
            else:
                group_by_col_info = groupBy_cols[0]

            question = self.have_agg_question(group_by_col_info, col[1], col[0], agg[0],
                                              bool_distinct=bool_distinct)

            cheat_sheet = {'yes': (True, None), 'no': (False, 0)}

        elif semantic_tag == HAV_OP:
            (col, agg), op = semantic_unit[1:3]

            closest_group_col_idx = helper_find_closest_bw(tag_seq, pointer - 1, tgt_id=GROUP_COL)
            groupBy_cols = []
            idx = closest_group_col_idx
            while idx > 0:
                if tag_seq[idx][0] == GROUP_COL:
                    groupBy_cols.append(self.group_by_agg_col_tab_description(tag_seq[idx][1][1], tag_seq[idx][1][0]))
                else:
                    break
                idx -= 1

            if len(groupBy_cols) > 1:
                group_by_col_info = ", ".join(groupBy_cols[:-1]) + " and " + groupBy_cols[-1]
            else:
                group_by_col_info = groupBy_cols[0]

            question = self.have_op_question(group_by_col_info, col[1], col[0], op[0],
                                             agg=None if agg[0] == "none_agg" else agg[0])
            cheat_sheet = {'yes': (True, None), 'no': (False, 0)}

        elif semantic_tag == HAV_OP_v2:
            (col, agg, bool_distinct), op = semantic_unit[1:3]

            closest_group_col_idx = helper_find_closest_bw(tag_seq, pointer - 1, tgt_id=GROUP_COL)
            groupBy_cols = []
            idx = closest_group_col_idx
            while idx > 0:
                if tag_seq[idx][0] == GROUP_COL:
                    groupBy_cols.append(self.group_by_agg_col_tab_description(tag_seq[idx][1][1], tag_seq[idx][1][0]))
                else:
                    break
                idx -= 1

            if len(groupBy_cols) > 1:
                group_by_col_info = ", ".join(groupBy_cols[:-1]) + " and " + groupBy_cols[-1]
            else:
                group_by_col_info = groupBy_cols[0]

            question = self.have_op_question(group_by_col_info, col[1], col[0], op[0],
                                             agg=None if agg[0] == "none_agg" else agg[0],
                                             bool_distinct=bool_distinct)
            cheat_sheet = {'yes': (True, None), 'no': (False, 0)}

        elif semantic_tag == HAV_ROOT_TERM:
            (col, agg), op, root_term = semantic_unit[1:4]

            closest_group_col_idx = helper_find_closest_bw(tag_seq, pointer - 1, tgt_id=GROUP_COL)
            groupBy_cols = []
            idx = closest_group_col_idx
            while idx > 0:
                if tag_seq[idx][0] == GROUP_COL:
                    groupBy_cols.append(self.group_by_agg_col_tab_description(tag_seq[idx][1][1], tag_seq[idx][1][0]))
                else:
                    break
                idx -= 1

            if len(groupBy_cols) > 1:
                group_by_col_info = ", ".join(groupBy_cols[:-1]) + " and " + groupBy_cols[-1]
            else:
                group_by_col_info = groupBy_cols[0]

            question = self.root_terminal_question(col[1], col[0], op[0], root_term, bool_having=True,
                                                   agg=None if agg[0] == "none_agg" else agg[0],
                                                   group_by_col_info=group_by_col_info)
            cheat_sheet = {'yes': (True, None), 'no': (False, 1)}

        elif semantic_tag == HAV_ROOT_TERM_v2:
            (col, agg, bool_distinct), op, root_term = semantic_unit[1:4]

            closest_group_col_idx = helper_find_closest_bw(tag_seq, pointer - 1, tgt_id=GROUP_COL)
            groupBy_cols = []
            idx = closest_group_col_idx
            while idx > 0:
                if tag_seq[idx][0] == GROUP_COL:
                    groupBy_cols.append(self.group_by_agg_col_tab_description(tag_seq[idx][1][1], tag_seq[idx][1][0]))
                else:
                    break
                idx -= 1

            if len(groupBy_cols) > 1:
                group_by_col_info = ", ".join(groupBy_cols[:-1]) + " and " + groupBy_cols[-1]
            else:
                group_by_col_info = groupBy_cols[0]

            question = self.root_terminal_question(col[1], col[0], op[0], root_term, bool_having=True,
                                                   agg=None if agg[0] == "none_agg" else agg[0],
                                                   group_by_col_info=group_by_col_info,
                                                   bool_distinct=bool_distinct)
            cheat_sheet = {'yes': (True, None), 'no': (False, 1)}

        elif semantic_tag == ORDER_COL:
            tab_col_item = semantic_unit[1]
            question = self.order_col_question(tab_col_item[1], tab_col_item[0])
            cheat_sheet = {'yes': (True, None), 'no': (False, 0)}

        elif semantic_tag == ORDER_AGG:
            col, agg = semantic_unit[1:3]
            question = self.order_agg_question(col[1], col[0], agg[0])
            cheat_sheet = {'yes': (True, None), 'no': (False, 0)}

        elif semantic_tag == ORDER_AGG_v2:
            col, agg, bool_distinct = semantic_unit[1:4]
            question = self.order_agg_question(col[1], col[0], agg[0], bool_distinct=bool_distinct)
            cheat_sheet = {'yes': (True, None), 'no': (False, 0)}

        elif semantic_tag == ORDER_DESC_ASC_LIMIT:
            (col, agg), desc_asc_limit = semantic_unit[1:3]
            question = self.order_desc_asc_limit_question(col[1], col[0], desc_asc_limit,
                                                          agg=None if agg[0] == "none_agg" else agg[0])
            cheat_sheet = {'yes': (True, None), 'no': (False, 0)}

        elif semantic_tag == ORDER_DESC_ASC:
            (col, agg, bool_distinct), desc_asc = semantic_unit[1:3]
            question = self.order_desc_asc_question(col[1], col[0], desc_asc,
                                                    agg=None if agg[0] == "none_agg" else agg[0],
                                                    bool_distinct=bool_distinct)
            cheat_sheet = {'yes': (True, None), 'no': (False, 1)}

        elif semantic_tag == ORDER_LIMIT:
            (col, agg, bool_distinct) = semantic_unit[1]
            question = self.order_limit_question(col[1], col[0],
                                                 agg=None if agg[0] == "none_agg" else agg[0],
                                                 bool_distinct=bool_distinct)
            cheat_sheet = {'yes': (True, None), 'no': (False, 1)}

        elif semantic_tag == IUEN:
            iuen = semantic_unit[1]
            question = self.iuen_question(iuen[0])
            if iuen[0] == "none":
                cheat_sheet = {'no': (True, None), 'yes': (False, 0)}
            else:
                cheat_sheet = {'yes': (True, None), 'no': (False, 0)}

        elif semantic_tag == IUEN_v2:
            iuen = semantic_unit[1]
            question = self.iuen_question(iuen[0])
            cheat_sheet = {'yes': (True, None), 'no': (False, 0)}

        else:
            print("WARNING: Unknown semantic_tag %s" % semantic_tag)
            question = ""
            cheat_sheet = None

        # check nested WHERE/HAVING condition or IUEN != none
        closest_root_idx = helper_find_closest_bw(tag_seq, pointer - 1, tgt_name="root")
        if closest_root_idx == -1: # not found, not nested
            return question, cheat_sheet
        else:
            root_tag = tag_seq[closest_root_idx][0]
            if root_tag == OUTSIDE: # IUEN != none
                return question, cheat_sheet
            else:
                closest_end_nested_idx = helper_find_closest_bw(tag_seq, pointer - 1, tgt_name=END_NESTED)
                if closest_end_nested_idx != -1 and closest_end_nested_idx > closest_root_idx:
                    # outside the nested WHERE/HAVING condition
                    return question, cheat_sheet

                # nested WHERE/HAVING condition
                if root_tag == WHERE_ROOT_TERM:
                    (col,), op = tag_seq[closest_root_idx][1:3]
                    question = self.where_having_nested_question(col[1], col[0], op[0], question)
                elif root_tag == HAV_ROOT_TERM:
                    (col, agg), op = tag_seq[closest_root_idx][1:3]
                    question = self.where_having_nested_question(col[1], col[0], op[0], question,
                                                                 agg=agg[0] if agg[0] != 'none_agg' else None,
                                                                 bool_having=True)
                elif root_tag == HAV_ROOT_TERM_v2:
                    (col, agg, bool_distinct), op = tag_seq[closest_root_idx][1:3]
                    question = self.where_having_nested_question(col[1], col[0], op[0], question,
                                                                 agg=agg[0] if agg[0] != 'none_agg' else None,
                                                                 bool_having=True, bool_distinct=bool_distinct)
                else:
                    raise ValueError("Unexpected nested condition: tag_seq: {}\nPointer: {}, closest root: {}.".format(
                        tag_seq, pointer, tag_seq[closest_root_idx]
                    ))
                return question, cheat_sheet

    def option_generation(self, cand_semantic_units, old_tag_seq, pointer):
        """
        Options generation.
        :param cand_semantic_units: a list of semantic units being the options.
        :param old_tag_seq: the original tag_seq, a sequence of semantic units.
        :param pointer: the pointer to the questioned semantic unit in old_tag_seq.
        :return: NL question, cheat_sheet = {choice idx: corresponding decision idx} (which will be used to simulate
                 user selections), the index for "none of the above".
        """
        semantic_tag = old_tag_seq[pointer][0]
        cheat_sheet = {}
        prefix, option_text = "", ""

        if semantic_tag == SELECT_COL:
            prefix = "Please select any options from the following list that the system needs to return information about:\n"
            for idx, su in enumerate(cand_semantic_units):
                tab_col_item = su[1]
                option_text += "(%d) %s;\n" % (idx+1, self.agg_col_tab_description(tab_col_item[1], tab_col_item[0]))
                cheat_sheet[idx+1] = tab_col_item[-1] # col id

        elif semantic_tag == SELECT_AGG:
            prefix = "Please select any options from the following list that the system needs to return:\n"
            for idx, su in enumerate(cand_semantic_units):
                col, (agg, agg_idx) = su[1:3]
                if agg == "none_agg":
                    option_text += "(%d) the value of %s;\n" % (
                        idx + 1, self.agg_col_tab_description(col[1], col[0]))
                else:
                    option_text += "(%d) %s;\n" % (idx+1, self.agg_col_tab_description(col[1], col[0], agg=agg.lower()))
                cheat_sheet[idx+1] = (col[-1], agg_idx)

        elif semantic_tag == SELECT_AGG_v2:
            prefix = "Please select any options from the following list that the system needs to return:\n"
            for idx, su in enumerate(cand_semantic_units):
                col, (agg, agg_idx), bool_distinct = su[1:4]
                if agg == "none_agg":
                    option_text += "(%d) the value of %s;\n" % (
                        idx + 1, self.agg_col_tab_description(col[1], col[0]))
                else:
                    option_text += "(%d) %s;\n" % (
                        idx+1, self.agg_col_tab_description(col[1], col[0], agg=agg.lower(), bool_distinct=bool_distinct))
                cheat_sheet[idx+1] = (col[-1], agg_idx, bool_distinct)

        elif semantic_tag == WHERE_COL:
            prefix = "Please select any options from the following list that the system needs to consider conditions about:\n"
            for idx, su in enumerate(cand_semantic_units):
                tab_col_item = su[1]
                option_text += "(%d) %s;\n" % (idx + 1, self.agg_col_tab_description(tab_col_item[1], tab_col_item[0]))
                cheat_sheet[idx + 1] = tab_col_item[-1] # col id

        elif semantic_tag == WHERE_OP:
            prefix = "Please select any options from the following list that the system needs to enforce as conditions:\n"
            for idx, su in enumerate(cand_semantic_units):
                ((col,), (op_name, op_idx)) = su[1:3]
                condition_text = "%s %s a value" % (self.agg_col_tab_description(col[1], col[0]),
                                                    self.where_op.get(op_name, op_name))
                option_text += "(%d) %s;\n" % (idx+1, condition_text)
                cheat_sheet[idx+1] = (col[-1], op_idx) # (col id, op id)

        elif semantic_tag == WHERE_VAL:
            prefix = "Please select any options from the following list that the system needs to enforce as conditions:\n"
            for idx, su in enumerate(cand_semantic_units):
                ((col,), (op_name, op_idx), val_item) = su[1:4]
                condition_text = "%s %s \"%s\"" % (self.agg_col_tab_description(col[1], col[0]),
                                                    self.where_op.get(op_name, op_name), val_item[-1])
                option_text += "(%d) %s;\n" % (idx+1, condition_text)
                cheat_sheet[idx+1] = (col[-1], op_idx, val_item[-1]) # (col id, op id, val name)

        elif semantic_tag == GROUP_COL:
            prefix = "Please select any options from the following list:\n"
            for idx, su in enumerate(cand_semantic_units):
                tab_col_item = su[1]
                group_col_text = "The system needs to group %s" % (
                    self.group_by_agg_col_tab_description(tab_col_item[1], tab_col_item[0]))
                option_text += "(%d) %s;\n" % (idx+1, group_col_text)
                cheat_sheet[idx+1] = tab_col_item[-1] # col id

        elif semantic_tag == HAV_COL:
            prefix = "(Following the last question) Please select any options from the following list that " \
                     "the system needs to consider conditions about:\n"
            for idx, su in enumerate(cand_semantic_units):
                tab_col_item = su[1]
                option_text += "(%d) %s;\n" % (idx+1, self.agg_col_tab_description(
                    tab_col_item[1], tab_col_item[0], bool_having=True))
                cheat_sheet[idx + 1] = tab_col_item[-1] # col id

        elif semantic_tag == HAV_AGG:
            prefix = "(Following the last question) Please select ONE option from the following list that the system " \
                     "needs to consider conditions about:\n"
            for idx, su in enumerate(cand_semantic_units):
                col, agg = su[1:3]
                if agg[0] == "none_agg":
                    option_text += "(%d) the value of %s;\n" % (
                        idx+1, self.agg_col_tab_description(col[1], col[0], bool_having=True))
                else:
                    option_text += "(%d) %s;\n" % (idx+1, self.agg_col_tab_description(
                        col[1], col[0], agg=agg[0].lower(), bool_having=True))
                cheat_sheet[idx + 1] = (col[-1], agg[1]) # (col id, agg id)

        elif semantic_tag == HAV_AGG_v2:
            prefix = "(Following the last question) Please select any options from the following list that the system " \
                     "needs to consider conditions about:\n"
            for idx, su in enumerate(cand_semantic_units):
                col, agg, bool_distinct = su[1:4]
                if agg[0] == "none_agg":
                    option_text += "(%d) the value of %s;\n" % (
                        idx+1, self.agg_col_tab_description(col[1], col[0], bool_having=True))
                else:
                    option_text += "(%d) %s;\n" % (idx+1, self.agg_col_tab_description(
                        col[1], col[0], agg=agg[0].lower(), bool_having=True, bool_distinct=bool_distinct))
                cheat_sheet[idx + 1] = (col[-1], agg[1], bool_distinct)

        elif semantic_tag == HAV_OP:
            prefix = "(Following the last question) Please select any options from the following list that " \
                     "the system needs to enforce as conditions:\n"
            for idx, su in enumerate(cand_semantic_units):
                (col, agg), op = su[1:3]
                condition_text = "%s %s a value" % (self.agg_col_tab_description(
                    col[1], col[0], agg=None if agg[0] == "none_agg" else agg[0], bool_having=True),
                                                    self.where_op.get(op[0], op[0]))
                option_text += "(%d) %s;\n" % (idx+1, condition_text)
                cheat_sheet[idx+1] = ((col[-1], agg[1]), op[1])

        elif semantic_tag == HAV_OP_v2:
            prefix = "(Following the last question) Please select any options from the following list that " \
                     "the system needs to enforce as conditions:\n"
            for idx, su in enumerate(cand_semantic_units):
                (col, agg, bool_distinct), op = su[1:3]
                condition_text = "%s %s a value" % (self.agg_col_tab_description(
                    col[1], col[0], agg=None if agg[0] == "none_agg" else agg[0], bool_having=True,
                    bool_distinct=bool_distinct), self.where_op.get(op[0], op[0]))
                option_text += "(%d) %s;\n" % (idx+1, condition_text)
                cheat_sheet[idx+1] = ((col[-1], agg[1], bool_distinct), op[1])

        elif semantic_tag == ORDER_COL:
            prefix = "Please select any options from the following list, based on which (and their calculations)" \
                     " the system sorts the results:\n"
            for idx, su in enumerate(cand_semantic_units):
                tab_col_item, = su[1]
                option_text += "(%d) %s;\n" % (idx + 1, self.agg_col_tab_description(
                    tab_col_item[1], tab_col_item[0]))
                cheat_sheet[idx + 1] = tab_col_item[-1] # col id

        elif semantic_tag == ORDER_AGG:
            prefix = "Please select ONE option from the following list, based on which the system " \
                     "sort the results:\n"
            for idx, su in enumerate(cand_semantic_units):
                col, agg = su[1:3]
                if agg[0] == "none_agg":
                    option_text += "(%d) the value of %s;\n" % (
                        idx+1, self.agg_col_tab_description(col[1], col[0]))
                else:
                    option_text += "(%d) %s;\n" % (idx + 1, self.agg_col_tab_description(col[1], col[0],
                                                                                         agg=agg[0].lower()))
                cheat_sheet[idx + 1] = (col[-1], agg[1]) # (col id, agg id)

        elif semantic_tag == ORDER_AGG_v2:
            prefix = "Please select any options from the following list, based on which the system " \
                     "sort the results:\n"
            for idx, su in enumerate(cand_semantic_units):
                col, agg, bool_distinct = su[1:4]
                if agg[0] == "none_agg":
                    option_text += "(%d) the value of %s;\n" % (
                        idx+1, self.agg_col_tab_description(col[1], col[0]))
                else:
                    option_text += "(%d) %s;\n" % (idx + 1, self.agg_col_tab_description(
                        col[1], col[0], agg=agg[0].lower(), bool_distinct=bool_distinct))
                cheat_sheet[idx + 1] = (col[-1], agg[1], bool_distinct)

        elif semantic_tag == ORDER_DESC_ASC_LIMIT:
            prefix = "(Following the last question) Please select ONE option from the following list:\n"
            for idx, su in enumerate(cand_semantic_units):
                (col, agg), desc_asc_limit = su[1:3]
                option_text += "(%d) %s;\n" % (idx+1, "The system should sort results " +
                                               self.desc_asc_limit[desc_asc_limit])
                cheat_sheet[idx + 1] = ((col[-1], agg[1]), desc_asc_limit)

        elif semantic_tag == IUEN:
            prefix = "Please select ONE option from the following list:\n"

            for idx, su in enumerate(cand_semantic_units):
                if su[1][0] == 'none':
                    iuen_text = "The system does NOT need to return information that satisfies a complicated situation " \
                                "as other options indicate"
                elif su[1][0] == 'except':
                    iuen_text = "The system needs to return information that satisfies some cases BUT NOT others, " \
                                "e.g., Find all airlines that have flights from airport 'CVO' BUT NOT from 'APG'"
                elif su[1][0] == 'union':
                    iuen_text = "The system needs to return information that satisfies either some cases OR others, " \
                                "e.g., What are the id and names of the countries which have more than 3 car makers " \
                                "OR produce the 'fiat' model?"
                else:
                    assert su[1][0] == 'intersect'
                    iuen_text = "The system needs to return information that satisfies BOTH some cases AND the others" \
                                " AT THE SAME TIME, e.g., Which district has BOTH stores with less than 3000 products " \
                                "AND stores with more than 10000 products?"
                option_text += "(%d) %s;\n" % (idx+1, iuen_text)
                cheat_sheet[idx + 1] = su[1][1] # iuen id

        elif semantic_tag == IUEN_v2:
            prefix = "Please select ONE option from the following list:\n"

            for idx, su in enumerate(cand_semantic_units):
                if su[1][0] == 'except':
                    iuen_text = "The system needs to return information that satisfies some cases BUT NOT others, " \
                                "e.g., Find all airlines that have flights from airport 'CVO' BUT NOT from 'APG'"
                elif su[1][0] == 'union':
                    iuen_text = "The system needs to return information that satisfies either some cases OR others, " \
                                "e.g., What are the id and names of the countries which have more than 3 car makers " \
                                "OR produce the 'fiat' model?"
                else:
                    assert su[1][0] == 'intersect'
                    iuen_text = "The system needs to return information that satisfies BOTH some cases AND the others" \
                                " AT THE SAME TIME, e.g., Which district has BOTH stores with less than 3000 products " \
                                "AND stores with more than 10000 products?"
                option_text += "(%d) %s;\n" % (idx + 1, iuen_text)
                cheat_sheet[idx + 1] = su[1][1]  # iuen id

        else:
            print("WARNING: Unknown semantic_tag %s" % semantic_tag)
            return "", cheat_sheet, -1

        if semantic_tag != IUEN:
            option_text += "(%d) None of the above options." % (len(cand_semantic_units) + 1)
            question = prefix + option_text
            return question, cheat_sheet, len(cheat_sheet) + 1
        else:
            question = prefix + option_text.strip()
            return question, cheat_sheet, -1
