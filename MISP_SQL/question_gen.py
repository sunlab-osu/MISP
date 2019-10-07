# question generator
from .utils import *


class QuestionGenerator:
    """
    This is the class for question generation.
    """
    def __init__(self):
        # the seed lexicon
        self.agg = {"avg": "average value in", "count": "number of", "sum": "sum of values in",
                    "min": "minimum value in", "max": "maximum value in"}
        self.where_op = {"like": "follows a pattern like", "not in": "is NOT IN", ">": "is greater than",
                         "<": "is less than", "=": "equals to", ">=": "is greater than or equivalent to",
                         "<=": "is less than or equivalent to", "!=": "does not equal to",
                         "in": "is IN", "between": "is between"}
        self.desc_asc_limit = {("desc", False): "in descending order", ("asc", False): "in ascending order",
                               ("desc", True): "in descending order and limited to top N",
                               ("asc", True): "in ascending order and limited to top N"}

    def agg_col_tab_description(self, col_name, tab_name, agg=None, bool_having=False):
        if agg is not None:
            agg = agg.lower()

        if col_name == "*":
            if bool_having:
                _col_name = "grouped items"
            else:
                _col_name = "all items"
            if agg is None:
                return bcolors.BLUE + "all table attributes or (possibly the number of) %s" % _col_name + \
                       bcolors.ENDC + " in the table " + bcolors.YELLOW + "\"" + tab_name + "\"" + bcolors.ENDC
            else:
                return self.agg.get(agg, agg) + " " + bcolors.BLUE + "%s" % _col_name + bcolors.ENDC + \
                       " in the table " + bcolors.YELLOW + "\"" + tab_name + "\"" + bcolors.ENDC
        else:
            if tab_name is None:
                col_descrip = bcolors.BLUE + "\"" + col_name + "\"" + bcolors.ENDC
            else:
                col_descrip = bcolors.BLUE + "\"" + col_name + "\"" + bcolors.ENDC + " in the table " +\
                              bcolors.YELLOW + "\"" + tab_name + "\"" + bcolors.ENDC
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
                return prefix + bcolors.UNDERLINE + "a given literal value (e.g., number 5, string \"France\")" +\
                       bcolors.ENDC
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
            return "Should the system return the value of %s " % self.agg_col_tab_description(col_name, tab_name) + \
                   bcolors.UNDERLINE + "without" + bcolors.ENDC + \
                   " doing any mathematical calculations (e.g., maximum, minimum, sum, average, count) on it?"
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

    def root_terminal_question(self, col_name, tab_name, where_op, root_or_terminal,
                               bool_having=False, agg=None, group_cols_info=None):
        agg_col_tab = self.agg_col_tab_description(col_name, tab_name, agg=agg, bool_having=bool_having)
        where_op_descrip = self.where_op.get(where_op, where_op)
        if bool_having:
            question = "Given that the system groups %s before doing any mathematical calculations " \
                       "(e.g., maximum, minimum, sum, average, count), " \
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
        return question

    def group_col_question(self, col_name, tab_name):
        assert tab_name is not None
        return "Does the system need to group %s" % self.group_by_agg_col_tab_description(col_name, tab_name) +\
               " before doing any mathematical calculations (e.g., maximum, minimum, sum, average, count)?"

    def group_none_having_question(self, group_by_cols_info):
        return "The system decides to group %s before doing any mathematical calculations (e.g., maximum, minimum," \
               " sum, average, count), but " % group_by_cols_info + bcolors.UNDERLINE + "without" + bcolors.ENDC + \
               " considering any other conditions. Is this correct?"

    def have_col_question(self, group_by_cols_info, col_name, tab_name):
        return "Given that the system groups %s before doing any mathematical calculations " \
               "(e.g., maximum, minimum, sum, average, count), " \
               "does the system need to consider any conditions about %s" % (
               group_by_cols_info, self.agg_col_tab_description(col_name, tab_name, bool_having=True))

    def have_agg_question(self, group_by_cols_info, col_name, tab_name, src_agg):
        src_agg = src_agg.lower()
        if src_agg == "none_agg":
            return "Given that the system groups %s before doing any mathematical calculations " \
                   "(e.g., maximum, minimum, sum, average, count), " % group_by_cols_info + \
                   "should the system return a value of %s " % self.agg_col_tab_description(
                   col_name, tab_name, bool_having=True) + \
                   bcolors.UNDERLINE + "without" + bcolors.ENDC + " doing any mathematical calculations on it?"
        else:
            return "Given that the system groups %s before doing any mathematical calculations " \
                   "(e.g., maximum, minimum, sum, average, count), " \
                   "does the system need to consider any conditions about %s?" % (
                   group_by_cols_info, self.agg_col_tab_description(col_name, tab_name, agg=src_agg, bool_having=True))

    def have_op_question(self, group_by_cols_info, col_name, tab_name, op, agg=None):
        return "The system groups %s before doing any mathematical calculations " \
               "(e.g., maximum, minimum, sum, average, count), " \
               "then considers the following condition: %s %s a value. Is the condition correct?" % (
               group_by_cols_info, self.agg_col_tab_description(col_name, tab_name, agg=agg, bool_having=True),
               self.where_op.get(op, op))

    def order_col_question(self, col_name, tab_name):
        return "Does the system need to order results based on %s?" % self.agg_col_tab_description(
               col_name, tab_name)

    def order_agg_question(self, col_name, tab_name, src_agg):
        src_agg = src_agg.lower()

        if src_agg == "none_agg":
            return "Should the system order results based on the value of %s " % self.agg_col_tab_description(
                   col_name, tab_name) + bcolors.UNDERLINE + "without" + bcolors.ENDC + \
                   " doing any mathematical calculations (e.g., maximum, minimum, sum, average, count) on it?"
        else:
            return "Does the system need to order results based on %s?" % self.agg_col_tab_description(
                   col_name, tab_name, agg=src_agg)

    def order_desc_asc_limit_question(self, col_name, tab_name, desc_asc_limit, agg=None):
        return "Given that the system orders the results based on %s, does it need to be %s?" % (
            self.agg_col_tab_description(col_name, tab_name, agg=agg), self.desc_asc_limit[desc_asc_limit])

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

    def where_having_nested_question(self, col_name, tab_name, where_op, right_question, agg=None, bool_having=False):
        revised_right_question = right_question[:-1] + " for this calculation?"
        return "Given that %s, answer the following question to help the system to calculate " % self.where_cond_descrip(
               col_name, tab_name, where_op, "root", agg=agg, bool_having=bool_having) +\
               bcolors.UNDERLINE + "the value(s)" + bcolors.ENDC + ":\n%s" % revised_right_question

    def question_generation(self, tag_item, tag_seq, pointer):
        """
        Generating NL questions.
        :param tag_item: the questioned semantic unit.
        :param tag_seq: the tag sequence produced by the parser.
        :param pointer: the pointer to tag_item in the tag_seq.
        :return: an NL question and answer_sheet = {'yes'/'no': (True/False, type of unit)}.
                We define three types: (1) type 0: for units like WHERE_COL, which have a sequential cardinality
                (i.e., #cols can be > 1); (2) type 1: for units like GROUP_NHAV, which have only binary choices
                (1->without a HAVING clause, 0->with a HAVING clause); (3) type 2: for units like WHERE_VAL, which have
                a singular cardinality (i.e., for a given WHERE condition, there chould be exactly one value).
                The answer_sheet is used to simulate user feedback. for example, {'yes': (True, None), 'no': (False, 0)}
                indicates that, if the user answers 'yes', she affirms the unit; if she answers 'no', she negates it.
        """
        assert tag_seq[pointer] == tag_item[0]

        template_id = tag_item[0][0]
        if template_id == SELECT_COL:
            tab_col_item, = tag_item
            question = self.select_col_question(tab_col_item[1][1], tab_col_item[1][0])
            answer_sheet = {'yes': (True, None), 'no': (False, 0)} #no->drop

        elif template_id == SELECT_AGG:
            col, agg = tag_item[0][1:3]
            question = self.select_agg_question(col[1], col[0], agg[0])
            if agg[0] == "none_agg":
                answer_sheet = {'yes': (True, None), 'no': (False, 0)}
            else:
                answer_sheet = {'yes': (True, None), 'no': (False, 0)}

        elif template_id == WHERE_COL:
            tab_col_item, = tag_item
            question = self.where_col_question(tab_col_item[1][1], tab_col_item[1][0])
            answer_sheet = {'yes': (True, None), 'no': (False, 0)}

        elif template_id == ANDOR:
            and_or, cols = tag_item[0][1:3]
            cols_info = [self.agg_col_tab_description(col[1], col[0]) for col in cols]
            question = self.andor_question(and_or, ", ".join(cols_info))
            answer_sheet = {'yes': (True, None), 'no': (False, 1)} #flip

        elif template_id == WHERE_OP:
            ((col,), op) = tag_item[0][1:3]
            question = self.where_op_question(self.agg_col_tab_description(col[1], col[0]), op[0])
            answer_sheet = {'yes': (True, None), 'no': (False, 0)}

        elif template_id == WHERE_VAL:
            ((col,), op, val_item) = tag_item[0][1:4]
            question = self.where_val_question(col[1], col[0], op[0], val_item[-1])
            answer_sheet = {'yes': (True, None), 'no': (False, 2)}

        elif template_id == WHERE_ROOT_TERM:
            ((col,), op, root_term) = tag_item[0][1:4]
            question = self.root_terminal_question(col[1], col[0], op[0], root_term)
            answer_sheet = {'yes': (True, None), 'no': (False, 1)} #no->flip

        elif template_id == GROUP_COL:
            tab_col_item, = tag_item
            question = self.group_col_question(tab_col_item[1][1], tab_col_item[1][0])
            answer_sheet = {'yes': (True, None), 'no': (False, 0)}  # no->drop

        elif template_id == GROUP_NHAV:
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
            answer_sheet = {'yes': (True, None), 'no': (False, 1)}

        elif template_id == HAV_COL:
            tab_col_item, = tag_item

            closest_group_col_idx = helper_find_closest_bw(tag_seq, pointer - 1, tgt_id=GROUP_COL)
            groupBy_cols = []
            idx = closest_group_col_idx
            while idx > 0:
                if tag_seq[idx][0] == GROUP_COL:
                    groupBy_cols.append(self.group_by_agg_col_tab_description(tag_seq[idx][1][1], tag_seq[idx][1][0]))
                else:
                    break
                idx -= 1
            question = self.have_col_question(", ".join(groupBy_cols), tab_col_item[1][1], tab_col_item[1][0])
            answer_sheet = {'yes': (True, None), 'no':(False, 0)}

        elif template_id == HAV_AGG:
            col, agg = tag_item[0][1:3]

            closest_group_col_idx = helper_find_closest_bw(tag_seq, pointer - 1, tgt_id=GROUP_COL)
            groupBy_cols = []
            idx = closest_group_col_idx
            while idx > 0:
                if tag_seq[idx][0] == GROUP_COL:
                    groupBy_cols.append(self.group_by_agg_col_tab_description(tag_seq[idx][1][1], tag_seq[idx][1][0]))
                else:
                    break
                idx -= 1

            question = self.have_agg_question(", ".join(groupBy_cols), col[1], col[0], agg[0])

            if agg[0] == "none_agg":
                answer_sheet = {'yes': (True, None), 'no': (False, 2)}
            else:
                answer_sheet = {'yes': (True, None), 'no': (False, 2)}

        elif template_id == HAV_OP:
            (col, agg), op = tag_item[0][1:3]

            closest_group_col_idx = helper_find_closest_bw(tag_seq, pointer - 1, tgt_id=GROUP_COL)
            groupBy_cols = []
            idx = closest_group_col_idx
            while idx > 0:
                if tag_seq[idx][0] == GROUP_COL:
                    groupBy_cols.append(self.group_by_agg_col_tab_description(tag_seq[idx][1][1], tag_seq[idx][1][0]))
                else:
                    break
                idx -= 1

            question = self.have_op_question(", ".join(groupBy_cols), col[1], col[0], op[0],
                                             agg=None if agg[0] == "none_agg" else agg[0])
            answer_sheet = {'yes': (True, None), 'no': (False, 0)}

        elif template_id == HAV_ROOT_TERM:
            (col, agg), op, root_term = tag_item[0][1:4]

            closest_group_col_idx = helper_find_closest_bw(tag_seq, pointer - 1, tgt_id=GROUP_COL)
            groupBy_cols = []
            idx = closest_group_col_idx
            while idx > 0:
                if tag_seq[idx][0] == GROUP_COL:
                    groupBy_cols.append(self.group_by_agg_col_tab_description(tag_seq[idx][1][1], tag_seq[idx][1][0]))
                else:
                    break
                idx -= 1

            question = self.root_terminal_question(col[1], col[0], op[0], root_term, bool_having=True,
                                                   agg=None if agg[0] == "none_agg" else agg[0],
                                                   group_cols_info=", ".join(groupBy_cols))
            answer_sheet = {'yes': (True, None), 'no': (False, 1)}

        elif template_id == ORDER_COL:
            tab_col_item, = tag_item
            question = self.order_col_question(tab_col_item[1][1], tab_col_item[1][0])
            answer_sheet = {'yes': (True, None), 'no': (False, 0)}

        elif template_id == ORDER_AGG:
            col, agg = tag_item[0][1:3]
            question = self.order_agg_question(col[1], col[0], agg[0])
            if agg == "none_agg":
                answer_sheet = {'yes': (True, None), 'no': (False, 2)}
            else:
                answer_sheet = {'yes': (True, None), 'no': (False, 2)}

        elif template_id == ORDER_DESC_ASC_LIMIT:
            (col, agg), desc_asc_limit = tag_item[0][1:3]
            question = self.order_desc_asc_limit_question(col[1], col[0], desc_asc_limit,
                                                          agg=None if agg[0] == "none_agg" else agg[0])
            answer_sheet = {'yes': (True, None), 'no': (False, 2)}

        elif template_id == IUEN:
            iuen = tag_item[0][1]
            question = self.iuen_question(iuen[0])
            if iuen[0] == "none":
                answer_sheet = {'no': (True, None), 'yes': (False, 2)}
            else:
                answer_sheet = {'yes': (True, None), 'no': (False, 2)}

        else:
            print("WARNING: Unknown template_id %s" % template_id)
            question = ""
            answer_sheet = None

        # check nested WHERE/HAVING condition
        closest_root_idx = helper_find_closest_bw(tag_seq, pointer - 1, tgt_name="root")
        if closest_root_idx == -1: # not found, not nested
            return question, answer_sheet
        else:
            root_tag = tag_seq[closest_root_idx][0]
            if root_tag == OUTSIDE: # IUEN != none
                return question, answer_sheet
            else:
                closest_end_nested_idx = helper_find_closest_bw(tag_seq, pointer - 1, tgt_name=END_NESTED)
                if closest_end_nested_idx != -1 and closest_end_nested_idx > closest_root_idx:
                    return question, answer_sheet
                if root_tag == WHERE_ROOT_TERM:
                    (col,), op = tag_seq[closest_root_idx][1:3]
                    question = self.where_having_nested_question(col[1], col[0], op[0], question)
                elif root_tag == HAV_ROOT_TERM:
                    (col, agg), op = tag_seq[closest_root_idx][1:3]
                    question = self.where_having_nested_question(col[1], col[0], op[0], question,
                                                                 agg=agg[0] if agg[0] != 'none_agg' else None,
                                                                 bool_having=True)
                else:
                    raise ValueError("Unexpected nested condition: tag_seq: {}\nPointer: {}, closest root: {}.".format(
                        tag_seq, pointer, tag_seq[closest_root_idx]
                    ))
                return question, answer_sheet

    def option_generation(self, cand_hyp_segs, old_tag_seq, pointer):
        """
        Options generation.
        :param cand_hyp_segs: a list of semantic units being the options.
        :param old_tag_seq: the original tag_seq, a sequence of semantic units.
        :param pointer: the pointer to the questioned semantic unit in old_tag_seq.
        :return: NL question, answer_sheet = {choice idx: corresponding decision idx},
                 the index for "none of the above".
        """
        template_id = old_tag_seq[pointer][0]
        answer_sheet = {}
        prefix, option_text = "", ""

        if template_id == SELECT_COL:
            prefix = "Please select any options from the following list that the system needs to return information about:\n"
            for idx, seg in enumerate(cand_hyp_segs):
                tab_col_item, = seg
                option_text += "(%d) %s;\n" % (idx+1, self.agg_col_tab_description(tab_col_item[1][1], tab_col_item[1][0]))
                answer_sheet[idx+1] = tab_col_item[1][-1] # col id

        elif template_id == SELECT_AGG:
            prefix = "Please select any options from the following list that the system needs to return:\n"
            for idx, seg in enumerate(cand_hyp_segs):
                col, (agg, agg_idx) = seg[0][1:3]
                if agg == "none_agg":
                    option_text += "(%d) %s (without performing any mathematical calculations);\n" % (
                        idx + 1, self.agg_col_tab_description(col[1], col[0]))
                else:
                    option_text += "(%d) %s;\n" % (idx+1, self.agg_col_tab_description(col[1], col[0], agg=agg.lower()))
                answer_sheet[idx+1] = (col[-1], agg_idx)

        elif template_id == WHERE_COL:
            prefix = "Please select any options from the following list that the system needs to consider conditions about:\n"
            for idx, seg in enumerate(cand_hyp_segs):
                tab_col_item, = seg
                option_text += "(%d) %s;\n" % (idx + 1, self.agg_col_tab_description(tab_col_item[1][1], tab_col_item[1][0]))
                answer_sheet[idx + 1] = tab_col_item[1][-1] # col id

        elif template_id == WHERE_OP:
            prefix = "Please select any options from the following list that the system needs to consider as conditions:\n"
            for idx, seg in enumerate(cand_hyp_segs):
                ((col,), (op_name, op_idx)) = seg[0][1:3]
                condition_text = "%s %s a value" % (self.agg_col_tab_description(col[1], col[0]),
                                                    self.where_op.get(op_name, op_name))
                option_text += "(%d) %s;\n" % (idx+1, condition_text)
                answer_sheet[idx+1] = (col[-1], op_idx) # (col id, op id)

        elif template_id == WHERE_VAL:
            prefix = "Please select any options from the following list that the system needs to consider as conditions:\n"
            for idx, seg in enumerate(cand_hyp_segs):
                ((col,), (op_name, op_idx), val_item) = seg[0][1:4]
                condition_text = "%s %s \"%s\"" % (self.agg_col_tab_description(col[1], col[0]),
                                                    self.where_op.get(op_name, op_name), val_item[-1])
                option_text += "(%d) %s;\n" % (idx+1, condition_text)
                answer_sheet[idx+1] = (col[-1], op_idx, val_item[-1]) # (col id, op id, val name)

        elif template_id == GROUP_COL:
            prefix = "Please select any options from the following list:\n"
            for idx, seg in enumerate(cand_hyp_segs):
                tab_col_item, = seg
                group_col_text = "The system needs to group %s before doing any mathematical calculations" % (
                    self.group_by_agg_col_tab_description(tab_col_item[1][1], tab_col_item[1][0]))
                option_text += "(%d) %s;\n" % (idx+1, group_col_text)
                answer_sheet[idx+1] = tab_col_item[1][-1] # col id

        elif template_id == HAV_COL:
            prefix = "(Following the last question) Please select any options from the following list that " \
                     "the system needs to consider conditions about:\n"
            for idx, seg in enumerate(cand_hyp_segs):
                tab_col_item, = seg
                option_text += "(%d) %s;\n" % (idx+1, self.agg_col_tab_description(
                    tab_col_item[1][1], tab_col_item[1][0], bool_having=True))
                answer_sheet[idx + 1] = tab_col_item[1][-1] # col id

        elif template_id == HAV_AGG:
            prefix = "(Following the last question) Please select ONE option from the following list that the system " \
                     "needs to consider conditions about:\n"
            for idx, seg in enumerate(cand_hyp_segs):
                col, agg = seg[0][1:3]
                if agg[0] == "none_agg":
                    option_text += "(%d) %s (without performing any mathematical calculations);\n" % (
                        idx+1, self.agg_col_tab_description(col[1], col[0], bool_having=True))
                else:
                    option_text += "(%d) %s;\n" % (idx+1, self.agg_col_tab_description(
                        col[1], col[0], agg=agg[0].lower(), bool_having=True))
                answer_sheet[idx + 1] = (col[-1], agg[1]) # (col id, agg id)

        elif template_id == HAV_OP:
            prefix = "(Following the last question) Please select any options from the following list that " \
                     "the system needs to consider as conditions:\n"
            for idx, seg in enumerate(cand_hyp_segs):
                (col, agg), op = seg[0][1:3]
                condition_text = "%s %s a value" % (self.agg_col_tab_description(
                    col[1], col[0], agg=None if agg[0] == "none_agg" else agg[0], bool_having=True),
                                                    self.where_op.get(op[0], op[0]))
                option_text += "(%d) %s;\n" % (idx+1, condition_text)
                answer_sheet[idx+1] = ((col[-1], agg[1]), op[1]) # ((col id, agg id), op id)

        elif template_id == ORDER_COL:
            prefix = "Please select any options from the following list, based on which (and their calculations) that" \
                     " the system needs to order results:\n"
            for idx, seg in enumerate(cand_hyp_segs):
                tab_col_item, = seg
                option_text += "(%d) %s;\n" % (idx + 1, self.agg_col_tab_description(
                    tab_col_item[1][1], tab_col_item[1][0]))
                answer_sheet[idx + 1] = tab_col_item[1][-1] # col id

        elif template_id == ORDER_AGG:
            prefix = "Please select ONE option from the following list, based on which that the system needs " \
                     "to order results:\n"
            for idx, seg in enumerate(cand_hyp_segs):
                col, agg = seg[0][1:3]
                if agg == "none_agg":
                    option_text += "(%d) %s (without performing any mathematical calculations);\n" % (
                        idx+1, self.agg_col_tab_description(col[1], col[0]))
                else:
                    option_text += "(%d) %s;\n" % (idx + 1, self.agg_col_tab_description(col[1], col[0],
                                                                                         agg=agg[0].lower()))
                answer_sheet[idx + 1] = (col[-1], agg[1]) # (col id, agg id)

        elif template_id == ORDER_DESC_ASC_LIMIT:
            prefix = "(Following the last question) Please select ONE option from the following list:\n"
            for idx, seg in enumerate(cand_hyp_segs):
                (col, agg), desc_asc_limit = seg[0][1:3]
                option_text += "(%d) %s;\n" % (idx+1, "The system should present results " +
                                               self.desc_asc_limit[desc_asc_limit])
                answer_sheet[idx + 1] = ((col[-1], agg[1]), desc_asc_limit)

        elif template_id == IUEN:
            prefix = "Please select ONE option from the following list:\n"

            for idx, seg in enumerate(cand_hyp_segs):
                if seg[0][1][0] == 'none':
                    iuen_text = "The system does NOT need to return information that satisfies a complicated situation " \
                                "as other options indicate"
                elif seg[0][1][0] == 'except':
                    iuen_text = "The system needs to return information that satisfies some cases BUT NOT others, " \
                                "e.g., Find all airlines that have flights from airport 'CVO' BUT NOT from 'APG'"
                elif seg[0][1][0] == 'union':
                    iuen_text = "The system needs to return information that satisfies either some cases OR others, " \
                                "e.g., What are the id and names of the countries which have more than 3 car makers " \
                                "OR produce the 'fiat' model?"
                else:
                    assert seg[0][1][0] == 'intersect'
                    iuen_text = "The system needs to return information that satisfies BOTH some cases AND the others" \
                                " AT THE SAME TIME, e.g., Which district has BOTH stores with less than 3000 products " \
                                "AND stores with more than 10000 products?"
                option_text += "(%d) %s;\n" % (idx+1, iuen_text)
                answer_sheet[idx + 1] = seg[0][1][1] # iuen id

        else:
            print("WARNING: Unknown template_id %s" % template_id)
            return "", answer_sheet, -1

        if template_id != IUEN:
            option_text += "(%d) None of the above options." % (len(cand_hyp_segs)+1)
            question = prefix + option_text
            return question, answer_sheet, len(answer_sheet) + 1
        else:
            question = prefix + option_text.strip()
            return question, answer_sheet, -1
