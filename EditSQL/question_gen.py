from MISP_SQL.question_gen import QuestionGenerator as BaseQuestionGenerator
from MISP_SQL.utils import WHERE_COL, GROUP_COL, ORDER_AGG_v2, HAV_AGG_v2


class QuestionGenerator(BaseQuestionGenerator):
    def __init__(self, bool_structure_question=False):
        BaseQuestionGenerator.__init__(self)
        self.bool_structure_question = bool_structure_question

    def option_generation(self, cand_semantic_units, old_tag_seq, pointer):
        question, cheat_sheet, sel_none_of_above = BaseQuestionGenerator.option_generation(
            self, cand_semantic_units, old_tag_seq, pointer)

        if self.bool_structure_question:
            semantic_tag = old_tag_seq[pointer][0]

            if semantic_tag == WHERE_COL:
                question = question[:-1] + ';\n'
                sel_invalid_structure = sel_none_of_above + 1
                question += "(%d) The system does not need to consider any conditions." % sel_invalid_structure

            elif semantic_tag == GROUP_COL:
                question = question[:-1] + ';\n'
                sel_invalid_structure = sel_none_of_above + 1
                question += "(%d) The system does not need to group any items." % sel_invalid_structure

            elif semantic_tag == HAV_AGG_v2:
                question = question[:-1] + ';\n'
                sel_invalid_structure = sel_none_of_above + 1
                question += "(%d) The system does not need to consider any conditions." % sel_invalid_structure

            elif semantic_tag == ORDER_AGG_v2:
                question = question[:-1] + ';\n'
                sel_invalid_structure = sel_none_of_above + 1
                question += "(%d) The system does not need to order the results." % sel_invalid_structure

        return question, cheat_sheet, sel_none_of_above


