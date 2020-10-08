# Environment: user simulator


class ErrorEvaluator:
    """
    This class defines an error evaluator, which will be used by user simulator (UserSim) to
    simulate users' feedback.
    """
    def __init__(self):
        pass

    def compare(self, g_sql, start_idx, tag_seq, bool_return_true_selections=False,
                bool_return_true_semantic_units=False):
        """
        A comparison between the current SQL with the ground truth.
        :param g_sql: the ground truth SQL.
        :param start_idx: the starting pointer to compare.
        :param tag_seq: a sequence of semantic units.
        :param bool_return_true_selections: Set to True if also returns the true selections for each semantic unit.
               This will be used to simulate users' selections.
        :param bool_return_true_semantic_units: Set to True if also returns the true semantic units to replace each
               old one. This will be used to simulate gold users' selections.
        :return: The ending index, evaluation output (a sequence of True/False, indicating whether the generated
               semantic unit is correct or not), a sequence of true selections (if bool_return_truths is True).
        """
        raise NotImplementedError


class UserSim:
    """
    This is the class for user simulator.
    """
    def __init__(self, error_evaluator):
        """
        Constructor of UserSim.
        :param error_evaluator: An instance of ErrorEvaluator.
        """
        self.user_type = "sim"
        self.patience = 3
        self.error_evaluator = error_evaluator

        self.ground_truth = None
        self.tag_seq = None
        self.dec_seq = None
        self.eval_outputs = None # evaluation output (right/wrong)
        self.true_selections = None # the true decisions used to simulate user selection

        # interaction record
        self.q_counter = 0 # number of questions
        self.questioned_pointers = []
        self.questioned_tags = []
        self.option_selections = []
        self.feedback_records = []

    def update_truth(self, ground_truth):
        """
        Update the user with the ground truth in this session. This will be used to simulate user feedback.
        :param ground_truth: the ground truth SQL query.
        :return:
        """
        self.ground_truth = ground_truth

    def update_pred(self, tag_seq, dec_seq):
        """
        Update the user with the generated SQL query.
        :param tag_seq: a sequence of semantic units (for the generated SQL query).
        :param dec_seq: a sequence of decoding decisions (for the generated SQL query).
        :return:
        """
        self.tag_seq = tag_seq
        self.dec_seq = dec_seq
        _, self.eval_outputs, self.true_selections = self.error_evaluator.compare(
            self.ground_truth, 0, self.tag_seq, bool_return_true_selections=True)

    def record_user_feedback(self, context, user_answer, bool_qa=False):
        """
        Record the interaction feedback.
        :param context: the interaction context.
        :param user_answer: the user feedback.
        :param bool_qa: Set to True if the feedback comes from QA.
        :return:
        """
        self.feedback_records.append((context, user_answer))

        if bool_qa:
            self.questioned_tags.append((context, user_answer))
            self.q_counter += 1 # number of questions + 1

    def clear_counter(self):
        """
        Clear session records.
        :return:
        """
        self.q_counter = 0
        self.questioned_pointers = []
        self.questioned_tags = []
        self.option_selections = []
        self.feedback_records = []

    def get_answer(self, pointer, answer_sheet):
        """
        Generate simulated user answers.
        :param pointer: the pointer to the questioned semantic unit.
        :param answer_sheet: a dict of {'yes'/'no': meta info}, used to simulate users' binary answers.
        :return: the user answer.
        """
        self.questioned_pointers.append(pointer)
        pointer_eval_output = self.eval_outputs[pointer] # True=right, False=wrong

        reverse_answer_sheet = {bool_right: ans for ans, (bool_right, _) in answer_sheet.items()}
        answer = reverse_answer_sheet[pointer_eval_output]

        # examine user patience
        valid_eval_outputs = [item for item_idx, item in enumerate(self.eval_outputs[:pointer + 1])
                              if item is not None and item_idx in self.questioned_pointers]
        if valid_eval_outputs[-3:].count(False) == self.patience:
            answer = 'exit'

        print("User answer: %s.\n" % answer)
        return answer

    def get_selection(self, pointer, answer_sheet, sel_none_of_above):
        """
        Generate simulated user selections.
        :param pointer: the pointer to the questioned semantic unit.
        :param answer_sheet: a dict of {choice idx: corresponding decision idx}, used to simulate users' selections.
        :param sel_none_of_above: the choice index of "none of the above".
        :return: user selections (a list of indices).
        """
        pointer_truth = self.true_selections[pointer] # ground-truth decision
        selections = []

        # if the prefix query is correct, possible true decisions exist
        if pointer_truth is not None:
            for select_id, select_val in answer_sheet.items():
                if len(pointer_truth) and select_val in pointer_truth:
                    selections.append(select_id)
                elif len(pointer_truth) == 0 and select_val is None:
                    selections.append(select_id)

        if len(selections) == 0: # none of the above
            selections.append(sel_none_of_above)

        print("User answer: %s.\n" % str(selections))

        return selections


class GoldUserSim(UserSim):
    """
    This is the class for "gold" user simulator.
    Different to UserSim, when the gold user is requested for a selection, she gives a gold label (if exists).
    """
    def __init__(self, error_evaluator):
        """
        Constructor of GoldUserSim.
        :param error_evaluator: An instance of ErrorEvaluator.
        """
        UserSim.__init__(self, error_evaluator)
        self.user_type = "gold_sim"
        self.true_semantic_units = None

    def update_pred(self, tag_seq, dec_seq):
        """
        Update the user with the generated SQL query.
        :param tag_seq: a sequence of semantic units (for the generated SQL query).
        :return:
        """
        self.tag_seq = tag_seq
        self.dec_seq = dec_seq
        _, self.eval_outputs, self.true_semantic_units = self.error_evaluator.compare(
            self.ground_truth, 0, self.tag_seq, bool_return_true_semantic_units=True)

    def get_gold_selection(self, pointer):
        """
        Generate gold user selections.
        :param pointer: the pointer to the questioned semantic unit.
        :return: gold_semantic_units (a list of gold semantic units),
                 gold_dec_items (a list of dec_items for each gold semantic unit),
                 sel_none_of_above (an integer, the option index referring to "none of the above"),
                 selections (a list of indices that gold user selects).
                 These lists will be used in feedback incorporation.
        """
        raise NotImplementedError


class RealUser(UserSim):
    """
    This is the class for real users (used in user study).
    """
    def __init__(self, error_evaluator, bool_undo=True):
        """
        Constructor of RealUser.
        :param error_evaluator: An instance of ErrorEvaluator.
        """
        UserSim.__init__(self, error_evaluator)
        self.user_type = "real"
        self.bool_undo = bool_undo

        self.undo_semantic_units = []

    def get_answer(self, pointer, *args):
        """
        Request for user answers.
        :param pointer: the pointer to the questioned semantic unit.
        :param args: dummy inputs.
        :return: the user answer.
        """
        self.questioned_pointers.append(pointer)

        if self.bool_undo:
            answer = input("Please enter yes(y)/no(n)/undo/exit: ").lower().strip()
            while answer not in {'yes', 'no', 'exit', 'y', 'n', 'undo'}:
                answer = input("Please enter yes(y)/no(n)/undo/exit: ").lower().strip()
        else:
            answer = input("Please enter yes(y)/no(n)/exit: ").lower().strip()
            while answer not in {'yes', 'no', 'exit', 'y', 'n'}:
                answer = input("Please enter yes(y)/no(n)/exit: ").lower().strip()

        if answer == 'y':
            answer = 'yes'
        elif answer == 'n':
            answer = 'no'

        return answer

    def get_selection(self, pointer, answer_sheet, sel_none_of_above):
        """
        Request for user selections.
        :param pointer: the pointer to the questioned semantic unit.
        :param answer_sheet: dummy inputs.
        :param sel_none_of_above: dummy inputs.
        :return: user selections.
        """
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
                return selections

        answer = input("Please enter the option id(s) delimited by comma ', ': ")
        selections = answer_parsing(answer)
        while selections is None:
            answer = input("Please enter the option id(s) delimited by comma ', ': ")
            selections = answer_parsing(answer)

        return selections
