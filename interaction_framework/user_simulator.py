# user simulator
from collections import defaultdict


class UserSim:
    def __init__(self, err_evaluator):
        self.user_type = "sim"
        self.err_evaluator = err_evaluator
        self.patience = 3

        self.g_sql = None
        self.BIO_history = None
        self.eval_outputs = None

        # qa
        self.waste_q_counter = 0
        self.necessary_q_counter = 0
        self.questioned_indices = []
        self.questioned_tags = []

        self.avoid_items = None
        self.confirmed_items = None
        self.confirmed_indices = []

    def update_truth(self, g_sql):
        self.g_sql = g_sql

    def update_pred(self, BIO_history):
        self.BIO_history = BIO_history
        # if self.user_type == "sim":
        _, self.eval_outputs = self.err_evaluator.compare(self.g_sql, 0, self.BIO_history)

    def clear_counter(self):
        self.waste_q_counter = 0
        self.necessary_q_counter = 0
        self.questioned_indices = []
        self.questioned_tags = []

        self.avoid_items = defaultdict(set)
        self.confirmed_items = defaultdict(set)
        self.confirmed_indices = []

    def get_answer(self, pointer, answer_sheet):
        self.questioned_indices.append(pointer)
        pointer_eval_output = self.eval_outputs[pointer] # True=right, False=wrong

        reverse_answer_sheet = {bool_right: ans for ans, (bool_right, _) in answer_sheet.items()}
        answer = reverse_answer_sheet[pointer_eval_output]

        # examine user patience
        valid_eval_outputs = [item for item_idx, item in enumerate(self.eval_outputs[:pointer + 1])
                              if item is not None and item_idx in self.questioned_indices]
        if valid_eval_outputs[-3:].count(False) == self.patience:
            answer = 'exit'

        return answer


class RealUser(UserSim):
    def __init__(self, err_evaluator):
        UserSim.__init__(self, err_evaluator)
        self.user_type = "real"

    def get_answer(self, pointer, *args):
        self.questioned_indices.append(pointer)
        answer = raw_input("Please enter yes(y)/no(n)/exit: ").lower().strip()
        while answer not in {'yes', 'no', 'exit', 'y', 'n'}:
            answer = raw_input("Please enter yes(y)/no(n)/exit: ").lower().strip()

        if answer == 'y':
            answer = 'yes'
        elif answer == 'n':
            answer = 'no'
        return answer
