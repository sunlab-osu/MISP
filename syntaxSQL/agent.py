# Interactive SyntaxSQLNet agent
from MISP_SQL.agent import Agent as BaseAgent
from .evaluation import build_foreign_key_map_from_json, evaluate_match_per_example


class Agent(BaseAgent):
    def __init__(self, world_model, error_detector, question_generator):
        BaseAgent.__init__(self, world_model, error_detector, question_generator)

        self.kmaps = build_foreign_key_map_from_json("syntaxSQL/data/tables.json")
        self.db_dir = "syntaxSQL/data/database/"

    def evaluation(self, input_item, hyp, bool_verbal=False):
        hardness, bool_err, exact_score, partial_scores, _, _ = evaluate_match_per_example(
            input_item['query'], hyp.sql, input_item['db_id'], self.db_dir, self.kmaps)

        if bool_verbal:
            print("(Hardness: {}) bool_err {}, exact_score {}, partial_scores {}".format(
                hardness, bool_err, exact_score, partial_scores))

        return hardness, bool_err, exact_score, partial_scores