# heuristic baselines for error detection

import numpy as np
from .question_gen import QuestionGenerator, OUTSIDE


class ErrorDetectorProbability:
    def __init__(self, threshold):
        self.prob_threshold = threshold

    def detection(self, BIO_history, start_pos=0, bool_return_first=False):
        if start_pos >= len(BIO_history):
            return []

        segments, pointers = QuestionGenerator.semantic_unit_segment(BIO_history)
        err_segment_pointer_pairs = []
        for segment, pointer in zip(segments, pointers):
            if pointer < start_pos:
                continue

            prob = segment[0][-2]
            if prob < self.prob_threshold:
                err_segment_pointer_pairs.append((segment, pointer))
                if bool_return_first:
                    return err_segment_pointer_pairs

        return err_segment_pointer_pairs


class ErrorDetectorBayDropout:
    def __init__(self, threshold):
        self.stddev_threshold = threshold

    def detection(self, BIO_history, start_pos=0, bool_return_first=False):
        if start_pos >= len(BIO_history):
            return []

        segments, pointers = QuestionGenerator.semantic_unit_segment(BIO_history)
        err_segment_pointer_pairs = []
        for segment, pointer in zip(segments, pointers):
            if pointer < start_pos:
                continue

            stddev = np.std(segment[0][-2])
            if stddev > self.stddev_threshold:
                err_segment_pointer_pairs.append((segment, pointer))
                if bool_return_first:
                    return err_segment_pointer_pairs

        return err_segment_pointer_pairs


class ErrorDetectorEvaluator:
    def __init__(self):
        pass

    def compare(self, g_sql, start_idx, BIO_history):
        raise NotImplementedError # return idx, eval_output

    def evaluation(self, err_segment_pointer_pairs, g_sql, BIO_history):
        _, eval_output = self.compare(g_sql, 0, BIO_history)

        tp, fp = 0, 0
        for _, pointer in err_segment_pointer_pairs:
            if eval_output[pointer]: # true->right, pred->wrong
                fp += 1
            else: #true->wrong, pred->wrong
                tp += 1

        return tp, fp, eval_output.count(False), eval_output.count(True)

