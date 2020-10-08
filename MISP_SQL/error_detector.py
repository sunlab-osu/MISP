# Error detector
from .utils import semantic_unit_segment, np


class ErrorDetector:
    """
    This is the class for Error Detector.
    """
    def __init__(self):
        return

    def detection(self, tag_seq, start_pos=0, bool_return_first=False, *args, **kwargs):
        """
        Error detection.
        :param tag_seq: a sequence of semantic units.
        :param start_pos: the starting pointer to examine.
        :param bool_return_first: Set to True to return the first error only.
        :return: a list of pairs of (erroneous semantic unit, its position in tag_seq).
        """
        raise NotImplementedError


class ErrorDetectorSim(ErrorDetector):
    """
    This is a simulated error detector which always detects the exact wrong decisions.
    """
    def __init__(self):
        ErrorDetector.__init__(self)

    def detection(self, tag_seq, start_pos=0, bool_return_first=False, eval_tf=None, *args, **kwargs):
        if start_pos >= len(tag_seq):
            return []

        semantic_units, pointers = semantic_unit_segment(tag_seq)
        err_su_pointer_pairs = []
        for semantic_unit, pointer in zip(semantic_units, pointers):
            if pointer < start_pos:
                continue

            bool_correct = eval_tf[pointer]
            if not bool_correct:
                err_su_pointer_pairs.append((semantic_unit, pointer))
                if bool_return_first:
                    return err_su_pointer_pairs

        return err_su_pointer_pairs


class ErrorDetectorProbability(ErrorDetector):
    """
    This is the probability-based error detector.
    """
    def __init__(self, threshold):
        """
        Constructor of the probability-based error detector.
        :param threshold: A float number; the probability threshold.
        """
        ErrorDetector.__init__(self)
        self.prob_threshold = threshold

    def detection(self, tag_seq, start_pos=0, bool_return_first=False, *args, **kwargs):
        if start_pos >= len(tag_seq):
            return []

        semantic_units, pointers = semantic_unit_segment(tag_seq)
        err_su_pointer_pairs = []
        for semantic_unit, pointer in zip(semantic_units, pointers):
            if pointer < start_pos:
                continue

            prob = semantic_unit[-2]
            # if the decision's probability is lower than the threshold, consider it as an error
            if prob < self.prob_threshold:
                err_su_pointer_pairs.append((semantic_unit, pointer))
                if bool_return_first:
                    return err_su_pointer_pairs

        return err_su_pointer_pairs


class ErrorDetectorBayesDropout(ErrorDetector):
    """
    This is the Bayesian Dropout-based error detector.
    """
    def __init__(self, threshold):
        """
        Constructor of the Bayesian Dropout-based error detector.
        :param threshold: A float number; the standard deviation threshold.
        """
        ErrorDetector.__init__(self)
        self.stddev_threshold = threshold

    def detection(self, tag_seq, start_pos=0, bool_return_first=False, *args, **kwargs):
        if start_pos >= len(tag_seq):
            return []

        semantic_units, pointers = semantic_unit_segment(tag_seq)
        err_su_pointer_pairs = []
        for semantic_unit, pointer in zip(semantic_units, pointers):
            if pointer < start_pos:
                continue

            # if the decision's stddev is greater than the threshold, consider it as an error
            stddev = np.std(semantic_unit[-2])
            if stddev > self.stddev_threshold:
                err_su_pointer_pairs.append((semantic_unit, pointer))
                if bool_return_first:
                    return err_su_pointer_pairs

        return err_su_pointer_pairs

