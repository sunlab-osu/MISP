# utils
import numpy as np
import copy

SELECT_COL = 'SELECT_COL'
SELECT_AGG = 'SELECT_AGG'
WHERE_COL = 'WHERE_COL'
WHERE_OP = 'WHERE_OP'
WHERE_VAL = 'WHERE_VAL' # for models with value prediction

# spider
WHERE_ROOT_TERM = 'WHERE_ROOT_TERM'
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
OUTSIDE = "O"
END_NESTED = "##END_NESTED##"

# spider -> editsql
ORDER_DESC_ASC = 'ORDER_DESC_ASC' # (ORDER_DESC_ASC, (col, agg, bool_distinct), desc_asc, p(desc_asc), dec_idx)
ORDER_LIMIT = 'ORDER_LIMIT' # (ORDER_DESC_ASC, (col, agg, bool_distinct), bool_limit, p(limit), dec_idx)
SELECT_AGG_v2 = 'SELECT_AGG_v2' # (SELECT_AGG_v2, col, agg, bool_distinct, avg_prob, dec_idx)
ORDER_AGG_v2 = 'ORDER_AGG_v2'
HAV_AGG_v2 = 'HAV_AGG_v2'
HAV_OP_v2 = 'HAV_OP_v2' # (HAV_OP_v2, (col, agg, bool_distinct), op, prob(op), dec_idx)
HAV_ROOT_TERM_v2 = 'HAV_ROOT_TERM_v2' # # (HAV_OP_v2, (col, agg, bool_distinct), op, 'root'/'terminal', prob, dec_idx)
IUEN_v2 = 'IUEN_v2'


def semantic_unit_segment(tag_seq):
    tag_item_lists, seg_pointers = [], []
    for idx, tag_item in enumerate(tag_seq):
        if tag_item[0] != OUTSIDE:
            tag_item_lists.append(tag_item)
            seg_pointers.append(idx)
    return tag_item_lists, seg_pointers


def helper_find_closest_bw(tag_seq, start_idx, tgt_name=None, tgt_id=None):
    skip_nested = []
    idx = start_idx
    while idx > 0:
        if len(skip_nested) > 0:
            if "root" in tag_seq[idx]:
                _ = skip_nested.pop()
            idx -= 1
        else:
            if (tgt_name is not None and tgt_name in tag_seq[idx]) or\
                    (tgt_id is not None and tag_seq[idx][0] == tgt_id): #include tgt_name == END_NESTED
                return idx
            elif END_NESTED in tag_seq[idx]:
                skip_nested.append(idx)
                idx -= 1
            else:
                idx -= 1

    return -1 # not found


class bcolors:
    """
    Usage: print bcolors.WARNING + "Warning: No active frommets remain. Continue?" + bcolors.ENDC
    """
    PINK = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Hypothesis:
    def __init__(self, dec_prefix):
        self.sql = None
        # Note: do not create hyp from scratch during decoding (may lead to wrong self.dec_prefix)
        self.dec_prefix = list(dec_prefix) # given decoding prefix, must execute

        self.tag_seq = [] # sequence of tags
        self.dec_seq = [] # sequence of decisions
        self.dec_seq_idx = 0

        self.logprob = 0.0
        self.length = 0
        self.logprob_list = None

        self.pred_aux_seq = []  # auxiliary information

    def copy(self):
        return copy.deepcopy(self)

    def add_logprob(self, logprob):
        self.logprob += logprob
        self.length += 1

    def set_passes_mode(self, dropout_hyp):
        self.test_tag_seq = list(self.tag_seq) # from decode without dropout

        for tag_idx, tag in enumerate(dropout_hyp.tag_seq):
            item_lst = list(tag)
            item_lst[-2] = [item_lst[-2]]
            self.tag_seq[tag_idx] = item_lst

        self.logprob_list = [dropout_hyp.logprob]

    def merge_hyp(self, hyp):
        # tag_seq, dec_seq, dec_seq_idx, logprob
        assert len(hyp.tag_seq) == len(self.tag_seq)
        for item_idx in range(len(hyp.tag_seq)):
            new_item = hyp.tag_seq[item_idx]
            self.tag_seq[item_idx][-2].append(new_item[-2])

        self.logprob_list.append(hyp.logprob)

    @staticmethod
    def length_penalty(sent_length, length_penalty_factor):
        # Following: https://arxiv.org/abs/1609.08144, Eqn 14, recommend factor = 0.6-0.7.
        # return ((5. + sent_length) / 6.) ** length_penalty_factor
        return (1.0 * sent_length) ** length_penalty_factor

    @staticmethod
    def sort_hypotheses(hypotheses, topK, length_penalty_factor):
        if topK is None:
            topK = np.inf
        sorted_hyps = sorted(hypotheses, key=lambda x: x.logprob / Hypothesis.length_penalty(x.length, length_penalty_factor),
                             reverse=True)
        return_hypotheses = []
        last_score = None
        count = 0
        for hyp in sorted_hyps:
            current_score = hyp.logprob / Hypothesis.length_penalty(hyp.length, length_penalty_factor)
            if last_score is None or current_score < last_score:
                if count < topK:
                    return_hypotheses.append(hyp)
                    last_score = current_score
                    count += 1
                else:
                    break
            else:
                assert current_score == last_score  # tie, include
                return_hypotheses.append(hyp)
        return return_hypotheses

    @staticmethod
    def print_hypotheses(hypotheses):
        for hyp in hypotheses:
            print("logprob: {}, tag_seq: {}\ndec_seq: {}".format(hyp.logprob, hyp.tag_seq, hyp.dec_seq))


