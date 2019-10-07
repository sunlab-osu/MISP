# world model
from collections import defaultdict


class WorldModel:
    """
    This is the class for world modeling, which takes charge of semantic parsing and user feedback incorporation.
    """
    def __init__(self, semparser, num_options, bool_str_revisable, bool_seek,
                 num_passes=1, dropout_rate=0.0):
        """
        Constructor of WorldModel.
        :param semparser: the base semantic parser.
        :param num_options: number of choices (except "none of the above").
        :param bool_str_revisable: whether the program structure can be revised accordingly.
        :param bool_seek: when all presented candidates are negated, whether to seek for the next
               most possible one or to take the original prediction.
        :param num_passes: number of passes for Bayesian dropout-based decoding.
        :param dropout_rate: dropout rate for Bayesian dropout-based decoding.
        """
        self.semparser = semparser
        self.num_options = num_options
        self.bool_str_revisable = bool_str_revisable
        self.bool_seek = bool_seek

        self.passes = num_passes
        self.dropout_rate = dropout_rate

        # used in feedback incorporation
        self.avoid_items = defaultdict(set) # a record of negated decisions
        self.confirmed_items = defaultdict(set) # a record of confirmed decisions

    def clear(self):
        """
        Clear session records.
        :return:
        """
        self.avoid_items = defaultdict(set)
        self.confirmed_items = defaultdict(set)

    def decode_per_pass(self, input_item, dec_beam_size=1, dec_prefix=None, stop_step=None,
                        avoid_items=None, confirmed_items=None, dropout_rate=0.0,
                        bool_collect_choices=False, bool_verbal=False):
        """
        Semantic parsing in one pass.
        :param input_item: input to the parser (parser-specific).
        :param dec_beam_size: beam search size.
        :param dec_prefix: the prefix decoding sequence; used when generating alternative choices.
        :param stop_step: the terminal decoding step; specified when generating alternative choices.
        :param avoid_items: a dict of {decoding step: negated candidates}.
        :param confirmed_items: a dict of {decoding step: confirmed candidates}.
        :param dropout_rate: rate in Bayesian dropout.
        :param bool_collect_choices: Set to True to collect choices.
        :param bool_verbal: Set to True to show intermediate information.
        :return: a list of possible hypotheses (class: utils.Hypothesis).
        """
        raise NotImplementedError

    def decode(self, input_item, dec_beam_size=1, dec_prefix=None, stop_step=None,
               avoid_items=None, confirmed_items=None, bool_collect_choices=False, bool_verbal=False):
        """
        Semantic parsing.
        :param input_item: input to the parser (parser-specific).
        :param dec_beam_size: beam search size.
        :param dec_prefix: the prefix decoding sequence; used when generating alternative choices.
        :param stop_step: the terminal decoding step; specified when generating alternative choices.
        :param avoid_items: a dict of {decoding step: negated candidates}.
        :param confirmed_items: a dict of {decoding step: confirmed candidates}.
        :param bool_collect_choices: Set to True to collect choices.
        :param bool_verbal: Set to True to show intermediate information.
        :return:
        """
        # decode without dropout
        hypotheses = self.decode_per_pass(input_item, dec_beam_size=dec_beam_size, dec_prefix=dec_prefix,
                                          stop_step=stop_step, avoid_items=avoid_items,
                                          confirmed_items=confirmed_items,
                                          bool_collect_choices=bool_collect_choices,
                                          bool_verbal=bool_verbal)
        if self.passes == 1 or bool_collect_choices:
            return hypotheses

        # for Bayesian dropout-based decoding, re-decode the same output with dropout
        for hyp in hypotheses:
            for pass_idx in range(self.passes):
                dropout_hyp = self.decode_per_pass(input_item, dec_prefix=hyp.dec_seq, stop_step=stop_step,
                                                   dropout_rate=self.dropout_rate)[0]
                if pass_idx == 0:
                    hyp.set_passes_mode(dropout_hyp)
                else:
                    hyp.merge_hyp(dropout_hyp)
        return hypotheses

    def apply_pos_feedback(self, seg, dec_seq):
        """
        Incorporate users' positive feedback.
        :param seg: the questioned semantic unit.
        :param dec_seq: the decoding sequence.
        :return: the updated prefix decoding sequence that has been confirmed.
        """
        raise NotImplementedError

    def apply_neg_feedback(self, unit_type, seg, dec_seq):
        """
        Incorporate users' negative feedback.
        :param unit_type: the type of semantic units (0, 1, 2).
        :param seg: the questioned semantic unit.
        :param dec_seq: the decoding sequence.
        :return: the updated prefix decoding sequence that has been confirmed.
        """
        raise NotImplementedError

    def decode_revised_structure(self, seg, pointer, hyp, input_item, bool_verbal=False):
        """
        Revise query structure (as the effect of user feedback incorporation).
        :param seg: the questioned semantic unit.
        :param pointer: the pointer to the questioned semantic unit.
        :param hyp: the SQL hypothesis.
        :param input_item: input to the parser (parser-specific).
        :param bool_verbal: Set to True to show intermediate information.
        :return: an updated hypothesis.
        """
        raise NotImplementedError

