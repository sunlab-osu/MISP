# world model
from collections import defaultdict


class WorldModel:
    """
    This is the class for world modeling, which takes charge of semantic parsing and user feedback incorporation.
    """
    def __init__(self, semparser, num_options, num_passes=1, dropout_rate=0.0):
        """
        Constructor of WorldModel.
        :param semparser: the base semantic parser.
        :param num_options: number of choices (except "none of the above").
        :param num_passes: number of passes for Bayesian dropout-based decoding.
        :param dropout_rate: dropout rate for Bayesian dropout-based decoding.
        """
        self.semparser = semparser
        self.num_options = num_options

        self.passes = num_passes
        self.dropout_rate = dropout_rate

        # used in feedback incorporation
        self.avoid_items = defaultdict(set) # a record of {decoding position: set of negated decisions}
        self.confirmed_items = defaultdict(set) # a record of {decoding position: set of confirmed decisions}

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
        Semantic parsing in one pass. This function will be used for (1) Regular greedy decoding;
        (2) Performing one-step beam search to generate alternative choices.
        :param input_item: input to the parser (parser-specific).
        :param dec_beam_size: beam search size (int).
        :param dec_prefix: the prefix decoding sequence (list); used when generating alternative choices.
            If specified, the generated queries should share this prefix sequence.
        :param stop_step: the decoding step to terminate (int); used when generating alternative choices.
            If specified, the decoding should terminate at this step. When dec_beam_size > 1, the last step
            in each decoding sequence will be considered as one choice.
        :param avoid_items: a dict of {decoding step: negated decision candidates}.
            If specified, negated choices will not be considered when the decoding proceeds to the according step.
        :param confirmed_items: a dict of {decoding step: confirmed decision candidates}.
            If specified, confirmed choices will be selected when the decoding proceeds to the according step.
        :param dropout_rate: dropout rate in Bayesian dropout (float).
        :param bool_collect_choices: Set to True to collect choices; used when generating alternative choices.
        :param bool_verbal: Set to True to print intermediate information.
        :return: a list of possible hypotheses (class: utils.Hypothesis).
        """
        raise NotImplementedError

    def decode(self, input_item, dec_beam_size=1, dec_prefix=None, stop_step=None,
               avoid_items=None, confirmed_items=None, bool_collect_choices=False, bool_verbal=False):
        """
        Semantic parsing. This function wraps the decode_per_pass function so the latter can be called for
        multiple times (when self.passes > 1) to calculate Bayesian dropout-based uncertainty.
        :param input_item: input to the parser (parser-specific).
        :param dec_beam_size: beam search size (int).
        :param dec_prefix: the prefix decoding sequence (list); used when generating alternative choices.
            If specified, the generated queries should share this prefix sequence.
        :param stop_step: the decoding step to terminate (int); used when generating alternative choices.
            If specified, the decoding should terminate at this step. When dec_beam_size > 1, the last step
            in each decoding sequence will be considered as one choice.
        :param avoid_items: a dict of {decoding step: negated decision candidates}.
            If specified, negated choices will not be considered when the decoding proceeds to the according step.
        :param confirmed_items: a dict of {decoding step: confirmed decision candidates}.
            If specified, confirmed choices will be selected when the decoding proceeds to the according step.
        :param bool_collect_choices: Set to True to collect choices; used when generating alternative choices.
        :param bool_verbal: Set to True to show intermediate information.
        :return: a list of possible hypotheses (class: utils.Hypothesis).
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

    def apply_pos_feedback(self, semantic_unit, dec_seq, dec_prefix):
        """
        Incorporate users' positive feedback (a confirmed semantic unit). The incorporation
        is usually achieved by (1) extending the current prefix decoding sequence (dec_prefix)
        with the confirmed decision and/or (2) adding the confirmed decision into
        self.confirmed_items[dec_idx] (dec_idx is the decoding position of the validated decision).
        :param semantic_unit: a confirmed semantic unit.
        :param dec_seq: the decoding sequence paired with the confirmed semantic unit.
        :param dec_prefix: the current prefix decoding sequence that has been confirmed.
        :return: the updated prefix decoding sequence (list) that has been confirmed.
        """
        raise NotImplementedError

    def apply_neg_feedback(self, semantic_unit, dec_seq, dec_prefix):
        """
        Incorporate users' negative feedback (a negated semantic unit). The incorporation
        is usually achieved by (1) adding the negated decision into self.avoid_items[dec_idx]
        (dec_idx is the decoding position of the validated decision) and/or (2) revising the
        current prefix decoding sequence (dec_prefix) - this is particularly useful for semantic
        units with unit_type=1 (which have binary choices, e.g., AND/OR, DESC/ASC); once the
        current decision is negated, the alternative one can be automatically selected.
        :param semantic_unit: a negated semantic unit.
        :param dec_seq: the decoding sequence paired with the negated semantic unit.
        :param dec_prefix: the current prefix decoding sequence that has been confirmed.
        :return: the updated prefix decoding sequence (list) that has been confirmed.
        """
        raise NotImplementedError

    def decode_revised_structure(self, semantic_unit, pointer, hyp, input_item, bool_verbal=False):
        """
        Revise query structure (as the side effect of user feedback incorporation). For example,
        when the user negated all available columns being WHERE_COL, this function removes the
        WHERE clause. The function is OPTIONAL.
        :param semantic_unit: the questioned semantic unit.
        :param pointer: the pointer to the questioned semantic unit.
        :param hyp: the SQL hypothesis.
        :param input_item: input to the parser (parser-specific).
        :param bool_verbal: set to True to show intermediate information.
        :return: the updated pointer in tag_seq, the updated hypothesis.
        """
        # raise NotImplementedError
        return pointer, hyp

    def refresh_decoding(self, input_item, dec_prefix, old_hyp, semantic_unit,
                         pointer, sel_none_of_above, user_selections, bool_verbal=False):
        """
        Refreshing the decoding after feedback incorporation.
        :param input_item: the input to decoder.
        :param dec_prefix: the current prefix decoding sequence that has been confirmed.
        :param old_hyp: the old decoding hypothesis.
        :param semantic_unit: the semantic unit questioned in current interaction.
        :param pointer: the position of the questioned semantic unit in tag_seq.
        :param sel_none_of_above: the option index corresponding to "none of the above".
        :param user_selections: user selections (list of option indices).
        :param bool_verbal: set to True to show intermediate information.
        :return: the pointer to the next semantic unit to examine, the updated hypothesis.
        """
        raise NotImplementedError

