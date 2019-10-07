# Interactive SQL generator

import numpy as np
from collections import defaultdict
import copy

from .question_gen import QuestionGenerator
from .user_simulator import RealUser, UserSim
from .err_detector import ErrorDetectorBayDropout


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

    def copy(self):
        return copy.deepcopy(self)

    def add_logprob(self, logprob):
        self.logprob += logprob
        self.length += 1

    def set_passes_mode(self, dropout_hyp): # used for dropout-based ED
        self.test_tag_seq = list(self.tag_seq) # from decode without dropout

        for tag_idx, tag in enumerate(dropout_hyp.tag_seq):
            item_lst = list(tag)
            item_lst[-2] = [item_lst[-2]]
            self.tag_seq[tag_idx] = item_lst

        self.logprob_list = [dropout_hyp.logprob]

    # def set_to_merge(self):
    #     for tag_idx, tag in enumerate(self.tag_seq):
    #         item_lst = list(tag)
    #         item_lst[-2] = [item_lst[-2]]
    #         self.tag_seq[tag_idx] = item_lst
    #
    #     self.logprob_list = [self.logprob]

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


class ISQL:
    def __init__(self, semparser, error_detector, question_generator, user_simulator,
                 num_options, length_penalty=0.0, bool_structure_rev=False, num_passes=1, dropout_rate=0.):
        self.semparser = semparser
        self.error_detector = error_detector
        self.q_gen = question_generator
        self.user_sim = user_simulator
        self.num_options = num_options # for candidate gen
        self.length_penalty = length_penalty # for beam search
        self.bool_structure_rev = bool_structure_rev # whether the kw structure can be revised

        # qa
        self.waste_q_counter = 0
        self.necessary_q_counter = 0

        if isinstance(self.error_detector, ErrorDetectorBayDropout):
            self.passes = num_passes
            self.dropout_rate = dropout_rate
        else: # do not apply dropout for all other EDs
            self.passes = 1
            self.dropout_rate = 0.0

    def decode_per_pass(self, input_item, dec_beam_size=1, dec_prefix=None, stop_step=None,
                        avoid_items=None, confirmed_items=None, bool_verbal=False, dropout_rate=0.0):
        raise NotImplementedError

    def decode(self, input_item, dec_beam_size=1, dec_prefix=None, stop_step=None,
                        avoid_items=None, confirmed_items=None, bool_verbal=False):
        # decode without dropout
        hypotheses = self.decode_per_pass(input_item, dec_beam_size=dec_beam_size, dec_prefix=dec_prefix,
                                          stop_step=stop_step, avoid_items=avoid_items, confirmed_items=confirmed_items,
                                          bool_verbal=bool_verbal)
        if self.passes == 1:
            return hypotheses

        # for multi-pass case, re-decode the same output with dropout
        for hyp in hypotheses:
            for pass_idx in range(self.passes):
                dropout_hyp = self.decode_per_pass(input_item, dec_prefix=hyp.dec_seq, stop_step=stop_step,
                                                   dropout_rate=self.dropout_rate)[0]
                if pass_idx == 0:
                    hyp.set_passes_mode(dropout_hyp)
                else:
                    hyp.merge_hyp(dropout_hyp)
        return hypotheses

    def verified_qa(self, question, pointer, answer_sheet, tag_seq):
        print("Question: %s" % question)
        user_feedback = self.user_sim.get_answer(pointer, answer_sheet)
        self.user_sim.questioned_tags.append((tag_seq[pointer], user_feedback))

        if self.user_sim.user_type == "sim":
            print("User answer: %s. " % user_feedback)
        print("")

        # if user_feedback != "exit":
        # record q_count
        if self.user_sim.eval_outputs[pointer]: # ask about a right decision -> waste of budge!
            self.user_sim.waste_q_counter += 1
            self.waste_q_counter += 1
        else:
            self.user_sim.necessary_q_counter += 1
            self.necessary_q_counter += 1

        return user_feedback

    def error_detection(self, input_item, sql_truth, hyp, bool_verbal=False):
        """
        :param input_item: input information
        :param sql_truth: the true sql in format that can be used by evaluator
        :param bool_verbal: set to True to show details about decoding
        :return: hyp, True/False (whether user exits)
        """
        # setup user simulator
        self.user_sim.update_truth(sql_truth)
        self.user_sim.update_pred(hyp.tag_seq)
        self.user_sim.clear_counter()

        # error detection
        start_pos = 0
        pos_id2qa_count = {} # to avoid repeatedly ask about one component
        pos_id2backup_hyp = {} #backup the original hyp
        err_segment_pointer_pairs = self.error_detector.detection(
            hyp.tag_seq, start_pos=start_pos, bool_return_first=True)

        while len(err_segment_pointer_pairs): # for each potential error SU
            seg, pointer = err_segment_pointer_pairs[0]
            seg_id = seg[0][0]
            if self.user_sim.user_type == "sim":
                print("SU: {}".format(seg)) # SU = Semantic Unit

            question, answer_sheet, option = self.q_gen.question_generation(seg, hyp.tag_seq, pointer)
            if len(question):
                user_feedback = self.verified_qa(question, pointer, answer_sheet, hyp.tag_seq)
                if user_feedback == "exit":
                    return hyp, True

                if answer_sheet[user_feedback][0]:
                    self.apply_pos_feedback(seg) # update confirmed items
                    self.user_sim.confirmed_indices.append(pointer) # keep track confirmed decisions
                    start_pos = pointer + 1
                else:
                    if answer_sheet[user_feedback][1] == 0:
                        # for "drop" case, record #drops for this position
                        pos_id2qa_count[(pointer, seg_id)] = pos_id2qa_count.get((pointer, seg_id), 0) + 1
                        if pos_id2qa_count[(pointer, seg_id)] == 1:
                            pos_id2backup_hyp[(pointer, seg_id)] = hyp
                        elif pos_id2qa_count[(pointer, seg_id)] > self.num_options:
                            hyp = pos_id2backup_hyp[(pointer, seg_id)] # back to the original hyp
                            self.user_sim.update_pred(hyp.tag_seq)
                            start_pos = pointer + 1
                            err_segment_pointer_pairs = self.error_detector.detection(
                                hyp.tag_seq, start_pos=start_pos, bool_return_first=True)
                            continue

                        hyp = self.apply_neg_feedback(seg, hyp.dec_seq, hyp.tag_seq, pointer,
                                                      answer_sheet[user_feedback][1], input_item,
                                                      bool_verbal=bool_verbal)
                        if hyp is None: # out of option, when structure cannot be changed.
                            hyp = pos_id2backup_hyp[(pointer, seg_id)]
                            self.user_sim.update_pred(hyp.tag_seq)
                            start_pos = pointer + 1
                            err_segment_pointer_pairs = self.error_detector.detection(
                                hyp.tag_seq, start_pos=start_pos, bool_return_first=True)
                            continue

                        self.user_sim.update_pred(hyp.tag_seq)
                        start_pos = pointer # stay, since the original item has been dropped
                    elif answer_sheet[user_feedback][1] == 1:
                        hyp = self.apply_neg_feedback(seg, hyp.dec_seq, hyp.tag_seq, pointer,
                                                      answer_sheet[user_feedback][1], input_item,
                                                      bool_verbal=bool_verbal)
                        self.user_sim.update_pred(hyp.tag_seq)
                        start_pos = pointer + 1
                    else:
                        cand_hypotheses = self.apply_neg_feedback(seg, hyp.dec_seq, hyp.tag_seq, pointer,
                                                                  answer_sheet[user_feedback][1], input_item,
                                                                  bool_verbal=bool_verbal)
                        if cand_hypotheses is not None:
                            # ask for each option
                            for cand_hyp in cand_hypotheses:
                                cand_segs, cand_pointers = QuestionGenerator.semantic_unit_segment(cand_hyp.tag_seq)
                                assert cand_pointers[-1] == pointer
                                cand_seg = cand_segs[-1]
                                cand_question, cand_answer_sheet, cand_option = self.q_gen.question_generation(
                                    cand_seg, cand_hyp.tag_seq, pointer)

                                # cand_complete_hyp = self.decode(input_item, dec_prefix=cand_hyp.dec_seq,
                                #                                 bool_verbal=bool_verbal)[0]
                                # self.user_sim.update_pred(cand_complete_hyp.tag_seq)
                                self.user_sim.update_pred(cand_hyp.tag_seq)

                                cand_user_feedback = self.verified_qa(cand_question, pointer, cand_answer_sheet, cand_hyp.tag_seq)
                                if cand_user_feedback == "exit":
                                    return hyp, True

                                if cand_answer_sheet[cand_user_feedback][0]:
                                    cand_complete_hyp = self.decode(input_item, dec_prefix=cand_hyp.dec_seq,
                                                                    bool_verbal=bool_verbal)[0]
                                    hyp = cand_complete_hyp
                                    break

                            self.user_sim.update_pred(hyp.tag_seq)

                        start_pos = pointer + 1
            else:
                if self.user_sim.user_type == "sim":
                    print("WARNING: empty question in seg %s, pointer %d\n" % (seg, pointer))
                start_pos = pointer + 1

            err_segment_pointer_pairs = self.error_detector.detection(
                hyp.tag_seq, start_pos=start_pos, bool_return_first=True)

        return hyp, False

    def apply_pos_feedback(self, seg):
        raise NotImplementedError

    def apply_neg_feedback(self, seg, dec_seq, BIO_history, pointer, fix_id, input_item, bool_verbal=False):
        raise NotImplementedError