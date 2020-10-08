"""
Model-based ISP Agent.

Change logs:
12/2019: Function "interactive_parsing_session" has been restructured with a new function
"self.world_model.refresh_decoding" required.

01/2020: Add self.bool_mistake_exit.

02/2020: Add gold simulator handling (verified_gold_opt_selection).
         Allow questions about SQL structure (sel_invalid_structure).

02/2020: Add "undo" choice to support user study.

"""

from .utils import semantic_unit_segment
import copy


class Agent:
    """
    This is the class for Model-based Interactive Semantic Parsing (MISP) agent.
    """
    def __init__(self, world_model, error_detector, question_generator,
                 bool_mistake_exit=False, bool_structure_question=False):
        """
        The constructor for Agent class.
        :param world_model: An instance of MISP_SQL.WorldModel.
        :param error_detector: An instance of MISP_SQL.ErrorDetector.
        :param question_generator: An instance of MISP_SQL.QuestionGenerator.
        :param bool_mistake_exit: Boolean; if True, the agent will not ask for further feedback if any wrong decision
            cannot be resolved in the interaction ("friendly agent").
        :param bool_structure_question: Boolean; if True, the agent can question about SQL's structure (e.g., whether
            WHERE clauses exist).
        """
        self.world_model = world_model
        self.error_detector = error_detector
        self.q_gen = question_generator
        self.bool_mistake_exit = bool_mistake_exit
        self.bool_structure_question = bool_structure_question

    def verified_qa(self, user, question, answer_sheet, pointer, tag_seq):
        """
        Q&A interaction.
        :param user: the user to interact with.
        :param question: the question to the user.
        :param answer_sheet: a dict of {user response: meta info};
               used by user simulator to generate proper feedback.
        :param pointer: the pointer to the questioned unit in the tagged sequence.
        :param tag_seq: a sequence of tagged semantic units.
        :return: user_feedback.
        """
        print("Question: %s" % question)
        user_feedback = user.get_answer(pointer, answer_sheet)
        user.record_user_feedback(tag_seq[pointer], user_feedback, bool_qa=True)

        return user_feedback

    def verified_opt_selection(self, user, opt_question, pointer, semantic_unit, cand_semantic_units,
                               opt_answer_sheet, sel_none_of_above):
        """
        User selection.
        :param user: the user to interact with.
        :param opt_question: the question and the options to the user.
        :param pointer: the pointer to the questioned unit in the tagged sequence.
        :param semantic_unit: the questioned semantic unit.
        :param cand_semantic_units: the list of candidate semantic units.
        :param opt_answer_sheet: a dict of {user selection: meta info};
               used by user simulator to select proper choices.
        :param sel_none_of_above: the index of "none of the above".
        :return: user_selections (a list of indices indicating user selections).
        """
        print("Question: %s" % opt_question)
        user_selections = user.get_selection(pointer, opt_answer_sheet, sel_none_of_above)
        user.option_selections.append((semantic_unit[0], opt_question, user_selections))
        # save to questioned_tags
        for opt_idx, cand_su in enumerate(cand_semantic_units):
            if (opt_idx + 1) in user_selections:
                user.record_user_feedback(cand_su, 'yes', bool_qa=False)
            else:
                user.record_user_feedback(cand_su, 'no', bool_qa=False)

        return user_selections

    def verified_gold_opt_selection(self, user, pointer, semantic_unit):
        """
        Gold user selection.
        :param user: the user to interact with (type = "gold_sim").
        :param pointer: the pointer to the questioned unit in the tagged sequence.
        :param semantic_unit: the questioned semantic unit.
        :return: gold_semantic_units (a list of gold semantic units to replace the old one),
                 gold_dec_items (a list of decisions corresponding to gold_semantic_units, used to incorporate feedback),
                 sel_none_of_above (an index indicating "none of the above"),
                 user selections (a list of indices indicating user selections).
        """
        print("Requesting gold answers...")
        gold_semantic_units, gold_dec_items, sel_none_of_above, user_selections = user.get_gold_selection(pointer)
        user.option_selections.append((semantic_unit[0], None, gold_semantic_units, gold_dec_items))
        # save to questioned tags
        for gold_su in gold_semantic_units:
            user.record_user_feedback(gold_su, 'yes', bool_qa=False)

        return gold_semantic_units, gold_dec_items, sel_none_of_above, user_selections

    def interactive_parsing_session(self, user, input_item, true_output, hyp, bool_verbal=False):
        """
        Interaction session.
        :param user: the user to interact.
        :param input_item: the input to the semantic parser; this is specific to the base parser.
        :param true_output: the true SQL; used by evaluator in user simulation.
        :param hyp: the initial hypothesis generated by the non-interactive base parser.
        :param bool_verbal: set to True to print details about decoding.
        :return: hyp, True/False (whether user exits)
        """
        # setup
        user.update_truth(true_output)
        user.update_pred(hyp.tag_seq, hyp.dec_seq)
        user.clear_counter()
        self.world_model.clear()

        # error detection
        start_pos = 0
        err_su_pointer_pairs = self.error_detector.detection(
            hyp.tag_seq, start_pos=start_pos, bool_return_first=True,
            eval_tf=user.eval_outputs)

        while len(err_su_pointer_pairs): # for each potential erroneous unit
            su, pointer = err_su_pointer_pairs[0]
            semantic_tag = su[0]
            print("Semantic Unit: {}".format(su))

            # question generation
            question, cheat_sheet = self.q_gen.question_generation(su, hyp.tag_seq, pointer)
            if len(question):
                # user Q&A interaction
                user_feedback = self.verified_qa(user, question, cheat_sheet, pointer, hyp.tag_seq)
                if user_feedback == "exit":
                    return hyp, True

                if cheat_sheet[user_feedback][0]: # user affirms the decision
                    self.world_model.apply_pos_feedback(su, hyp.dec_seq, hyp.dec_seq[:su[-1]])
                    start_pos = pointer + 1
                else: # user negates the decision
                    if cheat_sheet[user_feedback][1] == 0:
                        dec_seq_idx = su[-1]
                        dec_prefix = hyp.dec_seq[:dec_seq_idx]

                        # update negated items
                        dec_prefix = self.world_model.apply_neg_feedback(su, hyp.dec_seq, dec_prefix)

                        # perform one-step beam search to generate options
                        cand_hypotheses = self.world_model.decode(
                            input_item, dec_beam_size=self.world_model.num_options,
                            dec_prefix=dec_prefix,
                            avoid_items=self.world_model.avoid_items,
                            confirmed_items=self.world_model.confirmed_items,
                            stop_step=dec_seq_idx, bool_collect_choices=True,
                            bool_verbal=bool_verbal)

                        # prepare options
                        cand_semantic_units = []
                        for cand_hyp in cand_hypotheses:
                            cand_units, cand_pointers = semantic_unit_segment(cand_hyp.tag_seq)
                            assert cand_units[-1][0] == semantic_tag
                            cand_semantic_units.append(cand_units[-1])

                        if user.user_type == "gold_sim":
                            gold_semantic_units, gold_dec_items, sel_none_of_above, user_selections = \
                                self.verified_gold_opt_selection(user, pointer, su)

                            if len(gold_semantic_units):
                                old_dec_prefix = copy.deepcopy(dec_prefix)
                                for gold_su, gold_dec_item in zip(gold_semantic_units, gold_dec_items):
                                    dec_prefix = self.world_model.apply_pos_feedback(
                                        gold_su, old_dec_prefix + [gold_dec_item], dec_prefix)
                            else:
                                # "none of the above" or "invalid structure"
                                for idx in range(len(cand_semantic_units)):
                                    dec_prefix = self.world_model.apply_neg_feedback(
                                        cand_semantic_units[idx], cand_hypotheses[idx].dec_seq, dec_prefix)

                        else: # regular user simulator
                            # present options
                            opt_question, opt_answer_sheet, sel_none_of_above = self.q_gen.option_generation(
                                cand_semantic_units, hyp.tag_seq, pointer)

                            # user selection
                            user_selections = self.verified_opt_selection(
                                user, opt_question, pointer, su, cand_semantic_units, opt_answer_sheet, sel_none_of_above)

                            for idx in range(len(opt_answer_sheet)): # user selection feedback incorporation
                                if idx + 1 in user_selections:
                                    # update dec_prefix for components whose only choice is selected
                                    dec_prefix = self.world_model.apply_pos_feedback(
                                        cand_semantic_units[idx], cand_hypotheses[idx].dec_seq, dec_prefix)
                                else:
                                    dec_prefix = self.world_model.apply_neg_feedback(
                                        cand_semantic_units[idx], cand_hypotheses[idx].dec_seq, dec_prefix)

                        # refresh decoding
                        start_pos, hyp = self.world_model.refresh_decoding(
                            input_item, dec_prefix, hyp, su, pointer,
                            sel_none_of_above, user_selections,
                            bool_verbal=bool_verbal)
                        user.update_pred(hyp.tag_seq, hyp.dec_seq)

                        # a friendly agent will not ask for further feedback if any wrong decision is not resolved.
                        if self.bool_mistake_exit and (sel_none_of_above in user_selections or
                                                       sel_none_of_above + 1 in user_selections):
                            return hyp, False

                    else: # type 1 unit: for decisions with only yes/no choices, we "flip" the current decision
                        assert cheat_sheet[user_feedback][1] == 1
                        dec_seq_idx = su[-1]

                        dec_prefix = self.world_model.apply_neg_feedback(
                            su, hyp.dec_seq, hyp.dec_seq[:dec_seq_idx])
                        try:
                            hyp = self.world_model.decode(input_item, dec_prefix=dec_prefix,
                                                          avoid_items=self.world_model.avoid_items,
                                                          confirmed_items=self.world_model.confirmed_items,
                                                          bool_verbal=bool_verbal)[0]
                        except:
                            pass
                        user.update_pred(hyp.tag_seq, hyp.dec_seq)
                        start_pos = pointer + 1

            else:
                print("WARNING: empty question in su %s, pointer %d\n" % (su, pointer))
                start_pos = pointer + 1

            # error detection in the next turn
            err_su_pointer_pairs = self.error_detector.detection(
                hyp.tag_seq, start_pos=start_pos, bool_return_first=True,
                eval_tf=user.eval_outputs)

        return hyp, False

    def real_user_interactive_parsing_session(self, user, input_item, true_output, hyp, bool_verbal=False):
        """
        Interaction session, curated for real user study.
        :param user: the user to interact.
        :param input_item: the input to the semantic parser; this is specific to the base parser.
        :param true_output: the true SQL; used by evaluator in user simulation.
        :param hyp: the initial hypothesis generated by the non-interactive base parser.
        :param bool_verbal: set to True to print details about decoding.
        :return: hyp, True/False (whether user exits)
        """
        assert user.user_type == "real"

        def undo_execution(questioned_su, avoid_items, confirmed_items):
            assert len(tracker) >= 1, "Invalid undo!"
            hyp, start_pos = tracker.pop()

            # reset user states
            user.update_pred(hyp.tag_seq, hyp.dec_seq)

            # clear feedback after start_pos
            _tag_item_lists, _seg_pointers = semantic_unit_segment(hyp.tag_seq)
            clear_start_pointer = 0
            for clear_start_pointer in _seg_pointers:
                if clear_start_pointer >= start_pos:
                    break
            clear_start_dec_idx = _tag_item_lists[_seg_pointers.index(clear_start_pointer)][-1]
            poped_keys = [k for k in avoid_items.keys() if k >= clear_start_dec_idx]
            for k in poped_keys:
                avoid_items.pop(k)
            poped_keys = [k for k in confirmed_items.keys() if k >= clear_start_dec_idx]
            for k in poped_keys:
                confirmed_items.pop(k)

            # clear the last user feedback records
            last_record = user.feedback_records[-1]
            if last_record == (questioned_su, 'undo'):
                _ = user.feedback_records.pop()
                rm_su = user.feedback_records.pop()[0]
                rm_dec_idx = rm_su[-1]
            else:
                rm_su = user.feedback_records.pop()[0]
                rm_dec_idx = rm_su[-1]
                assert rm_dec_idx == questioned_su[-1]

            rm_start_idx = len(user.feedback_records) - 1
            while rm_start_idx >= 0 and user.feedback_records[rm_start_idx][0][-1] == rm_dec_idx:
                rm_start_idx -= 1
            user.feedback_records = user.feedback_records[:rm_start_idx + 1]

            return hyp, start_pos, avoid_items, confirmed_items

        # setup
        user.update_truth(true_output)
        user.update_pred(hyp.tag_seq, hyp.dec_seq)
        user.clear_counter()
        user.undo_semantic_units = []
        self.world_model.clear()

        # state tracker
        tracker = [] # a list of (hypothesis, starting position in tag_seq)

        # error detection
        start_pos = 0
        err_su_pointer_pairs = self.error_detector.detection(
            hyp.tag_seq, start_pos=start_pos, bool_return_first=True)

        while len(err_su_pointer_pairs): # for each potential erroneous unit
            su, pointer = err_su_pointer_pairs[0]
            semantic_tag = su[0]
            print("\nSemantic Unit: {}".format(su))

            # question generation
            question, cheat_sheet = self.q_gen.question_generation(su, hyp.tag_seq, pointer)
            if len(question):
                # user Q&A interaction
                user_feedback = self.verified_qa(user, question, cheat_sheet, pointer, hyp.tag_seq)
                if user_feedback == "exit":
                    return hyp, True

                if user_feedback == "undo":
                    user.undo_semantic_units.append((su, "Step1"))
                    hyp, start_pos, self.world_model.avoid_items, self.world_model.confirmed_items = undo_execution(
                        su, self.world_model.avoid_items, self.world_model.confirmed_items)

                    # error detection in the next turn
                    err_su_pointer_pairs = self.error_detector.detection(
                        hyp.tag_seq, start_pos=start_pos, bool_return_first=True)
                    continue

                tracker.append((hyp, start_pos))

                if cheat_sheet[user_feedback][0]: # user affirms the decision
                    self.world_model.apply_pos_feedback(su, hyp.dec_seq, hyp.dec_seq[:su[-1]])
                    start_pos = pointer + 1
                else: # user negates the decision
                    if cheat_sheet[user_feedback][1] == 0:
                        dec_seq_idx = su[-1]
                        dec_prefix = hyp.dec_seq[:dec_seq_idx]

                        # update negated items
                        dec_prefix = self.world_model.apply_neg_feedback(su, hyp.dec_seq, dec_prefix)

                        # perform one-step beam search to generate options
                        cand_hypotheses = self.world_model.decode(
                            input_item, dec_beam_size=self.world_model.num_options,
                            dec_prefix=dec_prefix,
                            avoid_items=self.world_model.avoid_items,
                            confirmed_items=self.world_model.confirmed_items,
                            stop_step=dec_seq_idx, bool_collect_choices=True,
                            bool_verbal=bool_verbal)

                        # prepare options
                        cand_semantic_units = []
                        for cand_hyp in cand_hypotheses:
                            cand_units, cand_pointers = semantic_unit_segment(cand_hyp.tag_seq)
                            assert cand_units[-1][0] == semantic_tag
                            cand_semantic_units.append(cand_units[-1])

                        # present options
                        opt_question, opt_answer_sheet, sel_none_of_above = self.q_gen.option_generation(
                            cand_semantic_units, hyp.tag_seq, pointer)

                        if user.bool_undo:
                            undo_opt = sel_none_of_above + (2 if self.bool_structure_question else 1)
                            opt_question = opt_question[:-1] + ";\n" + \
                                           "(%d) I want to undo my last choice!" % undo_opt

                        # user selection
                        user_selections = self.verified_opt_selection(
                            user, opt_question, pointer, su, cand_semantic_units, opt_answer_sheet, sel_none_of_above)

                        if user.bool_undo and user_selections == [undo_opt]:
                            user.undo_semantic_units.append((su, "Step2"))
                            hyp, start_pos, self.world_model.avoid_items, self.world_model.confirmed_items = undo_execution(
                                su, self.world_model.avoid_items, self.world_model.confirmed_items)

                            # error detection in the next turn
                            err_su_pointer_pairs = self.error_detector.detection(
                                hyp.tag_seq, start_pos=start_pos, bool_return_first=True)
                            continue

                        for idx in range(len(opt_answer_sheet)): # user selection feedback incorporation
                            if idx + 1 in user_selections:
                                # update dec_prefix for components whose only choice is selected
                                dec_prefix = self.world_model.apply_pos_feedback(
                                    cand_semantic_units[idx], cand_hypotheses[idx].dec_seq, dec_prefix)
                            else:
                                dec_prefix = self.world_model.apply_neg_feedback(
                                    cand_semantic_units[idx], cand_hypotheses[idx].dec_seq, dec_prefix)

                        # refresh decoding
                        start_pos, hyp = self.world_model.refresh_decoding(
                            input_item, dec_prefix, hyp, su, pointer,
                            sel_none_of_above, user_selections,
                            bool_verbal=bool_verbal)
                        user.update_pred(hyp.tag_seq, hyp.dec_seq)

                        # a friendly agent will not ask for further feedback if any wrong decision is not resolved.
                        if self.bool_mistake_exit and (sel_none_of_above in user_selections or
                                                       sel_none_of_above + 1 in user_selections):
                            return hyp, False

                    else: # type 1 unit: for decisions with only yes/no choices, we "flip" the current decision
                        assert cheat_sheet[user_feedback][1] == 1
                        dec_seq_idx = su[-1]

                        dec_prefix = self.world_model.apply_neg_feedback(
                            su, hyp.dec_seq, hyp.dec_seq[:dec_seq_idx])
                        try:
                            hyp = self.world_model.decode(input_item, dec_prefix=dec_prefix,
                                                          avoid_items=self.world_model.avoid_items,
                                                          confirmed_items=self.world_model.confirmed_items,
                                                          bool_verbal=bool_verbal)[0]
                        except:
                            pass
                        user.update_pred(hyp.tag_seq, hyp.dec_seq)
                        start_pos = pointer + 1

            else:
                print("WARNING: empty question in su %s, pointer %d\n" % (su, pointer))
                start_pos = pointer + 1

            # error detection in the next turn
            err_su_pointer_pairs = self.error_detector.detection(
                hyp.tag_seq, start_pos=start_pos, bool_return_first=True)

            if len(err_su_pointer_pairs) == 0 and user.bool_undo:
                print("\nThe system has finished SQL synthesis. This is the predicted SQL: {}".format(hyp.sql))
                # User can undo this example
                bool_undo_example = input("Please enter if you would like to undo your selections in the previous questions (y/n)?")
                if bool_undo_example == 'y':
                    hyp, start_pos, self.world_model.avoid_items, self.world_model.confirmed_items = undo_execution(
                        su, self.world_model.avoid_items, self.world_model.confirmed_items)

                    # error detection in the next turn
                    err_su_pointer_pairs = self.error_detector.detection(
                        hyp.tag_seq, start_pos=start_pos, bool_return_first=True)

        return hyp, False
