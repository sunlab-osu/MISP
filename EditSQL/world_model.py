from MISP_SQL.world_model import WorldModel as BaseWorldModel
from MISP_SQL.utils import SELECT_AGG_v2, WHERE_COL, WHERE_OP, WHERE_ROOT_TERM, GROUP_COL, HAV_AGG_v2, \
    HAV_OP_v2, HAV_ROOT_TERM_v2, ORDER_AGG_v2, ORDER_DESC_ASC, ORDER_LIMIT, IUEN_v2, semantic_unit_segment
from collections import defaultdict


class WorldModel(BaseWorldModel):
    def __init__(self, semparser, num_options, kmaps, num_passes=1, dropout_rate=0.0,
                 bool_structure_question=False):
        BaseWorldModel.__init__(self, semparser, num_options,
                                num_passes=num_passes, dropout_rate=dropout_rate)
        self.vocab = None
        if self.semparser is not None:
            self.vocab = self.semparser.decoder.token_predictor.vocabulary

        self.avoid_items = defaultdict(list)
        self.confirmed_items = defaultdict(list)

        self.kmaps = kmaps
        self.bool_structure_question = bool_structure_question

    def clear(self):
        """
        Clear session records.
        """
        self.avoid_items = defaultdict(list)
        self.confirmed_items = defaultdict(list)

    def decode_per_pass(self, input_item, dec_beam_size=1, dec_prefix=None, stop_step=None,
                        avoid_items=None, confirmed_items=None, dropout_rate=0.0,
                        bool_collect_choices=False, bool_verbal=False):

        final_encoder_state, encoder_states, schema_states, max_generation_length, snippets, input_sequence, \
            previous_queries, previous_query_states, input_schema = input_item

        hypotheses = self.semparser.decoder.beam_search(
            final_encoder_state, encoder_states, schema_states, max_generation_length,
            snippets=snippets, input_sequence=input_sequence, previous_queries=previous_queries,
            previous_query_states=previous_query_states, input_schema=input_schema,
            dropout_amount=dropout_rate, stop_step=stop_step, beam_size=dec_beam_size,
            dec_prefix=dec_prefix, avoid_items=avoid_items, confirmed_items=confirmed_items,
            bool_verbal=bool_verbal)

        return hypotheses

    def apply_pos_feedback(self, semantic_unit, dec_seq, dec_prefix):
        semantic_tag = semantic_unit[0]
        dec_seq_idx = semantic_unit[-1]

        if semantic_tag in {SELECT_AGG_v2, ORDER_AGG_v2}:
            if semantic_tag == SELECT_AGG_v2:
                keyword = "select"
            else:
                keyword = "order_by"

            # check duplicates
            prev_cols = []
            st_idx = len(dec_prefix) - 1
            while st_idx > 0 and dec_prefix[st_idx] != self.vocab.token_to_id(keyword):
                if isinstance(dec_prefix[st_idx], list):
                    prev_cols.append(dec_prefix[st_idx])
                st_idx -= 1
            if dec_seq[dec_seq_idx] in prev_cols:
                return dec_prefix

            if isinstance(dec_prefix[-1], list):
                dec_prefix.append(self.vocab.token_to_id(','))
            dec_prefix.append(dec_seq[dec_seq_idx])

        elif semantic_tag == GROUP_COL:
            # check duplicates
            prev_cols = []
            st_idx = len(dec_prefix) - 1
            while st_idx > 0 and dec_prefix[st_idx] != self.vocab.token_to_id('group_by'):
                if dec_prefix[st_idx] != self.vocab.token_to_id(','):
                    prev_cols.append(dec_prefix[st_idx])
                st_idx -= 1
            if dec_seq[dec_seq_idx] in prev_cols:
                return dec_prefix

            if dec_prefix[-1] != self.vocab.token_to_id('group_by') and \
                    dec_prefix[-1] != self.vocab.token_to_id(','):
                dec_prefix.append(self.vocab.token_to_id(','))
            dec_prefix.append(dec_seq[dec_seq_idx])

        elif semantic_tag == WHERE_COL:
            # self.confirmed_items[dec_seq_idx].append(dec_seq[dec_seq_idx])

            # revised 0206: remove duplicates
            prev_cols = []
            if dec_seq_idx in self.confirmed_items:
                prev_cols.extend(self.confirmed_items[dec_seq_idx])
            st_idx = len(dec_prefix) - 1
            while st_idx >= 0 and dec_prefix[st_idx] != self.vocab.token_to_id("where"):
                if not isinstance(dec_prefix[st_idx], list) and dec_prefix[st_idx] >= len(self.vocab):
                    prev_cols.append(dec_prefix[st_idx])
                st_idx -= 1

            if dec_seq[dec_seq_idx] not in prev_cols:
                self.confirmed_items[dec_seq_idx].append(dec_seq[dec_seq_idx])

        elif semantic_tag == HAV_AGG_v2:
            prev_cols = []
            if dec_seq_idx in self.confirmed_items:
                prev_cols.extend(self.confirmed_items[dec_seq_idx])
            st_idx = len(dec_prefix) - 1
            while st_idx >= 0 and dec_prefix[st_idx] != self.vocab.token_to_id("having"):
                if isinstance(dec_prefix[st_idx], list):
                    bool_found_col = False
                    for tok_idx in dec_prefix[st_idx]:
                        if tok_idx >= len(self.vocab):
                            bool_found_col = True
                            break

                    if bool_found_col:
                        prev_cols.append(dec_prefix[st_idx])
                st_idx -= 1

            if dec_seq[dec_seq_idx] not in prev_cols:
                self.confirmed_items[dec_seq_idx].append(dec_seq[dec_seq_idx])

        else: # WHERE_OP, WHERE_ROOT_TERM, HAV_OP_v2, HAV_ROOT_TERM_v2, ORDER_DESC_ASC, ORDER_LIMIT, IUEN_v2
            dec_prefix = dec_seq[:(dec_seq_idx + 1)]

        return dec_prefix

    def apply_neg_feedback(self, semantic_unit, dec_seq, dec_prefix):
        semantic_tag = semantic_unit[0]
        dec_seq_idx = semantic_unit[-1]

        if semantic_tag == ORDER_DESC_ASC:
            cur_decision = dec_seq[dec_seq_idx]
            if cur_decision == self.vocab.token_to_id("asc"):
                new_decision = self.vocab.token_to_id("desc")
            else:
                new_decision = self.vocab.token_to_id("asc")
            dec_prefix = dec_prefix + [new_decision]

        elif semantic_tag == ORDER_LIMIT:
            assert dec_seq[dec_seq_idx + 1] in {self.vocab.token_to_id("_EOS"), self.vocab.token_to_id(")")}
            dec_prefix = dec_prefix + [dec_seq[dec_seq_idx + 1]]

        elif semantic_tag in {WHERE_ROOT_TERM, HAV_ROOT_TERM_v2}:
            cur_decision = dec_seq[dec_seq_idx]
            if cur_decision == self.vocab.token_to_id('value'):
                new_decision = self.vocab.token_to_id('(')
            else:
                new_decision = self.vocab.token_to_id('value')
            dec_prefix = dec_prefix + [new_decision]

        else: # SELECT_AGG, WHERE_COL, WHERE_OP, GROUP_COL, HAV_AGG, HAV_OP, ORDER_AGG, IUEN_v2
            self.avoid_items[dec_seq_idx].append(dec_seq[dec_seq_idx])

        return dec_prefix

    def decode_revised_structure(self, semantic_unit, pointer, hyp, input_item, bool_verbal=False):
        semantic_tag = semantic_unit[0]
        dec_seq_idx = semantic_unit[-1]

        if semantic_tag == IUEN_v2:
            dec_prefix = hyp.dec_seq[:dec_seq_idx] + [self.vocab.token_to_id("_EOS")]

            hyp = self.decode(input_item, dec_beam_size=1,
                              dec_prefix=dec_prefix,
                              avoid_items=self.avoid_items,
                              confirmed_items=self.confirmed_items,
                              bool_verbal=bool_verbal)[0]

        pointer += 1

        return pointer, hyp

    def refresh_decoding(self, input_item, dec_prefix, old_hyp, semantic_unit,
                         pointer, sel_none_of_above, user_selections, bool_verbal=False):
        semantic_tag = semantic_unit[0]
        dec_seq_idx = semantic_unit[-1]

        if self.bool_structure_question and (sel_none_of_above + 1) in user_selections:
            if semantic_tag == WHERE_COL:
                keyword = "where"
            elif semantic_tag == GROUP_COL:
                keyword = "group_by"
            elif semantic_tag == HAV_AGG_v2:
                keyword = "having"
            else:
                assert semantic_tag == ORDER_AGG_v2
                keyword = "order_by"

            cur_dec_seq_idx = dec_seq_idx - 1
            while cur_dec_seq_idx >= 0 and dec_prefix[cur_dec_seq_idx] != self.vocab.token_to_id(keyword):
                cur_dec_seq_idx -= 1
            assert cur_dec_seq_idx >= 0 # must find the keyword
            new_dec_prefix = dec_prefix[:cur_dec_seq_idx] # keep what is before it

            # clear invalid confirmations/negations
            popped_confirmed_keys = [k for k in self.confirmed_items.keys() if k >= cur_dec_seq_idx]
            for popped_dec_idx in popped_confirmed_keys:
                self.confirmed_items.pop(popped_dec_idx)
            popped_avoid_keys = [k for k in self.avoid_items.keys() if k >= cur_dec_seq_idx]
            for popped_dec_idx in popped_avoid_keys:
                popped_items = self.avoid_items.pop(popped_dec_idx)
                if popped_dec_idx == cur_dec_seq_idx:
                    for popped_keyword in ["where", "group_by", "having", "order_by"]:
                        if self.vocab.token_to_id(popped_keyword) in popped_items: # in case of overwriting
                            self.avoid_items[cur_dec_seq_idx].append(self.vocab.token_to_id(popped_keyword))

            # TODO: ban all the follow-up decoding from choosing this keyword
            self.avoid_items[cur_dec_seq_idx].append(self.vocab.token_to_id(keyword)) # no the same keyword followed

            # adjust next examination position (start_pos)
            old_tag_seq = old_hyp.tag_seq[:pointer]
            start_pos = pointer - 1
            while start_pos >= 0 and old_tag_seq[start_pos][-1] >= cur_dec_seq_idx:
                start_pos -= 1
            start_pos += 1

            # re-decode a new sequence
            try:
                hyp = self.decode(input_item, dec_prefix=new_dec_prefix,
                                  avoid_items=self.avoid_items,
                                  confirmed_items=self.confirmed_items,
                                  bool_verbal=bool_verbal)[0]

                if self.vocab.token_to_id(keyword) in hyp.dec_seq[cur_dec_seq_idx:]:
                    print("\nWARNING: same keyword appears!")
                    print("\nDEBUG: new hyp.sql = {}\n".format(hyp.sql))
            except Exception:
                print("\nException in restructure re-decoding:\nold_hyp.sql = {}, old_hyp.dec_seq = {}\n"
                      "new_dec_prefix = {}\n".format(old_hyp.sql, old_hyp.dec_seq, new_dec_prefix))
                hyp = old_hyp
                start_pos = pointer + 1

        else:
            # get the last deciding position after interaction
            cur_dec_seq_idx = max([len(dec_prefix) - 1] + list(self.confirmed_items.keys()) +
                                  list(self.avoid_items.keys()))

            try:
                partial_hyp = self.decode(
                    input_item, dec_prefix=dec_prefix,
                    avoid_items=self.avoid_items,
                    confirmed_items=self.confirmed_items,
                    stop_step=cur_dec_seq_idx,
                    bool_verbal=bool_verbal)[0]
            except Exception:
                if semantic_unit[0] != IUEN_v2:
                    print("Exception in refresh_decoding:\nold_hyp.sql = {}, old_hyp.dec_seq = {}\n"
                          "dec_prefix = {}\n".format(old_hyp.sql, old_hyp.dec_seq, dec_prefix))
                start_pos, hyp = self.decode_revised_structure(
                    semantic_unit, pointer, old_hyp, input_item,
                    bool_verbal=bool_verbal)
            else:
                _, cand_pointers = semantic_unit_segment(partial_hyp.tag_seq)
                last_pointer = cand_pointers[-1]
                start_pos = last_pointer + 1
                hyp = self.decode(
                    input_item, dec_prefix=dec_prefix,
                    avoid_items=self.avoid_items,
                    confirmed_items=self.confirmed_items,
                    bool_verbal=bool_verbal)[0]

        return start_pos, hyp
