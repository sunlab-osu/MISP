import json
import torch
import datetime
import time
import argparse
import numpy as np
import torch.nn as nn
import traceback
from collections import defaultdict
import copy
from itertools import combinations

from utils import *
from word_embedding import WordEmbedding
from models.agg_predictor import AggPredictor
from models.col_predictor import ColPredictor
from models.desasc_limit_predictor import DesAscLimitPredictor
from models.having_predictor import HavingPredictor
from models.keyword_predictor import KeyWordPredictor
from models.multisql_predictor import MultiSqlPredictor
from models.root_teminal_predictor import RootTeminalPredictor
from models.andor_predictor import AndOrPredictor
from models.op_predictor import OpPredictor
from preprocess_train_dev_data import index_to_column_name #, index_to_column_name_original


SQL_OPS = ('none','intersect', 'union', 'except')
KW_OPS = ('where','groupBy','orderBy')
AGG_OPS = ('max', 'min', 'count', 'sum', 'avg')
ROOT_TERM_OPS = ("root","terminal")
COND_OPS = ("and","or")
DEC_ASC_OPS = (("asc",True),("asc",False),("desc",True),("desc",False))
NEW_WHERE_OPS = ('=','>','<','>=','<=','!=','like','not in','in','between')
KW_WITH_COL = ("select","where","groupBy","orderBy","having")

from interaction_framework.question_gen import SELECT_COL, SELECT_AGG, WHERE_COL, WHERE_OP, WHERE_ROOT_TERM, ANDOR, GROUP_COL, GROUP_NHAV,\
    HAV_COL, HAV_AGG, HAV_OP, HAV_ROOT_TERM, ORDER_COL, ORDER_AGG, ORDER_DESC_ASC_LIMIT, IUEN, MISSING_KW,\
    MISSING_COL, MISSING_AGG, MISSING_OP, REDUNDANT_COL, REDUNDANT_OP, REDUNDANT_AGG, OUTSIDE, END_NESTED
from interaction_framework.ISQL import Hypothesis as BasicHypothesis


class Stack:
     def __init__(self):
         self.items = []

     def isEmpty(self):
         return self.items == []

     def push(self, item):
         self.items.append(item)

     def pop(self):
         return self.items.pop()

     def peek(self):
         return self.items[len(self.items)-1]

     def size(self):
         return len(self.items)

     def insert(self,i,x):
         return self.items.insert(i,x)


def to_batch_tables(tables, B, table_type):
    # col_lens = []
    col_seq = []
    ts = [tables["table_names"],tables["column_names"],tables["column_types"]]
    tname_toks = [x.split(" ") for x in ts[0]]
    col_type = ts[2]
    cols = [x.split(" ") for xid, x in ts[1]]
    tab_seq = [xid for xid, x in ts[1]]
    cols_add = []
    for tid, col, ct in zip(tab_seq, cols, col_type):
        col_one = [ct]
        if tid == -1:
            tabn = ["all"]
        else:
            if table_type=="no": tabn = []
            else: tabn = tname_toks[tid]
        for t in tabn:
            if t not in col:
                col_one.append(t)
        col_one.extend(col)
        cols_add.append(col_one)

    col_seq = [cols_add] * B

    return col_seq


class Hypothesis(BasicHypothesis):
    def __init__(self, dec_prefix):
        BasicHypothesis.__init__(self, dec_prefix)
        self.stack = Stack()
        self.stack.push(("root", None))
        self.history = [["root"]] * 2
        self.current_sql = {}

        # attributes across modules
        self.label = None
        self.andor_cond = ""
        self.has_limit = False
        self.sql_stack = []
        self.idx_stack = []
        self.kw_stack = []
        self.kw = ""
        self.nested_label = ""
        self.has_having = False

        # probs while decoding components (DFS)
        self.prob_stack = Stack()
        self.prob_stack.push((OUTSIDE, ("root", None), 1.0))
        self.tag_seq = [(OUTSIDE, ("root", None), 1.0, None)]  # each tuple = (QID, content, prob*, dec_seq_idx)
        # self.history_dec_prob_kw = []

        # timer
        self.time_spent = 0


class SuperModel(nn.Module):
    def __init__(self, word_emb, N_word, N_h=300, N_depth=2, gpu=True,
                 trainable_emb=False, table_type="std", use_hs=True, dr=0.3, temperature=False):
        super(SuperModel, self).__init__()
        self.gpu = gpu
        self.N_h = N_h
        self.N_depth = N_depth
        self.trainable_emb = trainable_emb
        self.table_type = table_type
        self.use_hs = use_hs
        self.SQL_TOK = ['<UNK>', '<END>', 'WHERE', 'AND', 'EQL', 'GT', 'LT', '<BEG>']

        # word embedding layer
        self.embed_layer = WordEmbedding(word_emb, N_word, gpu,
                self.SQL_TOK, trainable=trainable_emb)

        # initial all modules
        self.multi_sql = MultiSqlPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth,gpu=gpu, use_hs=use_hs, dr=dr, temperature=temperature)
        self.multi_sql.eval()

        self.key_word = KeyWordPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth,gpu=gpu, use_hs=use_hs, dr=dr, temperature=temperature)
        self.key_word.eval()

        self.col = ColPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth,gpu=gpu, use_hs=use_hs, dr=dr, temperature=temperature)
        self.col.eval()

        self.op = OpPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth,gpu=gpu, use_hs=use_hs, dr=dr, temperature=temperature)
        self.op.eval()

        self.agg = AggPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth,gpu=gpu, use_hs=use_hs, dr=dr, temperature=temperature)
        self.agg.eval()

        self.root_teminal = RootTeminalPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth,gpu=gpu, use_hs=use_hs, dr=dr, temperature=temperature)
        self.root_teminal.eval()

        self.des_asc = DesAscLimitPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth,gpu=gpu, use_hs=use_hs, dr=dr, temperature=temperature)
        self.des_asc.eval()

        self.having = HavingPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth,gpu=gpu, use_hs=use_hs, dr=dr, temperature=temperature)
        self.having.eval()

        self.andor = AndOrPredictor(N_word=N_word, N_h=N_h, N_depth=N_depth, gpu=gpu, use_hs=use_hs, dr=dr, temperature=temperature)
        self.andor.eval()

        self.softmax = nn.Softmax() #dim=1
        self.CE = nn.CrossEntropyLoss()
        self.log_softmax = nn.LogSoftmax()
        self.mlsml = nn.MultiLabelSoftMarginLoss()
        self.bce_logit = nn.BCEWithLogitsLoss()
        self.sigm = nn.Sigmoid()
        if gpu:
            self.cuda()
        self.path_not_found = 0

    def forward(self,q_seq,history,tables):
        # if self.part:
        #     return self.part_forward(q_seq,history,tables)
        # else:
        return self.full_forward(q_seq, history, tables)

    def beam_search(self, q_seq, dec_prefix, tables, beam_size, length_penalty_factor=0.0,
                    bool_verbal=True, stop_step=None, avoid_items=None, confirmed_items=None):

        B = len(q_seq)
        if bool_verbal:
            print("q_seq:{}".format(q_seq[0]))
            print("dec_prefix: {}".format(dec_prefix))

        dec_prefix = dec_prefix[::-1]

        q_emb_var, q_len = self.embed_layer.gen_x_q_batch(q_seq)
        col_seq = to_batch_tables(tables, B, self.table_type)
        col_emb_var, col_name_len, col_len = self.embed_layer.gen_col_batch(col_seq)

        mkw_emb_var = self.embed_layer.gen_word_list_embedding(["none", "except", "intersect", "union"], (B))
        mkw_len = np.full(q_len.shape, 4, dtype=np.int64)
        kw_emb_var = self.embed_layer.gen_word_list_embedding(["where", "group by", "order by"], (B))
        kw_len = np.full(q_len.shape, 3, dtype=np.int64)

        max_time = 2  # set timer to prevent infinite recursion in SQL generation
        hypotheses = [Hypothesis(dec_prefix)]
        completed_hypotheses = []

        counter = 0
        while True:
            if bool_verbal: print("\nStep counter: %d" % counter)
            counter += 1
            new_hypotheses = [] # beam for this step

            for hyp in hypotheses:
                if hyp.stack.isEmpty():
                    if len(hyp.sql_stack) > 0:
                        hyp.current_sql = hyp.sql_stack[0]

                    # while len(hyp.history_dec_prob_kw) > 1:
                    #     hyp.history_dec_prob[0].append(hyp.history_dec_prob_kw.pop())
                    #     hyp.history_dec_prob[0].append((OUTSIDE, END_NESTED, 1.0, None))
                    # hyp.history_dec_prob[0].append(hyp.history_dec_prob_kw.pop()) #append the missing_kw question
                    if bool_verbal:
                        print("Completed %d-th hypotheses: " % len(completed_hypotheses))
                        print("history:{}".format(hyp.history[0]))
                        print("tag_seq:{}".format(hyp.tag_seq))
                        print("dec_seq: {}".format(hyp.dec_seq))
                        print("{}".format(hyp.current_sql))
                    completed_hypotheses.append(hyp)  # add to completion
                else:
                    if hyp.time_spent > max_time:
                        if bool_verbal: print("Failed: long time recursion!")
                        continue # failed and not completed -> drop
                    step_hypotheses = None # hypotheses from this step, will be added to 'new_hypotheses'

                    start_time = time.time()
                    vet = hyp.stack.pop()
                    act_prob = hyp.prob_stack.pop()
                    hs_emb_var, hs_len = self.embed_layer.gen_x_history_batch(hyp.history)

                    while len(hyp.idx_stack) > 0 and hyp.stack.size() < hyp.idx_stack[-1]: # if -> while
                        # Come here when you have nested SQL (including IUEN, nested WHERE/HAVING conditions)
                        hyp.idx_stack.pop()
                        hyp.current_sql = hyp.sql_stack.pop()
                        hyp.kw = hyp.kw_stack.pop()
                        # hyp.history_dec_prob[0].append(hyp.history_dec_prob_kw.pop())
                        hyp.tag_seq.append((OUTSIDE, END_NESTED, 1.0, None))

                    if isinstance(vet, tuple) and vet[0] == "root":
                        if hyp.history[0][-1] != "root":
                            hyp.history[0].append("root")
                            hyp.tag_seq.append((OUTSIDE, "root", 1.0, None))
                            hs_emb_var, hs_len = self.embed_layer.gen_x_history_batch(hyp.history)
                        if vet[1] != "original":
                            hyp.idx_stack.append(hyp.stack.size())
                            hyp.sql_stack.append(hyp.current_sql)
                            hyp.kw_stack.append(hyp.kw)
                        else:
                            hyp.idx_stack.append(hyp.stack.size())
                            hyp.sql_stack.append(hyp.sql_stack[-1])
                            hyp.kw_stack.append(hyp.kw)
                        if "sql" in hyp.current_sql:
                            hyp.current_sql["nested_sql"] = {}
                            hyp.current_sql["nested_label"] = hyp.nested_label
                            hyp.current_sql = hyp.current_sql["nested_sql"]
                        elif isinstance(vet[1], dict):
                            vet[1]["sql"] = {}
                            hyp.current_sql = vet[1]["sql"]
                        elif vet[1] != "original":
                            hyp.current_sql["sql"] = {}
                            hyp.current_sql = hyp.current_sql["sql"]

                        if vet[1] == "nested" or vet[1] == "original":
                            hyp.stack.push("none")
                            hyp.history[0].append("none")
                            hyp.prob_stack.push("none")
                            hyp.tag_seq.append((OUTSIDE, "none", 1.0, None))
                        else:
                            # IUEN
                            score = self.multi_sql.forward(q_emb_var, q_len, hs_emb_var, hs_len, mkw_emb_var, mkw_len)
                            score_prob = softmax(score[0].data.cpu().numpy())
                            if len(hyp.dec_prefix):  # if specified
                                partial_vet, label_idx = hyp.dec_prefix.pop()
                                assert partial_vet == vet
                                sorted_label_indices = [label_idx]
                            elif avoid_items is not None and hyp.dec_seq_idx in avoid_items:
                                sorted_label_indices = [idx for idx in np.argsort(-score[0].data.cpu().numpy())
                                                        if idx not in avoid_items[hyp.dec_seq_idx]][:beam_size]
                            else:
                                sorted_label_indices = np.argsort(-score[0].data.cpu().numpy())[:beam_size]

                            # add to beam
                            step_hypotheses = []
                            ticker = None
                            for label_idx in sorted_label_indices:
                                if len(sorted_label_indices) == 1:
                                    new_hyp = hyp
                                else:
                                    new_hyp = hyp.copy()
                                new_hyp.label = SQL_OPS[label_idx]
                                new_hyp.history[0].append(new_hyp.label)
                                new_hyp.stack.push(new_hyp.label)

                                new_hyp.tag_seq.append((IUEN, new_hyp.label, score_prob[label_idx], new_hyp.dec_seq_idx))
                                new_hyp.dec_seq_idx += 1
                                new_hyp.prob_stack.push((new_hyp.label, score_prob[label_idx]))

                                new_hyp.dec_seq.append((copy.deepcopy(vet), label_idx))
                                new_hyp.add_logprob(np.log(score_prob[label_idx]))

                                if ticker is None: # record time spent per each candidate
                                    ticker = time.time() - start_time
                                new_hyp.time_spent += ticker

                                step_hypotheses.append(new_hyp)

                        if step_hypotheses is None:
                            step_hypotheses = [hyp]

                        for step_hyp in step_hypotheses:
                            if step_hyp.label != "none":
                                step_hyp.nested_label = step_hyp.label

                        new_hypotheses.extend(step_hypotheses)

                    elif vet in ('intersect', 'except', 'union'):
                        hyp.stack.push(("root", "nested"))
                        hyp.stack.push(("root", "original"))

                        hyp.prob_stack.push(("root", "nested"))
                        hyp.prob_stack.push(("root", "original"))
                        new_hypotheses.append(hyp)

                    elif vet == "none":
                        score = self.key_word.forward(q_emb_var, q_len, hs_emb_var, hs_len, kw_emb_var, kw_len)
                        kw_num_score, kw_score = [x.data.cpu().numpy() for x in score]
                        kw_num_score_prob = softmax(kw_num_score[0])
                        kw_score_prob = sigmoid(kw_score[0])

                        num_kw_kws_pairs = []
                        if len(hyp.dec_prefix):
                            partial_vet, num_kw, kws = hyp.dec_prefix.pop()
                            assert partial_vet == vet
                            num_kw_kws_pairs.append((num_kw, kws))
                        else:
                            sorted_num_kw_list = np.argsort(-kw_num_score[0])[:beam_size]
                            sorted_kws = np.argsort(-kw_score[0])
                            for num_kw in sorted_num_kw_list:
                                if num_kw == 0:
                                    num_kw_kws_pairs.append((0, []))
                                elif num_kw == 1:
                                    num_kw_kws_pairs.extend([(1, [kw]) for kw in sorted_kws[:beam_size]])
                                else:
                                    combs = combinations(sorted_kws, num_kw)
                                    comb_score = []
                                    for comb in combs:
                                        score = sum([np.log(kw_score_prob[c_idx]) for c_idx in comb])
                                        comb_score.append((comb, score))
                                    sorted_comb_score = sorted(comb_score, key=lambda x:x[1], reverse=True)[:beam_size]
                                    num_kw_kws_pairs.extend([(num_kw, sorted(comb, reverse=True)) for comb, _ in sorted_comb_score])
                                    # num_kw_kws_pairs.extend([(num_kw, sorted(comb, reverse=True)) for comb in combs])

                        step_hypotheses = []
                        ticker = None
                        for num_kw, kws in num_kw_kws_pairs:
                            if len(num_kw_kws_pairs) == 1:
                                step_hyp = hyp
                            else:
                                step_hyp = hyp.copy()
                            # step_hyp.history_dec_prob_kw.append((MISSING_KW, num_kw, [KW_OPS[kw] for kw in kws],
                            #                                      kw_num_score_prob[num_kw], step_hyp.partial_seq_idx))

                            for kw in kws:
                                tag = OUTSIDE
                                step_hyp.stack.push(KW_OPS[kw])
                                step_hyp.prob_stack.push((tag, KW_OPS[kw], kw_num_score_prob[num_kw], kw_score_prob[kw], step_hyp.dec_seq_idx))
                                # step_hyp.logprob += np.log(kw_score_prob[kw]) # add it when working on this kw

                            step_hyp.stack.push("select")
                            step_hyp.prob_stack.push((OUTSIDE, "select", 1.0, step_hyp.dec_seq_idx))  # SELECT_QID + BEGIN
                            step_hyp.dec_seq.append((vet, num_kw, kws))

                            step_hyp.dec_seq_idx += 1
                            step_hyp.add_logprob(np.log(kw_num_score_prob[num_kw]))

                            if ticker is None:
                                ticker = time.time() - start_time
                            step_hyp.time_spent += ticker

                            step_hypotheses.append(step_hyp)

                        # step_hypotheses = sorted(step_hypotheses, key=lambda x: x.logprob/length_penalty(x.length), reverse=True)[:beam_size]
                        step_hypotheses = Hypothesis.sort_hypotheses(step_hypotheses, beam_size, length_penalty_factor)
                        new_hypotheses.extend(step_hypotheses)

                    elif vet in ("select", "orderBy", "where", "groupBy", "having"):
                        hyp.kw = vet
                        hyp.current_sql[hyp.kw] = []
                        hyp.history[0].append(vet)
                        hyp.stack.push(("col", vet))
                        hyp.prob_stack.push(("col", vet))
                        if vet != "having":
                            hyp.tag_seq.append(act_prob)
                            hyp.add_logprob(np.log(act_prob[-2]))  # logprob for kw
                        new_hypotheses.append(hyp)

                    elif isinstance(vet, tuple) and vet[0] == "col":
                        score = self.col.forward(q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len)
                        col_num_score, col_score = [x.data.cpu().numpy() for x in score]
                        col_num_score_prob = softmax(col_num_score[0])
                        col_score_prob = sigmoid(col_score[0])

                        col_num_cols_pair = []
                        if len(hyp.dec_prefix):
                            partial_vet, col_num, cols = hyp.dec_prefix.pop()
                            assert partial_vet == vet
                            col_num_cols_pair.append((col_num, cols))
                        else:
                            sorted_cols = np.argsort(-col_score[0])
                            sorted_col_num = np.argsort(-col_num_score[0]) + 1

                            # filter avoid_items
                            if avoid_items is not None and hyp.dec_seq_idx in avoid_items:
                                sorted_cols = [col for col in sorted_cols if col not in avoid_items[hyp.dec_seq_idx]]
                                sorted_col_num = [col_num for col_num in sorted_col_num if col_num <= len(sorted_cols)]

                            # fix confirmed items
                            if confirmed_items is not None and hyp.dec_seq_idx in confirmed_items:
                                fixed_cols = list(confirmed_items[hyp.dec_seq_idx])
                                sorted_col_num = [col_num - len(fixed_cols) for col_num in sorted_col_num if col_num >= len(fixed_cols)]
                                sorted_cols = [col for col in sorted_cols if col not in fixed_cols] # available candidates
                            else:
                                fixed_cols = []

                            for col_num in sorted_col_num[:beam_size]:
                                if col_num == 0:
                                    col_num_cols_pair.append((len(fixed_cols), fixed_cols))
                                elif col_num == 1:
                                    col_num_cols_pair.extend([(len(fixed_cols) + 1, fixed_cols + [col]) for col in sorted_cols[:beam_size]])
                                elif beam_size == 1:
                                    col_num_cols_pair.append((len(fixed_cols) + col_num, fixed_cols + list(sorted_cols[:col_num])))
                                else:
                                    # combs = combinations(sorted_cols[:max(beam_size, col_num)], col_num)
                                    # col_num_cols_pair.extend([(col_num, list(comb)) for comb in combs])
                                    combs = combinations(sorted_cols[:10], col_num) # to reduce beam search time
                                    comb_score = []
                                    for comb in combs:
                                        score = sum([np.log(col_score_prob[c_idx]) for c_idx in comb])
                                        comb_score.append((comb, score))
                                    sorted_comb_score = sorted(comb_score, key=lambda x:x[1], reverse=True)[:beam_size]
                                    col_num_cols_pair.extend([(len(fixed_cols) + col_num, fixed_cols + list(comb)) for comb, _ in sorted_comb_score])

                        step_hypotheses = []
                        ticker = None
                        for col_num, cols in col_num_cols_pair:
                            if len(col_num_cols_pair) == 1:
                                step_hyp = hyp
                            else:
                                step_hyp = hyp.copy()

                            col_names = [index_to_column_name(col, tables) for col in cols]
                            original_col_names = [index_to_column_name(col, tables) for col in cols]

                            step_hyp.dec_seq.append((vet, col_num, cols))
                            step_hyp.add_logprob(np.log(col_num_score_prob[col_num - 1]))
                            for col, col_name, original_col_name in zip(cols, col_names, original_col_names):
                                if vet[1] == "where":
                                    step_hyp.stack.push(("op", "where", col))
                                    step_hyp.tag_seq.append((WHERE_COL, original_col_name, col_score_prob[col], step_hyp.dec_seq_idx))
                                    step_hyp.prob_stack.push(("op", "where", original_col_name))
                                elif vet[1] != "groupBy":
                                    step_hyp.stack.push(("agg", vet[1], col))
                                    if vet[1] == "select":
                                        tag = SELECT_COL
                                    elif vet[1] == "orderBy":
                                        tag = ORDER_COL
                                    else:
                                        assert vet[1] == "having"
                                        tag = HAV_COL
                                    step_hyp.tag_seq.append((tag, original_col_name, col_score_prob[col], step_hyp.dec_seq_idx))
                                    # step_hyp.prob_stack.push((tag, col_name, col_num_score_prob[col_num - 1], col_score_prob[col], None))
                                    step_hyp.prob_stack.push(("agg", vet[1], original_col_name))
                                elif vet[1] == "groupBy":
                                    step_hyp.history[0].append(col_name)
                                    step_hyp.tag_seq.append((GROUP_COL, original_col_name, col_score_prob[col], step_hyp.dec_seq_idx))
                                    step_hyp.current_sql[step_hyp.kw].append(col_name)

                                step_hyp.add_logprob(np.log(col_score_prob[col]))

                            # step_hyp.history_dec_prob[0].append((REDUNDANT_COL, vet[1], col_num, col_names,
                            #                                      col_num_score_prob[col_num - 1],
                            #                                      [col_score_prob[col] for col in cols],
                            #                                      step_hyp.partial_seq_idx))
                            # step_hyp.history_dec_prob[0].append((MISSING_COL, vet[1], col_num, col_names,
                            #                                      col_num_score_prob[col_num - 1],
                            #                                      [col_score_prob[col] for col in cols],
                            #                                      step_hyp.partial_seq_idx))
                            step_hyp.dec_seq_idx += 1

                            if ticker is None:
                                ticker = time.time() - start_time
                            step_hyp.time_spent += ticker

                            # predict and/or when there is multiple cols in where condition
                            if col_num > 1 and vet[1] == "where":
                                score = self.andor.forward(q_emb_var, q_len, hs_emb_var, hs_len)
                                score_prob = softmax(score[0].data.cpu().numpy())

                                andor_indices = []
                                if len(step_hyp.dec_prefix):
                                    partial_vet, label = step_hyp.dec_prefix.pop()
                                    assert partial_vet == vet
                                    andor_indices.append(label)
                                else:
                                    andor_indices = np.argsort(-score[0].data.cpu().numpy())[:beam_size]

                                for label in andor_indices:
                                    if len(andor_indices) == 1:
                                        step_andor_hyp = step_hyp
                                    else:
                                        step_andor_hyp = step_hyp.copy()

                                    step_andor_hyp.andor_cond = COND_OPS[label]
                                    step_andor_hyp.current_sql[step_andor_hyp.kw].append(step_andor_hyp.andor_cond)
                                    step_andor_hyp.dec_seq.append((vet, label))
                                    step_andor_hyp.tag_seq.append((ANDOR, step_andor_hyp.andor_cond, col_names,
                                                                   score_prob[label], step_andor_hyp.dec_seq_idx))
                                    step_andor_hyp.dec_seq_idx += 1
                                    step_andor_hyp.add_logprob(np.log(score_prob[label]))
                                    step_hypotheses.append(step_andor_hyp)
                            elif vet[1] == "groupBy" and col_num > 0:
                                score = self.having.forward(q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len,
                                                            col_name_len, np.full(B, cols[0], dtype=np.int64))
                                score_prob = softmax(score[0].data.cpu().numpy())

                                having_indices = []
                                if len(step_hyp.dec_prefix):
                                    partial_vet, label = step_hyp.dec_prefix.pop()
                                    assert partial_vet == vet
                                    having_indices.append(label)
                                else:
                                    having_indices = np.argsort(-score[0].data.cpu().numpy())[:beam_size] #[0, 1]

                                for label in having_indices:
                                    if len(having_indices) == 1:
                                        step_having_hyp = step_hyp
                                    else:
                                        step_having_hyp = step_hyp.copy()

                                    step_having_hyp.dec_seq.append((vet, label))
                                    step_having_hyp.add_logprob(np.log(score_prob[label]))

                                    if label == 1:
                                        step_having_hyp.has_having = (label == 1)
                                        step_having_hyp.stack.push("having")
                                        step_having_hyp.tag_seq.append((OUTSIDE, "having", score_prob[1],
                                                                         step_having_hyp.dec_seq_idx))
                                        step_having_hyp.prob_stack.push("having")
                                    else:
                                        step_having_hyp.tag_seq.append((GROUP_NHAV, "none_having", score_prob[0],
                                                                        step_having_hyp.dec_seq_idx))
                                    step_having_hyp.dec_seq_idx += 1
                                    step_hypotheses.append(step_having_hyp)
                            else:
                                step_hypotheses.append(step_hyp)
                        step_hypotheses = Hypothesis.sort_hypotheses(step_hypotheses, beam_size, length_penalty_factor)
                        new_hypotheses.extend(step_hypotheses)

                    elif isinstance(vet, tuple) and vet[0] == "agg":
                        hyp.history[0].append(index_to_column_name(vet[2], tables))
                        # hyp.add_logprob(np.log(act_prob[-1])) #col
                        col_name = act_prob[-1]

                        if vet[1] not in ("having", "orderBy"):  # DEBUG-ed 20180817
                            try:
                                hyp.current_sql[hyp.kw].append(index_to_column_name(vet[2], tables))
                            except Exception as e:
                                # print(e)
                                traceback.print_exc()
                                print("history:{},current_sql:{} stack:{}".format(
                                    hyp.history[0], hyp.current_sql, hyp.stack.items))
                                print("idx_stack:{}".format(hyp.idx_stack))
                                print("sql_stack:{}".format(hyp.sql_stack))
                                exit(1)
                        hs_emb_var, hs_len = self.embed_layer.gen_x_history_batch(hyp.history)

                        score = self.agg.forward(q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len,
                                                 np.full(B, vet[2], dtype=np.int64))
                        agg_num_score, agg_score = [x.data.cpu().numpy() for x in score]
                        agg_num_score_prob = softmax(agg_num_score[0])
                        agg_score_prob = softmax(agg_score[0])

                        agg_num_idxs_pairs = []
                        if len(hyp.dec_prefix):
                            partial_vet, agg_num, agg_idxs = hyp.dec_prefix.pop()
                            assert partial_vet == vet
                            agg_num_idxs_pairs.append((agg_num, agg_idxs))
                        else:
                            sorted_agg_num_list = list(np.argsort(-agg_num_score[0]))
                            if vet[1] in {"having", "orderBy"}:
                                sorted_agg_num_list = [agg_num for agg_num in sorted_agg_num_list if agg_num <= 1]
                            sorted_agg_idxs = list(np.argsort(-agg_score[0]))

                            # filter avoid items
                            if avoid_items is not None and hyp.dec_seq_idx in avoid_items:
                                avoid_aggs = list(avoid_items[hyp.dec_seq_idx])
                                if "none_agg" in avoid_aggs:
                                    sorted_agg_num_list.remove(0)
                                    avoid_aggs.remove("none_agg")
                                sorted_agg_idxs = [agg for agg in sorted_agg_idxs if agg not in avoid_aggs]
                                sorted_agg_num_list = [agg_num for agg_num in sorted_agg_num_list if agg_num <= len(sorted_agg_idxs)]

                            # fix confirmed items
                            if confirmed_items is not None and hyp.dec_seq_idx in confirmed_items:
                                assert vet[1] == "select"
                                fixed_aggs = list(confirmed_items[hyp.dec_seq_idx])
                                sorted_agg_num_list = [agg_num - len(fixed_aggs) for agg_num in sorted_agg_num_list if agg_num >= len(fixed_aggs)]
                                sorted_agg_idxs = [agg for agg in sorted_agg_idxs if agg not in fixed_aggs] # available candidates
                            else:
                                fixed_aggs = []

                            for agg_num in sorted_agg_num_list[:beam_size]:
                                if agg_num == 0:
                                    agg_num_idxs_pairs.append((len(fixed_aggs), fixed_aggs))
                                elif agg_num == 1:
                                    agg_num_idxs_pairs.extend([(len(fixed_aggs) + 1, fixed_aggs + [agg]) for agg in sorted_agg_idxs[:beam_size]])
                                elif beam_size == 1:
                                    agg_num_idxs_pairs.append((len(fixed_aggs) + agg_num, fixed_aggs + sorted_agg_idxs[:agg_num]))
                                else:
                                    # combs = combinations(sorted_agg_idxs[:max(beam_size, agg_num)], agg_num)
                                    # agg_num_idxs_pairs.extend([(agg_num, list(comb)) for comb in combs])
                                    combs = combinations(sorted_agg_idxs, agg_num)
                                    comb_score = []
                                    for comb in combs:
                                        score = sum([np.log(agg_score_prob[c_idx]) for c_idx in comb])
                                        comb_score.append((comb, score))
                                    sorted_comb_score = sorted(comb_score, key=lambda x: x[1], reverse=True)[:beam_size]
                                    agg_num_idxs_pairs.extend([(len(fixed_aggs) + agg_num, fixed_aggs + list(comb))
                                                               for comb, _ in sorted_comb_score])

                        ticker = None
                        step_hypotheses = []
                        for agg_num, agg_idxs in agg_num_idxs_pairs:
                            if len(agg_num_idxs_pairs) == 1:
                                step_hyp = hyp
                            else:
                                step_hyp = hyp.copy()

                            step_hyp.dec_seq.append((vet, agg_num, agg_idxs))
                            step_hyp.add_logprob(np.log(agg_num_score_prob[agg_num]))

                            if len(agg_idxs) > 0:
                                step_hyp.history[0].append(AGG_OPS[agg_idxs[0]])
                                step_hyp.add_logprob(np.log(agg_score_prob[agg_idxs[0]]))
                                if vet[1] not in ("having", "orderBy"):
                                    assert vet[1] == "select"
                                    step_hyp.current_sql[step_hyp.kw].append(AGG_OPS[agg_idxs[0]])
                                    step_hyp.tag_seq.append((SELECT_AGG, col_name, AGG_OPS[agg_idxs[0]],
                                                             agg_score_prob[agg_idxs[0]], step_hyp.dec_seq_idx))
                                elif vet[1] == "orderBy":
                                    step_hyp.stack.push(("des_asc", vet[2], AGG_OPS[agg_idxs[0]]))  # DEBUG-ed 20180817
                                    step_hyp.prob_stack.push(("des_asc", vet[2], (col_name, AGG_OPS[agg_idxs[0]])))
                                    step_hyp.tag_seq.append((ORDER_AGG, col_name, AGG_OPS[agg_idxs[0]],
                                                             agg_score_prob[agg_idxs[0]],
                                                             step_hyp.dec_seq_idx))
                                else:  # having
                                    step_hyp.stack.push(("op", "having", vet[2], AGG_OPS[agg_idxs[0]]))
                                    step_hyp.prob_stack.push(("op", "having", vet[2], (col_name, AGG_OPS[agg_idxs[0]])))
                                    step_hyp.tag_seq.append((HAV_AGG, col_name, AGG_OPS[agg_idxs[0]],
                                                             agg_score_prob[agg_idxs[0]],
                                                             step_hyp.dec_seq_idx))
                            for agg in agg_idxs[1:]:
                                step_hyp.history[0].append(index_to_column_name(vet[2], tables))
                                step_hyp.history[0].append(AGG_OPS[agg])
                                if vet[1] not in ("having", "orderBy"):
                                    step_hyp.current_sql[step_hyp.kw].append(index_to_column_name(vet[2], tables))
                                    step_hyp.current_sql[step_hyp.kw].append(AGG_OPS[agg])
                                    step_hyp.tag_seq.append((SELECT_AGG, col_name, AGG_OPS[agg],
                                                             agg_score_prob[agg], step_hyp.dec_seq_idx))
                                    step_hyp.add_logprob(np.log(agg_score_prob[agg]))
                                elif vet[1] == "orderBy":
                                    raise ValueError("orderBy should not have multiple AGG for {}".format(vet))
                                    # step_hyp.stack.push(("des_asc", vet[2], AGG_OPS[agg]))
                                    # step_hyp.prob_stack.push(("des_asc", vet[2], AGG_OPS[agg]))
                                else:
                                    raise ValueError("having should not have multiple AGG for {}".format(vet))
                                    # step_hyp.stack.push(("op", "having", vet[2], AGG_OPS[agg])) #agg_idxs
                                    # step_hyp.prob_stack.push(("op", "having", vet[2], AGG_OPS[agg], act_prob)) #agg_idxs

                            if len(agg_idxs) == 0:
                                assert agg_num == 0
                                if vet[1] in {"orderBy", "having", "select"}:
                                    if vet[1] == "orderBy":
                                        tag = ORDER_AGG
                                    elif vet[1] == "select":
                                        tag = SELECT_AGG
                                    else:
                                        tag = HAV_AGG
                                    step_hyp.tag_seq.append((tag, col_name, "none_agg",
                                                             agg_num_score_prob[agg_num], step_hyp.dec_seq_idx))
                                if vet[1] not in ("having", "orderBy"):
                                    step_hyp.current_sql[step_hyp.kw].append("none_agg")
                                elif vet[1] == "orderBy":
                                    step_hyp.stack.push(("des_asc", vet[2], "none_agg"))
                                    step_hyp.prob_stack.push(("des_asc", vet[2], (col_name, "none_agg")))
                                else: #having
                                    step_hyp.stack.push(("op", "having", vet[2], "none_agg"))
                                    step_hyp.prob_stack.push(("op", "having", vet[2], (col_name, "none_agg")))

                            # if vet[1] == "select":
                            #     step_hyp.history_dec_prob[0].append((REDUNDANT_AGG, "select", col_name, agg_num,
                            #                                          [AGG_OPS[agg] for agg in agg_idxs],
                            #                                          agg_num_score_prob[agg_num],
                            #                                          [agg_score_prob[agg] for agg in agg_idxs],
                            #                                          step_hyp.partial_seq_idx))
                            #     step_hyp.history_dec_prob[0].append((MISSING_AGG, "select", col_name, agg_num,
                            #                                          [AGG_OPS[agg] for agg in agg_idxs],
                            #                                          agg_num_score_prob[agg_num],
                            #                                          [agg_score_prob[agg] for agg in agg_idxs],
                            #                                          step_hyp.partial_seq_idx))
                            step_hyp.dec_seq_idx += 1

                            if ticker is None:
                                ticker = time.time() - start_time
                            step_hyp.time_spent += ticker
                            step_hypotheses.append(step_hyp)

                        # step_hypotheses = sorted(step_hypotheses, key=lambda x: x.logprob/length_penalty(x.length), reverse=True)[:beam_size]
                        step_hypotheses = Hypothesis.sort_hypotheses(step_hypotheses, beam_size, length_penalty_factor)
                        new_hypotheses.extend(step_hypotheses)

                    elif isinstance(vet, tuple) and vet[0] == "op":  # WHERE, HAVING
                        if vet[1] == "where":
                            hyp.history[0].append(index_to_column_name(vet[2], tables))
                            hs_emb_var, hs_len = self.embed_layer.gen_x_history_batch(hyp.history)
                            # hyp.add_logprob(np.log(act_prob[-2]))  # col
                            col_agg = (act_prob[-1],)
                        else:
                            col_agg = act_prob[-1]

                        score = self.op.forward(q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len,
                                                np.full(B, vet[2], dtype=np.int64))
                        op_num_score, op_score = [x.data.cpu().numpy() for x in score]
                        op_num_score_prob = softmax(op_num_score[0])
                        op_score_prob = sigmoid(op_score[0])

                        op_num_ops_pairs = []
                        if len(hyp.dec_prefix):
                            partial_vet, op_num, ops = hyp.dec_prefix.pop()
                            assert partial_vet == vet
                            op_num_ops_pairs.append((op_num, ops))
                        else:
                            sorted_op_num_list = np.argsort(-op_num_score[0]) + 1
                            sorted_ops = np.argsort(-op_score[0][:len(NEW_WHERE_OPS)]) # TODO: keep consistent?

                            # filter avoid items
                            if avoid_items is not None and hyp.dec_seq_idx in avoid_items:
                                sorted_ops = [op for op in sorted_ops if op not in avoid_items[hyp.dec_seq_idx]]
                                sorted_op_num_list = [op_num for op_num in sorted_op_num_list if op_num <= len(sorted_ops)]

                            # fix confirmed items
                            if confirmed_items is not None and hyp.dec_seq_idx in confirmed_items:
                                fixed_ops = list(confirmed_items[hyp.dec_seq_idx])
                                sorted_op_num_list = [op_num - len(fixed_ops) for op_num in sorted_op_num_list if op_num >= len(fixed_ops)]
                                sorted_ops = [op for op in sorted_ops if op not in fixed_ops]
                            else:
                                fixed_ops = []

                            for op_num in sorted_op_num_list[:beam_size]:
                                # combs = combinations(sorted_ops[:max(beam_size, op_num)], op_num)
                                # op_num_ops_pairs.extend([(op_num, list(comb)) for comb in combs])
                                if op_num == 0:
                                    op_num_ops_pairs.append((len(fixed_ops), fixed_ops))
                                elif op_num == 1:
                                    op_num_ops_pairs.extend([(len(fixed_ops) + 1, fixed_ops + [op]) for op in sorted_ops[:beam_size]])
                                elif beam_size == 1:
                                    op_num_ops_pairs.append((len(fixed_ops) + op_num, fixed_ops + list(sorted_ops[:op_num])))
                                else:
                                    combs = combinations(sorted_ops, op_num)
                                    comb_score = []
                                    for comb in combs:
                                        score = sum([np.log(op_score_prob[c_idx]) for c_idx in comb])
                                        comb_score.append((comb, score))
                                    sorted_comb_score = sorted(comb_score, key=lambda x:x[1], reverse=True)[:beam_size]
                                    op_num_ops_pairs.extend([(len(fixed_ops) + op_num, fixed_ops + list(comb)) for comb, _ in sorted_comb_score])

                        ticker = None
                        step_hypotheses = []
                        for op_num, ops in op_num_ops_pairs:
                            if len(op_num_ops_pairs) == 1:
                                step_hyp = hyp
                            else:
                                step_hyp = hyp.copy()

                            step_hyp.dec_seq.append((vet, op_num, ops))
                            step_hyp.add_logprob(np.log(op_num_score_prob[op_num - 1]))

                            if op_num > 0:
                                step_hyp.history[0].append(NEW_WHERE_OPS[ops[0]])
                                # step_hyp.logprob += op_score_prob[ops[0]]
                                if vet[1] == "having":
                                    step_hyp.stack.push(("root_teminal", vet[2], vet[3], ops[0]))
                                    step_hyp.prob_stack.push(("root_teminal", vet[2], vet[3], vet[1], col_agg, NEW_WHERE_OPS[ops[0]]))
                                    step_hyp.tag_seq.append((HAV_OP, col_agg, NEW_WHERE_OPS[ops[0]],
                                                             op_score_prob[ops[0]], step_hyp.dec_seq_idx))
                                else: #where
                                    step_hyp.stack.push(("root_teminal", vet[2], ops[0]))
                                    step_hyp.prob_stack.push(("root_teminal", vet[2], vet[1], col_agg, NEW_WHERE_OPS[ops[0]]))
                                    step_hyp.tag_seq.append((WHERE_OP, col_agg, NEW_WHERE_OPS[ops[0]],
                                                             op_score_prob[ops[0]], step_hyp.dec_seq_idx))
                                step_hyp.add_logprob(np.log(op_score_prob[ops[0]]))
                            for op in ops[1:]:
                                step_hyp.history[0].append(index_to_column_name(vet[2], tables))
                                step_hyp.history[0].append(NEW_WHERE_OPS[op])
                                # step_hyp.logprob += np.log(op_score_prob[op])
                                if vet[1] == "having":
                                    step_hyp.stack.push(("root_teminal", vet[2], vet[3], op))
                                    step_hyp.prob_stack.push(("root_teminal", vet[2], vet[3], vet[1], col_agg, NEW_WHERE_OPS[op]))
                                    step_hyp.tag_seq.append((HAV_OP, col_agg, NEW_WHERE_OPS[op],
                                                             op_score_prob[op], step_hyp.dec_seq_idx))
                                else:
                                    step_hyp.stack.push(("root_teminal", vet[2], op))
                                    step_hyp.prob_stack.push(("root_teminal", vet[2], vet[1], col_agg, NEW_WHERE_OPS[op]))
                                    step_hyp.tag_seq.append((WHERE_OP, col_agg, NEW_WHERE_OPS[op],
                                                             op_score_prob[op], step_hyp.dec_seq_idx))
                                step_hyp.add_logprob(np.log(op_score_prob[op]))

                            # step_hyp.history_dec_prob[0].append((REDUNDANT_OP, vet[1], col_agg, op_num,
                            #                                      [NEW_WHERE_OPS[op] for op in ops],
                            #                                      op_num_score_prob[op_num - 1],
                            #                                      [op_score_prob[op] for op in ops],
                            #                                      step_hyp.partial_seq_idx))
                            # step_hyp.history_dec_prob[0].append((MISSING_OP, vet[1], col_agg, op_num,
                            #                                      [NEW_WHERE_OPS[op] for op in ops],
                            #                                      op_num_score_prob[op_num - 1],
                            #                                      [op_score_prob[op] for op in ops],
                            #                                      step_hyp.partial_seq_idx))
                            step_hyp.dec_seq_idx += 1

                            if ticker is None:
                                ticker = time.time() - start_time
                            step_hyp.time_spent += ticker
                            step_hypotheses.append(step_hyp)

                        # step_hypotheses = sorted(step_hypotheses, key=lambda x: x.logprob/length_penalty(x.length), reverse=True)[:beam_size]
                        step_hypotheses = Hypothesis.sort_hypotheses(step_hypotheses, beam_size, length_penalty_factor)
                        new_hypotheses.extend(step_hypotheses)

                    elif isinstance(vet, tuple) and vet[0] == "root_teminal":
                        # hyp.add_logprob(np.log(act_prob[-1][-1][-2])) #op
                        (src_kw, col_agg, op_name) = act_prob[-3:]
                        if src_kw == "where":
                            tag = WHERE_ROOT_TERM
                        else:
                            assert src_kw == "having"
                            tag = HAV_ROOT_TERM

                        score = self.root_teminal.forward(q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len,
                                                          col_name_len, np.full(B, vet[1], dtype=np.int64))
                        score_prob = softmax(score[0].data.cpu().numpy())
                        if len(hyp.dec_prefix):
                            partial_vet, label_idx = hyp.dec_prefix.pop()
                            assert partial_vet == vet
                            label_indices = [label_idx]
                        else:
                            # label_idx = np.argmax(score[0].data.cpu().numpy())
                            label_indices = np.argsort(-score[0].data.cpu().numpy())[:beam_size]

                        ticker = None
                        for label_idx in label_indices:
                            if len(label_indices) == 1:
                                step_hyp = hyp
                            else:
                                step_hyp = hyp.copy()

                            step_hyp.label = ROOT_TERM_OPS[label_idx]

                            step_hyp.dec_seq.append((vet, label_idx))
                            step_hyp.add_logprob(np.log(score_prob[label_idx]))

                            if len(vet) == 4:
                                step_hyp.current_sql[step_hyp.kw].append(index_to_column_name(vet[1], tables))
                                step_hyp.current_sql[step_hyp.kw].append(vet[2])
                                step_hyp.current_sql[step_hyp.kw].append(NEW_WHERE_OPS[vet[3]])
                            else:
                                # print("kw:{}".format(kw))
                                try:
                                    step_hyp.current_sql[step_hyp.kw].append(index_to_column_name(vet[1], tables))
                                except Exception as e:
                                    # print(e)
                                    traceback.print_exc()
                                    print("history:{},current_sql:{} stack:{}".format(
                                        step_hyp.history[0], step_hyp.current_sql, step_hyp.stack.items))
                                    print("idx_stack:{}".format(step_hyp.idx_stack))
                                    print("sql_stack:{}".format(step_hyp.sql_stack))
                                    exit(1)
                                step_hyp.current_sql[step_hyp.kw].append(NEW_WHERE_OPS[vet[2]])
                            if step_hyp.label == "root":
                                step_hyp.history[0].append("root")
                                step_hyp.current_sql[step_hyp.kw].append({})
                                step_hyp.stack.push(("root", step_hyp.current_sql[step_hyp.kw][-1]))

                                step_hyp.tag_seq.append((tag, col_agg, op_name, "root", score_prob[label_idx], step_hyp.dec_seq_idx))
                                step_hyp.prob_stack.push(("root", step_hyp.current_sql[step_hyp.kw][-1]))
                                step_hyp.dec_seq_idx += 1
                            else:
                                step_hyp.current_sql[step_hyp.kw].append("terminal")
                                step_hyp.tag_seq.append((tag, col_agg, op_name, "terminal", score_prob[label_idx], step_hyp.dec_seq_idx))
                                step_hyp.dec_seq_idx += 1

                            if ticker is None:
                                ticker = time.time() - start_time
                            step_hyp.time_spent += ticker
                            new_hypotheses.append(step_hyp)

                    elif isinstance(vet, tuple) and vet[0] == "des_asc":
                        col_agg = act_prob[-1]

                        hyp.current_sql[hyp.kw].append(index_to_column_name(vet[1], tables))
                        hyp.current_sql[hyp.kw].append(vet[2])
                        score = self.des_asc.forward(q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len,
                                                     np.full(B, vet[1], dtype=np.int64))
                        score_prob = softmax(score[0].data.cpu().numpy())
                        if len(hyp.dec_prefix):
                            partial_vet, label = hyp.dec_prefix.pop()
                            assert partial_vet == vet
                            sorted_label_indices = [label]
                        elif avoid_items is not None and hyp.dec_seq_idx in avoid_items:
                            sorted_label_indices = [idx for idx in np.argsort(-score[0].data.cpu().numpy())
                                                    if idx not in avoid_items[hyp.dec_seq_idx]][:beam_size]
                        else:
                            sorted_label_indices = np.argsort(-score[0].data.cpu().numpy())[:beam_size]

                        for label in sorted_label_indices:
                            if len(sorted_label_indices) == 1:
                                step_hyp = hyp
                            else:
                                step_hyp = hyp.copy()
                            step_hyp.dec_seq.append((vet, label))
                            step_hyp.add_logprob(np.log(score_prob[label]))

                            dec_asc, has_limit = DEC_ASC_OPS[label]
                            step_hyp.history[0].append(dec_asc)
                            step_hyp.tag_seq.append((ORDER_DESC_ASC_LIMIT, col_agg,
                                                     (dec_asc, has_limit), score_prob[label],
                                                     step_hyp.dec_seq_idx))
                            step_hyp.dec_seq_idx += 1
                            step_hyp.current_sql[step_hyp.kw].append(dec_asc)
                            step_hyp.current_sql[step_hyp.kw].append(has_limit)
                            new_hypotheses.append(step_hyp)

            if len(new_hypotheses) == 0:
                # sort completed hypotheses
                # sorted_completed_hypotheses = sorted(completed_hypotheses, key=lambda x:x.logprob/length_penalty(x.length), reverse=True)[:beam_size]
                sorted_completed_hypotheses = Hypothesis.sort_hypotheses(completed_hypotheses, beam_size, length_penalty_factor)
                # outputs = [(hyp.current_sql, hyp.history_dec_prob[0], hyp.partial_seq) for hyp in sorted_completed_hypotheses]
                return sorted_completed_hypotheses

            # sort new_hypotheses
            if bool_verbal: Hypothesis.print_hypotheses(new_hypotheses)
            # sorted_new_hypotheses = sorted(new_hypotheses, key=lambda x:x.logprob/length_penalty(x.length), reverse=True)
            # hypotheses = sorted_new_hypotheses[:beam_size]
            hypotheses = Hypothesis.sort_hypotheses(new_hypotheses, beam_size, length_penalty_factor)

            if stop_step is not None: # for one-step beam search; the partial_seq lengths must be the same for all hyps
                dec_seq_length = len(hypotheses[0].dec_seq)
                if dec_seq_length == stop_step + 1:
                    for hyp in hypotheses:
                        assert len(hyp.dec_seq) == dec_seq_length
                    return hypotheses

    def full_forward(self, q_seq, history, tables):
        B = len(q_seq)
        # print("q_seq:{}".format(q_seq))
        # print("Batch size:{}".format(B))
        q_emb_var, q_len = self.embed_layer.gen_x_q_batch(q_seq)
        col_seq = to_batch_tables(tables, B, self.table_type)
        col_emb_var, col_name_len, col_len = self.embed_layer.gen_col_batch(col_seq)

        mkw_emb_var = self.embed_layer.gen_word_list_embedding(["none","except","intersect","union"],(B))
        mkw_len = np.full(q_len.shape, 4,dtype=np.int64)
        kw_emb_var = self.embed_layer.gen_word_list_embedding(["where", "group by", "order by"], (B))
        kw_len = np.full(q_len.shape, 3, dtype=np.int64)

        stack = Stack()
        stack.push(("root",None))
        history = [["root"]]*B
        andor_cond = ""
        has_limit = False
        # sql = {}
        current_sql = {}
        sql_stack = []
        idx_stack = []
        kw_stack = []
        kw = ""
        nested_label = ""
        has_having = False

        timeout = time.time() + 2 # set timer to prevent infinite recursion in SQL generation
        failed = False
        while not stack.isEmpty():
            if time.time() > timeout: failed=True; break
            vet = stack.pop()
            # print(vet)
            hs_emb_var, hs_len = self.embed_layer.gen_x_history_batch(history)
            if len(idx_stack) > 0 and stack.size() < idx_stack[-1]:
                # print("pop!!!!!!!!!!!!!!!!!!!!!!")
                idx_stack.pop()
                current_sql = sql_stack.pop()
                kw = kw_stack.pop()
                # current_sql = current_sql["sql"]
            # history.append(vet)
            # print("hs_emb:{} hs_len:{}".format(hs_emb_var.size(),hs_len.size()))
            if isinstance(vet,tuple) and vet[0] == "root":
                if history[0][-1] != "root":
                    history[0].append("root")
                    hs_emb_var, hs_len = self.embed_layer.gen_x_history_batch(history)
                if vet[1] != "original":
                    idx_stack.append(stack.size())
                    sql_stack.append(current_sql)
                    kw_stack.append(kw)
                else:
                    idx_stack.append(stack.size())
                    sql_stack.append(sql_stack[-1])
                    kw_stack.append(kw)
                if "sql" in current_sql:
                    current_sql["nested_sql"] = {}
                    current_sql["nested_label"] = nested_label
                    current_sql = current_sql["nested_sql"]
                elif isinstance(vet[1],dict):
                    vet[1]["sql"] = {}
                    current_sql = vet[1]["sql"]
                elif vet[1] != "original":
                    current_sql["sql"] = {}
                    current_sql = current_sql["sql"]
                # print("q_emb_var:{} hs_emb_var:{} mkw_emb_var:{}".format(q_emb_var.size(),hs_emb_var.size(),mkw_emb_var.size()))
                if vet[1] == "nested" or vet[1] == "original":
                    stack.push("none")
                    history[0].append("none")
                else:
                    score = self.multi_sql.forward(q_emb_var,q_len,hs_emb_var,hs_len,mkw_emb_var,mkw_len)
                    label = np.argmax(score[0].data.cpu().numpy())
                    label = SQL_OPS[label]
                    history[0].append(label)
                    stack.push(label)
                if label != "none":
                    nested_label = label

            elif vet in ('intersect', 'except', 'union'):
                stack.push(("root","nested"))
                stack.push(("root","original"))
                # history[0].append("root")
            elif vet == "none":
                score = self.key_word.forward(q_emb_var,q_len,hs_emb_var,hs_len,kw_emb_var,kw_len)
                kw_num_score, kw_score = [x.data.cpu().numpy() for x in score]
                # print("kw_num_score:{}".format(kw_num_score))
                # print("kw_score:{}".format(kw_score))
                num_kw = np.argmax(kw_num_score[0])
                kw_score = list(np.argsort(-kw_score[0])[:num_kw])
                kw_score.sort(reverse=True)
                # print("num_kw:{}".format(num_kw))
                for kw in kw_score:
                    stack.push(KW_OPS[kw])
                stack.push("select")
            elif vet in ("select","orderBy","where","groupBy","having"):
                kw = vet
                current_sql[kw] = []
                history[0].append(vet)
                stack.push(("col",vet))
                # score = self.andor.forward(q_emb_var,q_len,hs_emb_var,hs_len)
                # label = score[0].data.cpu().numpy()
                # andor_cond = COND_OPS[label]
                # history.append("")
            # elif vet == "groupBy":
            #     score = self.having.forward(q_emb_var,q_len,hs_emb_var,hs_len,col_emb_var,col_len,)
            elif isinstance(vet,tuple) and vet[0] == "col":
                # print("q_emb_var:{} hs_emb_var:{} col_emb_var:{}".format(q_emb_var.size(), hs_emb_var.size(),col_emb_var.size()))
                score = self.col.forward(q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len)
                col_num_score, col_score = [x.data.cpu().numpy() for x in score]
                col_num = np.argmax(col_num_score[0]) + 1  # double check
                cols = np.argsort(-col_score[0])[:col_num]
                # print(col_num)
                # print("col_num_score:{}".format(col_num_score))
                # print("col_score:{}".format(col_score))
                for col in cols:
                    if vet[1] == "where":
                        stack.push(("op","where",col))
                    elif vet[1] != "groupBy":
                        stack.push(("agg",vet[1],col))
                    elif vet[1] == "groupBy":
                        history[0].append(index_to_column_name(col, tables))
                        current_sql[kw].append(index_to_column_name(col, tables))
                #predict and or or when there is multi col in where condition
                if col_num > 1 and vet[1] == "where":
                    score = self.andor.forward(q_emb_var,q_len,hs_emb_var,hs_len)
                    label = np.argmax(score[0].data.cpu().numpy())
                    andor_cond = COND_OPS[label]
                    current_sql[kw].append(andor_cond)
                if vet[1] == "groupBy" and col_num > 0:
                    score = self.having.forward(q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len, np.full(B, cols[0],dtype=np.int64))
                    label = np.argmax(score[0].data.cpu().numpy())
                    if label == 1:
                        has_having = (label == 1)
                        # stack.insert(-col_num,"having")
                        stack.push("having")
                # history.append(index_to_column_name(cols[-1], tables[0]))
            elif isinstance(vet,tuple) and vet[0] == "agg":
                history[0].append(index_to_column_name(vet[2], tables))
                if vet[1] not in ("having","orderBy"): #DEBUG-ed 20180817
                    try:
                        current_sql[kw].append(index_to_column_name(vet[2], tables))
                    except Exception as e:
                        # print(e)
                        traceback.print_exc()
                        print("history:{},current_sql:{} stack:{}".format(history[0], current_sql,stack.items))
                        print("idx_stack:{}".format(idx_stack))
                        print("sql_stack:{}".format(sql_stack))
                        exit(1)
                hs_emb_var, hs_len = self.embed_layer.gen_x_history_batch(history)

                score = self.agg.forward(q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len, np.full(B, vet[2],dtype=np.int64))
                agg_num_score, agg_score = [x.data.cpu().numpy() for x in score]
                agg_num = np.argmax(agg_num_score[0])  # double check
                agg_idxs = np.argsort(-agg_score[0])[:agg_num]
                # print("agg:{}".format([AGG_OPS[agg] for agg in agg_idxs]))
                if len(agg_idxs) > 0:
                    history[0].append(AGG_OPS[agg_idxs[0]])
                    if vet[1] not in ("having", "orderBy"):
                        current_sql[kw].append(AGG_OPS[agg_idxs[0]])
                    elif vet[1] == "orderBy":
                        stack.push(("des_asc", vet[2], AGG_OPS[agg_idxs[0]])) #DEBUG-ed 20180817
                    else:
                        stack.push(("op","having",vet[2],AGG_OPS[agg_idxs[0]]))
                for agg in agg_idxs[1:]:
                    history[0].append(index_to_column_name(vet[2], tables))
                    history[0].append(AGG_OPS[agg])
                    if vet[1] not in ("having", "orderBy"):
                        current_sql[kw].append(index_to_column_name(vet[2], tables))
                        current_sql[kw].append(AGG_OPS[agg])
                    elif vet[1] == "orderBy":
                        stack.push(("des_asc", vet[2], AGG_OPS[agg]))
                    else:
                        stack.push(("op", "having", vet[2], agg_idxs))
                if len(agg_idxs) == 0:
                    if vet[1] not in ("having", "orderBy"):
                        current_sql[kw].append("none_agg")
                    elif vet[1] == "orderBy":
                        stack.push(("des_asc", vet[2], "none_agg"))
                    else:
                        stack.push(("op", "having", vet[2], "none_agg"))
                # current_sql[kw].append([AGG_OPS[agg] for agg in agg_idxs])
                # if vet[1] == "having":
                #     stack.push(("op","having",vet[2],agg_idxs))
                # if vet[1] == "orderBy":
                #     stack.push(("des_asc",vet[2],agg_idxs))
                # if vet[1] == "groupBy" and has_having:
                #     stack.push("having")
            elif isinstance(vet,tuple) and vet[0] == "op":
                if vet[1] == "where":
                    # current_sql[kw].append(index_to_column_name(vet[2], tables))
                    history[0].append(index_to_column_name(vet[2], tables))
                    hs_emb_var, hs_len = self.embed_layer.gen_x_history_batch(history)

                score = self.op.forward(q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len, np.full(B, vet[2],dtype=np.int64))

                op_num_score, op_score = [x.data.cpu().numpy() for x in score]
                op_num = np.argmax(op_num_score[0]) + 1  # num_score 0 maps to 1 in truth, must have at least one op
                ops = np.argsort(-op_score[0])[:op_num]
                # current_sql[kw].append([NEW_WHERE_OPS[op] for op in ops])
                if op_num > 0:
                    history[0].append(NEW_WHERE_OPS[ops[0]])
                    if vet[1] == "having":
                        stack.push(("root_teminal", vet[2],vet[3],ops[0]))
                    else:
                        stack.push(("root_teminal", vet[2],ops[0]))
                    # current_sql[kw].append(NEW_WHERE_OPS[ops[0]])
                for op in ops[1:]:
                    history[0].append(index_to_column_name(vet[2], tables))
                    history[0].append(NEW_WHERE_OPS[op])
                    # current_sql[kw].append(index_to_column_name(vet[2], tables))
                    # current_sql[kw].append(NEW_WHERE_OPS[op])
                    if vet[1] == "having":
                        stack.push(("root_teminal", vet[2],vet[3],op))
                    else:
                        stack.push(("root_teminal", vet[2],op))
                # stack.push(("root_teminal",vet[2]))
            elif isinstance(vet,tuple) and vet[0] == "root_teminal":
                score = self.root_teminal.forward(q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len, np.full(B, vet[1],dtype=np.int64))

                label = np.argmax(score[0].data.cpu().numpy())
                label = ROOT_TERM_OPS[label]
                if len(vet) == 4:
                    current_sql[kw].append(index_to_column_name(vet[1], tables))
                    current_sql[kw].append(vet[2])
                    current_sql[kw].append(NEW_WHERE_OPS[vet[3]])
                else:
                    # print("kw:{}".format(kw))
                    try:
                        current_sql[kw].append(index_to_column_name(vet[1], tables))
                    except Exception as e:
                        # print(e)
                        traceback.print_exc()
                        print("history:{},current_sql:{} stack:{}".format(history[0], current_sql, stack.items))
                        print("idx_stack:{}".format(idx_stack))
                        print("sql_stack:{}".format(sql_stack))
                        exit(1)
                    current_sql[kw].append(NEW_WHERE_OPS[vet[2]])
                if label == "root":
                    history[0].append("root")
                    current_sql[kw].append({})
                    # current_sql = current_sql[kw][-1]
                    stack.push(("root",current_sql[kw][-1]))
                else:
                    current_sql[kw].append("terminal")
            elif isinstance(vet,tuple) and vet[0] == "des_asc":
                current_sql[kw].append(index_to_column_name(vet[1], tables))
                current_sql[kw].append(vet[2])
                score = self.des_asc.forward(q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len, np.full(B, vet[1],dtype=np.int64))
                label = np.argmax(score[0].data.cpu().numpy())
                dec_asc,has_limit = DEC_ASC_OPS[label]
                history[0].append(dec_asc)
                current_sql[kw].append(dec_asc)
                current_sql[kw].append(has_limit)
        # print("{}".format(current_sql))

        if failed: return None
        print("history:{}".format(history[0]))
        if len(sql_stack) > 0:
            current_sql = sql_stack[0]
        # print("{}".format(current_sql))
        return current_sql

    def gen_col(self,col,table,table_alias_dict):
        colname = table["column_names_original"][col[2]][1]
        table_idx = table["column_names_original"][col[2]][0]
        if table_idx not in table_alias_dict:
            return colname
        return "T{}.{}".format(table_alias_dict[table_idx],colname)

    def gen_group_by(self,sql,kw,table,table_alias_dict):
        ret = []
        for i in range(0,len(sql)):
            # if len(sql[i+1]) == 0:
            # if sql[i+1] == "none_agg":
            ret.append(self.gen_col(sql[i],table,table_alias_dict))
            # else:
            #     ret.append("{}({})".format(sql[i+1], self.gen_col(sql[i], table, table_alias_dict)))
            # for agg in sql[i+1]:
            #     ret.append("{}({})".format(agg,gen_col(sql[i],table,table_alias_dict)))
        return "{} {}".format(kw,",".join(ret))

    def gen_select(self,sql,kw,table,table_alias_dict):
        ret = []
        for i in range(0,len(sql),2):
            # if len(sql[i+1]) == 0:
            if sql[i+1] == "none_agg" or not isinstance(sql[i+1],basestring): #DEBUG-ed 20180817
                ret.append(self.gen_col(sql[i],table,table_alias_dict))
            else:
                ret.append("{}({})".format(sql[i+1], self.gen_col(sql[i], table, table_alias_dict)))
            # for agg in sql[i+1]:
            #     ret.append("{}({})".format(agg,gen_col(sql[i],table,table_alias_dict)))
        return "{} {}".format(kw,",".join(ret))

    def gen_where(self,sql,table,table_alias_dict):
        if len(sql) == 0:
            return ""
        start_idx = 0
        andor = "and"
        if isinstance(sql[0],basestring):
            start_idx += 1
            andor = sql[0]
        ret = []
        for i in range(start_idx,len(sql),3):
            col = self.gen_col(sql[i],table,table_alias_dict)
            op = sql[i+1]
            val = sql[i+2]
            where_item = ""
            if val == "terminal":
                where_item = "{} {} '{}'".format(col,op,val)
            else:
                val = self.gen_sql(val,table)
                where_item = "{} {} ({})".format(col,op,val)
            if op == "between":
                #TODO temprarily fixed
                where_item += " and 'terminal'"
            ret.append(where_item)
        return "where {}".format(" {} ".format(andor).join(ret))

    def gen_orderby(self,sql,table,table_alias_dict):
        ret = []
        limit = ""
        if sql[-1] == True:
            limit = "limit 1"
        for i in range(0,len(sql),4):
            if sql[i+1] == "none_agg" or not isinstance(sql[i+1],basestring): #DEBUG-ed 20180817
                ret.append("{} {}".format(self.gen_col(sql[i],table,table_alias_dict), sql[i+2]))
            else:
                ret.append("{}({}) {}".format(sql[i+1], self.gen_col(sql[i], table, table_alias_dict),sql[i+2]))
        return "order by {} {}".format(",".join(ret),limit)

    def gen_having(self,sql,table,table_alias_dict):
        ret = []
        for i in range(0,len(sql),4):
            if sql[i+1] == "none_agg":
                col = self.gen_col(sql[i],table,table_alias_dict)
            else:
                col = "{}({})".format(sql[i+1], self.gen_col(sql[i], table, table_alias_dict))
            op = sql[i+2]
            val = sql[i+3]
            if val == "terminal":
                ret.append("{} {} '{}'".format(col,op,val))
            else:
                val = self.gen_sql(val, table)
                ret.append("{} {} ({})".format(col, op, val))
        return "having {}".format(",".join(ret))

    def find_shortest_path(self,start,end,graph):
        stack = [[start,[]]]
        visited = set()
        while len(stack) > 0:
            ele,history = stack.pop()
            if ele == end:
                return history
            for node in graph[ele]:
                if node[0] not in visited:
                    stack.append((node[0],history+[(node[0],node[1])]))
                    visited.add(node[0])
        print("table {} table {}".format(start,end))
        # print("could not find path!!!!!{}".format(self.path_not_found))
        self.path_not_found += 1
        # return []
    def gen_from(self,candidate_tables,table):
        def find(d,col):
            if d[col] == -1:
                return col
            return find(d,d[col])
        def union(d,c1,c2):
            r1 = find(d,c1)
            r2 = find(d,c2)
            if r1 == r2:
                return
            d[r1] = r2

        ret = ""
        if len(candidate_tables) <= 1:
            if len(candidate_tables) == 1:
                ret = "from {}".format(table["table_names_original"][list(candidate_tables)[0]])
            else:
                ret = "from {}".format(table["table_names_original"][0])
            #TODO: temporarily settings
            return {},ret
        # print("candidate:{}".format(candidate_tables))
        table_alias_dict = {}
        uf_dict = {}
        for t in candidate_tables:
            uf_dict[t] = -1
        idx = 1
        graph = defaultdict(list)
        for acol,bcol in table["foreign_keys"]:
            t1 = table["column_names"][acol][0]
            t2 = table["column_names"][bcol][0]
            graph[t1].append((t2,(acol,bcol)))
            graph[t2].append((t1,(bcol, acol)))
            # if t1 in candidate_tables and t2 in candidate_tables:
            #     r1 = find(uf_dict,t1)
            #     r2 = find(uf_dict,t2)
            #     if r1 == r2:
            #         continue
            #     union(uf_dict,t1,t2)
            #     if len(ret) == 0:
            #         ret = "from {} as T{} join {} as T{} on T{}.{}=T{}.{}".format(table["table_names"][t1],idx,table["table_names"][t2],
            #                                                                       idx+1,idx,table["column_names_original"][acol][1],idx+1,
            #                                                                       table["column_names_original"][bcol][1])
            #         table_alias_dict[t1] = idx
            #         table_alias_dict[t2] = idx+1
            #         idx += 2
            #     else:
            #         if t1 in table_alias_dict:
            #             old_t = t1
            #             new_t = t2
            #             acol,bcol = bcol,acol
            #         elif t2 in table_alias_dict:
            #             old_t = t2
            #             new_t = t1
            #         else:
            #             ret = "{} join {} as T{} join {} as T{} on T{}.{}=T{}.{}".format(ret,table["table_names"][t1], idx,
            #                                                                           table["table_names"][t2],
            #                                                                           idx + 1, idx,
            #                                                                           table["column_names_original"][acol][1],
            #                                                                           idx + 1,
            #                                                                           table["column_names_original"][bcol][1])
            #             table_alias_dict[t1] = idx
            #             table_alias_dict[t2] = idx + 1
            #             idx += 2
            #             continue
            #         ret = "{} join {} as T{} on T{}.{}=T{}.{}".format(ret,new_t,idx,idx,table["column_names_original"][acol][1],
            #                                                        table_alias_dict[old_t],table["column_names_original"][bcol][1])
            #         table_alias_dict[new_t] = idx
            #         idx += 1
        # visited = set()
        candidate_tables = list(candidate_tables)
        start = candidate_tables[0]
        table_alias_dict[start] = idx
        idx += 1
        ret = "from {} as T1".format(table["table_names_original"][start])
        try:
            for end in candidate_tables[1:]:
                if end in table_alias_dict:
                    continue
                path = self.find_shortest_path(start, end, graph)
                prev_table = start
                if not path:
                    table_alias_dict[end] = idx
                    idx += 1
                    ret = "{} join {} as T{}".format(ret, table["table_names_original"][end],
                                                                      table_alias_dict[end],
                                                                      )
                    continue
                for node, (acol, bcol) in path:
                    if node in table_alias_dict:
                        prev_table = node
                        continue
                    table_alias_dict[node] = idx
                    idx += 1
                    ret = "{} join {} as T{} on T{}.{} = T{}.{}".format(ret, table["table_names_original"][node],
                                                                      table_alias_dict[node],
                                                                      table_alias_dict[prev_table],
                                                                      table["column_names_original"][acol][1],
                                                                      table_alias_dict[node],
                                                                      table["column_names_original"][bcol][1])
                    prev_table = node
        except:
            traceback.print_exc()
            print("db:{}".format(table["db_id"]))
            # print(table["db_id"])
            return table_alias_dict,ret
        # if len(candidate_tables) != len(table_alias_dict):
        #     print("error in generate from clause!!!!!")
        return table_alias_dict,ret

    def gen_sql(self, sql,table):
        select_clause = ""
        from_clause = ""
        groupby_clause = ""
        orderby_clause = ""
        having_clause = ""
        where_clause = ""
        nested_clause = ""
        cols = {}
        candidate_tables = set()
        nested_sql = {}
        nested_label = ""
        parent_sql = sql
        # if "sql" in sql:
        #     sql = sql["sql"]
        if "nested_label" in sql:
            nested_label = sql["nested_label"]
            nested_sql = sql["nested_sql"]
            sql = sql["sql"]
        elif "sql" in sql:
            sql = sql["sql"]
        for key in sql:
            if key not in KW_WITH_COL:
                continue
            for item in sql[key]:
                if isinstance(item,tuple) and len(item) == 3:
                    if table["column_names"][item[2]][0] != -1:
                        candidate_tables.add(table["column_names"][item[2]][0])
        table_alias_dict,from_clause = self.gen_from(candidate_tables,table)
        ret = []
        if "select" in sql:
            select_clause = self.gen_select(sql["select"],"select",table,table_alias_dict)
            if len(select_clause) > 0:
                ret.append(select_clause)
            else:
                print("select not found:{}".format(parent_sql))
        else:
            print("select not found:{}".format(parent_sql))
        if len(from_clause) > 0:
            ret.append(from_clause)
        if "where" in sql:
            where_clause = self.gen_where(sql["where"],table,table_alias_dict)
            if len(where_clause) > 0:
                ret.append(where_clause)
        if "groupBy" in sql: ## DEBUG-ed order
            groupby_clause = self.gen_group_by(sql["groupBy"],"group by",table,table_alias_dict)
            if len(groupby_clause) > 0:
                ret.append(groupby_clause)
        if "orderBy" in sql:
            orderby_clause = self.gen_orderby(sql["orderBy"],table,table_alias_dict)
            if len(orderby_clause) > 0:
                ret.append(orderby_clause)
        if "having" in sql:
            having_clause = self.gen_having(sql["having"],table,table_alias_dict)
            if len(having_clause) > 0:
                ret.append(having_clause)
        if len(nested_label) > 0:
            nested_clause = "{} {}".format(nested_label,self.gen_sql(nested_sql,table))
            if len(nested_clause) > 0:
                ret.append(nested_clause)
        return " ".join(ret)

    def check_acc(self, pred_sql, gt_sql):
        pass
