import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import copy

BOOL_WD_DROP=False

class WordEmbedding(nn.Module):
    def __init__(self, word_emb, N_word, gpu, SQL_TOK,
            our_model, trainable=False):
        super(WordEmbedding, self).__init__()
        self.trainable = trainable
        self.N_word = N_word
        self.our_model = our_model
        self.gpu = gpu
        self.SQL_TOK = SQL_TOK

        if trainable:
            print "Using trainable embedding"
            self.w2i, word_emb_val = word_emb
            self.embedding = nn.Embedding(len(self.w2i), N_word)
            self.embedding.weight = nn.Parameter(
                    torch.from_numpy(word_emb_val.astype(np.float32)))
        else:
            self.word_emb = word_emb
            print "Using fixed embedding"

        if self.gpu:
            self.new_long_tensor = torch.cuda.LongTensor
            self.new_tensor = torch.cuda.FloatTensor
        else:
            self.new_long_tensor = torch.LongTensor
            self.new_tensor = torch.FloatTensor

    def gen_x_batch(self, q, col, dropout_rate=0.0):
        B = len(q)
        val_embs = []
        val_len = np.zeros(B, dtype=np.int64)
        for i, (one_q, one_col) in enumerate(zip(q, col)):
            if self.trainable:
                q_val = map(lambda x:self.w2i.get(x, 0), one_q)
            else:
                q_val = map(lambda x:self.word_emb.get(x, np.zeros(self.N_word, dtype=np.float32)), one_q)
            if self.our_model:
                if self.trainable:
                    val_embs.append([1] + q_val + [2])  #<BEG> and <END>
                else:
                    val_embs.append([np.zeros(self.N_word, dtype=np.float32)] + q_val + [np.zeros(self.N_word, dtype=np.float32)])  #<BEG> and <END>
                val_len[i] = 1 + len(q_val) + 1
            else:
                one_col_all = [x for toks in one_col for x in toks+[',']]
                if self.trainable:
                    col_val = map(lambda x:self.w2i.get(x, 0), one_col_all)
                    val_embs.append( [0 for _ in self.SQL_TOK] + col_val + [0] + q_val+ [0])
                else:
                    col_val = map(lambda x:self.word_emb.get(x, np.zeros(self.N_word, dtype=np.float32)), one_col_all)
                    val_embs.append( [np.zeros(self.N_word, dtype=np.float32) for _ in self.SQL_TOK] + col_val +
                                     [np.zeros(self.N_word, dtype=np.float32)] + q_val +
                                     [np.zeros(self.N_word, dtype=np.float32)])
                val_len[i] = len(self.SQL_TOK) + len(col_val) + 1 + len(q_val) + 1
        max_len = max(val_len)

        if self.trainable:
            val_tok_array = np.zeros((B, max_len), dtype=np.int64)
            for i in range(B):
                for t in range(len(val_embs[i])):
                    val_tok_array[i,t] = val_embs[i][t]
            val_tok = torch.from_numpy(val_tok_array)
            if self.gpu:
                val_tok = val_tok.cuda()
            val_tok_var = Variable(val_tok)

            # val_inp_var = self.embedding(val_tok_var)
            if dropout_rate > 0.0 and BOOL_WD_DROP:
                tmp_embedding = copy.deepcopy(self.embedding)
                emb_mask = self.new_tensor(tmp_embedding.weight.size()).fill_(1. - dropout_rate).bernoulli()\
                    .div_(1. - dropout_rate)
                tmp_embedding.weight = nn.Parameter(tmp_embedding.weight.data * emb_mask)
            else:
                tmp_embedding = self.embedding
            val_inp_var = tmp_embedding(val_tok_var)

        else:
            val_emb_array = np.zeros((B, max_len, self.N_word), dtype=np.float32)
            for i in range(B):
                for t in range(len(val_embs[i])):
                    val_emb_array[i,t,:] = val_embs[i][t]
            val_inp = torch.from_numpy(val_emb_array)
            if self.gpu:
                val_inp = val_inp.cuda()
            val_inp_var = Variable(val_inp)
        return val_inp_var, val_len

    def gen_col_batch(self, cols, dropout_rate=0.):
        ret = []
        col_len = np.zeros(len(cols), dtype=np.int64)

        names = []
        for b, one_cols in enumerate(cols):
            names = names + one_cols
            col_len[b] = len(one_cols)

        name_inp_var, name_len = self.str_list_to_batch(names, dropout_rate=dropout_rate)
        return name_inp_var, name_len, col_len

    def str_list_to_batch(self, str_list, dropout_rate=0.):
        B = len(str_list)

        val_embs = []
        val_len = np.zeros(B, dtype=np.int64)
        for i, one_str in enumerate(str_list):
            if self.trainable:
                val = [self.w2i.get(x, 0) for x in one_str]
            else:
                val = [self.word_emb.get(x, np.zeros(
                    self.N_word, dtype=np.float32)) for x in one_str]
            val_embs.append(val)
            val_len[i] = len(val)
        max_len = max(val_len)

        if self.trainable:
            val_tok_array = np.zeros((B, max_len), dtype=np.int64)
            for i in range(B):
                for t in range(len(val_embs[i])):
                    val_tok_array[i,t] = val_embs[i][t]
            val_tok = torch.from_numpy(val_tok_array)
            if self.gpu:
                val_tok = val_tok.cuda()
            val_tok_var = Variable(val_tok)

            # val_inp_var = self.embedding(val_tok_var)
            if dropout_rate > 0.0 and BOOL_WD_DROP:
                tmp_embedding = copy.deepcopy(self.embedding)
                emb_mask = self.new_tensor(tmp_embedding.weight.size()).fill_(1. - dropout_rate).bernoulli()\
                    .div_(1. - dropout_rate)
                tmp_embedding.weight = nn.Parameter(tmp_embedding.weight.data * emb_mask)
            else:
                tmp_embedding = self.embedding
            val_inp_var = tmp_embedding(val_tok_var)

        else:
            val_emb_array = np.zeros(
                    (B, max_len, self.N_word), dtype=np.float32)
            for i in range(B):
                for t in range(len(val_embs[i])):
                    val_emb_array[i,t,:] = val_embs[i][t]
            val_inp = torch.from_numpy(val_emb_array)
            if self.gpu:
                val_inp = val_inp.cuda()
            val_inp_var = Variable(val_inp)

        return val_inp_var, val_len
