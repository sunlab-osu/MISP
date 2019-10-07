# Adapted to support beam search
# @author: Ziyu Yao

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from net_utils import run_lstm, col_name_encode
from beam_search_utils import *


class SQLNetCondPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth, max_col_num, max_tok_num, use_ca, gpu,
                 dr=0.3, temperature=False):
        super(SQLNetCondPredictor, self).__init__()
        self.N_h = N_h
        self.max_tok_num = max_tok_num
        self.max_col_num = max_col_num
        self.gpu = gpu
        self.use_ca = use_ca

        self.cond_num_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=dr, bidirectional=True)
        self.cond_num_att = nn.Linear(N_h, 1)
        self.cond_num_out = nn.Sequential(nn.Linear(N_h, N_h),
                nn.Tanh(), nn.Linear(N_h, 5))
        self.cond_num_name_enc = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=dr, bidirectional=True)
        self.cond_num_col_att = nn.Linear(N_h, 1)
        self.cond_num_col2hid1 = nn.Linear(N_h, 2*N_h)
        self.cond_num_col2hid2 = nn.Linear(N_h, 2*N_h)

        self.cond_col_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=dr, bidirectional=True)
        if use_ca:
            print "Using column attention on where predicting"
            self.cond_col_att = nn.Linear(N_h, N_h)
        else:
            print "Not using column attention on where predicting"
            self.cond_col_att = nn.Linear(N_h, 1)
        self.cond_col_name_enc = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=dr, bidirectional=True)
        self.cond_col_out_K = nn.Linear(N_h, N_h)
        self.cond_col_out_col = nn.Linear(N_h, N_h)
        self.cond_col_out = nn.Sequential(nn.ReLU(), nn.Linear(N_h, 1))

        self.cond_op_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=dr, bidirectional=True)
        if use_ca:
            self.cond_op_att = nn.Linear(N_h, N_h)
        else:
            self.cond_op_att = nn.Linear(N_h, 1)
        self.cond_op_out_K = nn.Linear(N_h, N_h)
        self.cond_op_name_enc = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=dr, bidirectional=True)
        self.cond_op_out_col = nn.Linear(N_h, N_h)
        self.cond_op_out = nn.Sequential(nn.Linear(N_h, N_h), nn.Tanh(),
                nn.Linear(N_h, 3))

        self.cond_str_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=dr, bidirectional=True)
        self.cond_str_decoder = nn.LSTM(input_size=self.max_tok_num,
                hidden_size=N_h, num_layers=N_depth,
                batch_first=True, dropout=dr)
        self.cond_str_name_enc = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=dr, bidirectional=True)
        self.cond_str_out_g = nn.Linear(N_h, N_h)
        self.cond_str_out_h = nn.Linear(N_h, N_h)
        self.cond_str_out_col = nn.Linear(N_h, N_h)
        self.cond_str_out = nn.Sequential(nn.ReLU(), nn.Linear(N_h, 1))

        self.softmax = nn.Softmax()

        if temperature:
            self.T1 = nn.Parameter(torch.FloatTensor([1.])) #cond_num_out
            self.T2 = nn.Parameter(torch.FloatTensor([1.])) #cond_col_out
            self.T3 = nn.Parameter(torch.FloatTensor([1.])) #cond_op_out
            self.T4 = nn.Parameter(torch.FloatTensor([1.])) #cond_str_out
            # self.T4 = 1. # this will make beam search results change
        else:
            self.T1 = self.T2 = self.T3 = self.T4 = 1.

    def gen_gt_batch(self, split_tok_seq):
        B = len(split_tok_seq)
        max_len = max([max([len(tok) for tok in tok_seq]+[0]) for 
            tok_seq in split_tok_seq]) - 1 # The max seq len in the batch.
        if max_len < 1:
            max_len = 1
        ret_array = np.zeros((
            B, 4, max_len, self.max_tok_num), dtype=np.float32)
        ret_len = np.zeros((B, 4))
        for b, tok_seq in enumerate(split_tok_seq):
            idx = 0
            for idx, one_tok_seq in enumerate(tok_seq):
                out_one_tok_seq = one_tok_seq[:-1]
                ret_len[b, idx] = len(out_one_tok_seq)
                for t, tok_id in enumerate(out_one_tok_seq):
                    ret_array[b, idx, t, tok_id] = 1
            if idx < 3:
                ret_array[b, idx+1:, 0, 1] = 1
                ret_len[b, idx+1:] = 1

        ret_inp = torch.from_numpy(ret_array)
        if self.gpu:
            ret_inp = ret_inp.cuda()
        ret_inp_var = Variable(ret_inp)

        return ret_inp_var, ret_len #[B, IDX, max_len, max_tok_num]

    def cols_forward(self, x_emb_var, x_len, col_inp_var, col_name_len, col_len, dropout_rate=0.):
        B = len(x_len)
        max_x_len = max(x_len)

        # Predict the number of conditions
        # First use column embeddings to calculate the initial hidden unit
        # Then run the LSTM and predict condition number.
        e_num_col, col_num = col_name_encode(col_inp_var, col_name_len,
                                             col_len, self.cond_num_name_enc, dropout_rate=dropout_rate)

        num_col_att_val = self.cond_num_col_att(e_num_col).squeeze(2)
        for idx, num in enumerate(col_num):
            if num < max(col_num):
                num_col_att_val[idx, num:] = -100
        num_col_att = self.softmax(num_col_att_val)
        K_num_col = (e_num_col * num_col_att.unsqueeze(2)).sum(1)
        cond_num_h1 = self.cond_num_col2hid1(K_num_col).view(
            B, 4, self.N_h / 2).transpose(0, 1).contiguous()
        cond_num_h2 = self.cond_num_col2hid2(K_num_col).view(
            B, 4, self.N_h / 2).transpose(0, 1).contiguous()

        h_num_enc, _ = run_lstm(self.cond_num_lstm, x_emb_var, x_len,
                                hidden=(cond_num_h1, cond_num_h2), dropout_rate=dropout_rate)

        num_att_val = self.cond_num_att(h_num_enc).squeeze(2)

        for idx, num in enumerate(x_len):
            if num < max_x_len:
                num_att_val[idx, num:] = -100
        num_att = self.softmax(num_att_val)

        K_cond_num = (h_num_enc * num_att.unsqueeze(2).expand_as(
            h_num_enc)).sum(1)

        if dropout_rate > 0.:
            K_cond_num_mask = torch.FloatTensor(K_cond_num.size()[-1]).view(1, -1)\
                .fill_(1. - dropout_rate).bernoulli().div_(1. - dropout_rate)
            if K_cond_num.is_cuda:
                K_cond_num_mask = K_cond_num_mask.cuda()
            K_cond_num.data = K_cond_num.data * K_cond_num_mask
        cond_num_score = self.cond_num_out(K_cond_num) / self.T1

        # Predict the columns of conditions
        e_cond_col, _ = col_name_encode(col_inp_var, col_name_len, col_len,
                                        self.cond_col_name_enc, dropout_rate=dropout_rate)

        h_col_enc, _ = run_lstm(self.cond_col_lstm, x_emb_var, x_len, dropout_rate=dropout_rate)
        if self.use_ca:
            col_att_val = torch.bmm(e_cond_col,
                                    self.cond_col_att(h_col_enc).transpose(1, 2))
            for idx, num in enumerate(x_len):
                if num < max_x_len:
                    col_att_val[idx, :, num:] = -100
            col_att = self.softmax(col_att_val.view(
                (-1, max_x_len))).view(B, -1, max_x_len)
            K_cond_col = (h_col_enc.unsqueeze(1) * col_att.unsqueeze(3)).sum(2)
        else:
            col_att_val = self.cond_col_att(h_col_enc).squeeze()
            for idx, num in enumerate(x_len):
                if num < max_x_len:
                    col_att_val[idx, num:] = -100
            col_att = self.softmax(col_att_val)
            K_cond_col = (h_col_enc *
                          col_att_val.unsqueeze(2)).sum(1).unsqueeze(1)

        if dropout_rate > 0.:
            # K_cond_col: [batch_size, col_size, hid_size]
            K_cond_col_mask = torch.FloatTensor(K_cond_col.size()[-1]).view(1, 1, -1)\
                .fill_(1. - dropout_rate).bernoulli().div_(1. - dropout_rate)
            if K_cond_col.is_cuda:
                K_cond_col_mask = K_cond_col_mask.cuda()
            K_cond_col.data = K_cond_col.data * K_cond_col_mask

        cond_col_score = self.cond_col_out(self.cond_col_out_K(K_cond_col) +
                                           self.cond_col_out_col(e_cond_col)).squeeze(2) / self.T2
        max_col_num = max(col_num)
        for b, num in enumerate(col_num):
            if num < max_col_num:
                cond_col_score[b, num:] = -100

        return cond_num_score, cond_col_score

    def op_forward(self, x_emb_var, x_len, col_inp_var, col_name_len,
            col_len, chosen_col_gt, dropout_rate=0.):
        B = len(x_len)
        max_x_len = max(x_len)
        e_cond_col, _ = col_name_encode(col_inp_var, col_name_len,
                                        col_len, self.cond_op_name_enc, dropout_rate=dropout_rate)
        col_emb = []
        for b in range(B):
            cur_col_emb = torch.stack([e_cond_col[b, x] for x in chosen_col_gt[b]] +
                [e_cond_col[b, 0]] * (4 - len(chosen_col_gt[b])))  # Pad the columns to maximum (4)
            col_emb.append(cur_col_emb)
        col_emb = torch.stack(col_emb)

        h_op_enc, _ = run_lstm(self.cond_op_lstm, x_emb_var, x_len, dropout_rate=dropout_rate)
        if self.use_ca:
            op_att_val = torch.matmul(self.cond_op_att(h_op_enc).unsqueeze(1),
                                      col_emb.unsqueeze(3)).squeeze(-1)
            for idx, num in enumerate(x_len):
                if num < max_x_len:
                    op_att_val[idx, :, num:] = -100
            op_att = self.softmax(op_att_val.view(B * 4, -1)).view(B, 4, -1)
            K_cond_op = (h_op_enc.unsqueeze(1) * op_att.unsqueeze(3)).sum(2)
        else:
            op_att_val = self.cond_op_att(h_op_enc).squeeze()
            for idx, num in enumerate(x_len):
                if num < max_x_len:
                    op_att_val[idx, num:] = -100
            op_att = self.softmax(op_att_val)
            K_cond_op = (h_op_enc * op_att.unsqueeze(2)).sum(1).unsqueeze(1)

        if dropout_rate > 0.:
            # K_cond_op: [batch_size, op_size, hid_size]
            K_cond_op_mask = torch.FloatTensor(K_cond_op.size()[-1]).view(1, 1, -1)\
                .fill_(1. - dropout_rate).bernoulli().div_(1. - dropout_rate)
            if K_cond_op.is_cuda:
                K_cond_op_mask = K_cond_op_mask.cuda()
            K_cond_op.data = K_cond_op.data * K_cond_op_mask

        cond_op_score = self.cond_op_out(self.cond_op_out_K(K_cond_op) +
                                         self.cond_op_out_col(col_emb)).squeeze(-1) / self.T3

        return cond_op_score

    def val_beam_search(self, x_emb_var, x_len, col_inp_var, col_name_len,
                        col_len, chosen_col_gt, beam_size,
                        avoid_idxes_list=None, given_idxes=None, dropout_rate=0.):
        B = len(x_len)
        assert B == 1, "Beam search works for one example per batch."
        max_x_len = max(x_len)

        if beam_size is None:
            true_beam_size = beam_size = 100 # inf option
        else:
            true_beam_size = beam_size
            if avoid_idxes_list is not None:
                beam_size = true_beam_size + len(avoid_idxes_list)
            # beam_size = beam_size * 2 # since we will have to remove avoid_idxes

        h_str_enc, _ = run_lstm(self.cond_str_lstm, x_emb_var, x_len, dropout_rate=dropout_rate)
        e_cond_col, _ = col_name_encode(col_inp_var, col_name_len,
                                        col_len, self.cond_str_name_enc, dropout_rate=dropout_rate)
        col_emb = []
        for b in range(B):
            cur_col_emb = torch.stack([e_cond_col[b, x] for x in chosen_col_gt[b]])
            col_emb.append(cur_col_emb)
        col_emb = torch.stack(col_emb)

        h_ext = h_str_enc.unsqueeze(1).unsqueeze(1)
        col_ext = col_emb.unsqueeze(2).unsqueeze(2)

        initial_beam = BeamHyp(logprob=0., str_idxes=[0], state=None)
        partial_generations = TopN(beam_size)
        partial_generations.push(initial_beam)
        complete_generations = TopN(beam_size)

        t = 0
        while t < 50:
            partial_generations_list = partial_generations.extract(sort=True)[:beam_size]
            partial_generations.reset()
            cur_B = len(partial_generations_list)

            # inputs
            cur_inp = np.zeros((cur_B, 1, self.max_tok_num), dtype=np.float32)
            for cur_b in range(cur_B):
                cur_inp[cur_b, 0, partial_generations_list[cur_b].str_idxes[-1]] = 1
            if self.gpu:
                cur_inp = Variable(torch.from_numpy(cur_inp).cuda())
            else:
                cur_inp = Variable(torch.from_numpy(cur_inp))

            # states
            if t == 0:
                g_str_s_flat, (cur_hs, cur_cs) = self.cond_str_decoder(cur_inp)
            else:
                state_hs = torch.stack([partial_generations_list[cur_b].state[0]
                                        for cur_b in range(cur_B)], dim=1)
                state_cs = torch.stack([partial_generations_list[cur_b].state[1]
                                        for cur_b in range(cur_B)], dim=1)
                cur_h = (state_hs, state_cs)
                g_str_s_flat, (cur_hs, cur_cs) = self.cond_str_decoder(cur_inp, cur_h)

            if dropout_rate > 0.:
                # g_str_s_flat: [batch_size, 4->1, hid_size]
                g_str_s_flat_mask = torch.FloatTensor(g_str_s_flat.size()[-1]).view(1, 1, -1) \
                    .fill_(1. - dropout_rate).bernoulli().div_(1. - dropout_rate)
                if g_str_s_flat.is_cuda:
                    g_str_s_flat_mask = g_str_s_flat_mask.cuda()
                g_str_s_flat.data = g_str_s_flat.data * g_str_s_flat_mask

            g_str_s = g_str_s_flat.view(cur_B, 1, 1, self.N_h) #[cur_B, 4->1, 1, self.N_h]
            g_ext = g_str_s.unsqueeze(3)

            cur_cond_str_score = self.cond_str_out(
                self.cond_str_out_h(h_ext) + self.cond_str_out_g(g_ext)
                + self.cond_str_out_col(col_ext)).squeeze(-1) / self.T4 #[cur_B, 4->1, |wd cands|]
            for cur_b, num in enumerate(x_len):
                if num < max_x_len:
                    cur_cond_str_score[cur_b, :, num:] = -100
            cur_cond_str_score = cur_cond_str_score.view(cur_B, max_x_len)
            cur_prob_scores = self.softmax(cur_cond_str_score)

            for cur_b in range(cur_B):
                prob_score = cur_prob_scores[cur_b]

                if given_idxes is not None:
                    chosed_idx = given_idxes[t + 1]
                    chosed_prob = prob_score.data.cpu().numpy()[chosed_idx]
                    top_probs = [chosed_prob]
                    top_idxes = [chosed_idx]
                else:
                    top_probs, top_idxes = prob_score.topk(min(beam_size, max_x_len))
                    top_probs = top_probs.data.cpu().numpy()
                    top_idxes = top_idxes.data.cpu().numpy()

                for prob, idx in zip(top_probs, top_idxes):
                    str_idxes = partial_generations_list[cur_b].str_idxes + [idx]
                    logprob = partial_generations_list[cur_b].logprob + np.log(prob)
                    state_h = cur_hs[:,cur_b,:]
                    state_c = cur_cs[:,cur_b,:]
                    state = (state_h, state_c)
                    tmp_beam = BeamHyp(logprob, str_idxes, state)

                    if idx == x_len[0] - 1: #<END>
                        complete_generations.push(tmp_beam)
                    else:
                        partial_generations.push(tmp_beam)

            if partial_generations.size() == 0: #complete_generations.size() == beam_size
                break
            t += 1

        complete_generations_list = complete_generations.extract(sort=True)
        if len(complete_generations_list) == 0:
            print("## WARNING: beam search with no <EOS>...")
            complete_generations_list = partial_generations.extract(sort=True)

        results = []
        for gen in complete_generations_list:
            str_idxes = gen.str_idxes
            if avoid_idxes_list is not None and tuple(str_idxes) in avoid_idxes_list:
                continue

            logprob = gen.logprob
            results.append((str_idxes, logprob))

        return results[:true_beam_size]

    def forward(self, x_emb_var, x_len, col_inp_var, col_name_len,
            col_len, col_num, gt_where, gt_cond, reinforce):
        max_x_len = max(x_len)
        B = len(x_len)
        if reinforce:
            raise NotImplementedError('Our model doesn\'t have RL')

        # Predict the number of conditions
        # First use column embeddings to calculate the initial hidden unit
        # Then run the LSTM and predict condition number.
        e_num_col, col_num = col_name_encode(col_inp_var, col_name_len,
                col_len, self.cond_num_name_enc)
        num_col_att_val = self.cond_num_col_att(e_num_col).squeeze()
        for idx, num in enumerate(col_num):
            if num < max(col_num):
                num_col_att_val[idx, num:] = -100
        num_col_att = self.softmax(num_col_att_val)
        K_num_col = (e_num_col * num_col_att.unsqueeze(2)).sum(1)
        cond_num_h1 = self.cond_num_col2hid1(K_num_col).view(
                B, 4, self.N_h/2).transpose(0, 1).contiguous()
        cond_num_h2 = self.cond_num_col2hid2(K_num_col).view(
                B, 4, self.N_h/2).transpose(0, 1).contiguous()

        h_num_enc, _ = run_lstm(self.cond_num_lstm, x_emb_var, x_len,
                hidden=(cond_num_h1, cond_num_h2))

        num_att_val = self.cond_num_att(h_num_enc).squeeze()

        for idx, num in enumerate(x_len):
            if num < max_x_len:
                num_att_val[idx, num:] = -100
        num_att = self.softmax(num_att_val)

        K_cond_num = (h_num_enc * num_att.unsqueeze(2).expand_as(
            h_num_enc)).sum(1)
        cond_num_score = self.cond_num_out(K_cond_num) / self.T1

        #Predict the columns of conditions
        e_cond_col, _ = col_name_encode(col_inp_var, col_name_len, col_len,
                self.cond_col_name_enc)

        h_col_enc, _ = run_lstm(self.cond_col_lstm, x_emb_var, x_len)
        if self.use_ca:
            col_att_val = torch.bmm(e_cond_col,
                    self.cond_col_att(h_col_enc).transpose(1, 2))
            for idx, num in enumerate(x_len):
                if num < max_x_len:
                    col_att_val[idx, :, num:] = -100
            col_att = self.softmax(col_att_val.view(
                (-1, max_x_len))).view(B, -1, max_x_len)
            K_cond_col = (h_col_enc.unsqueeze(1) * col_att.unsqueeze(3)).sum(2)
        else:
            col_att_val = self.cond_col_att(h_col_enc).squeeze()
            for idx, num in enumerate(x_len):
                if num < max_x_len:
                    col_att_val[idx, num:] = -100
            col_att = self.softmax(col_att_val)
            K_cond_col = (h_col_enc *
                    col_att_val.unsqueeze(2)).sum(1).unsqueeze(1)

        cond_col_score = self.cond_col_out(self.cond_col_out_K(K_cond_col) +
                self.cond_col_out_col(e_cond_col)).squeeze() / self.T2
        max_col_num = max(col_num)
        for b, num in enumerate(col_num):
            if num < max_col_num:
                cond_col_score[b, num:] = -100

        #Predict the operator of conditions
        chosen_col_gt = []
        if gt_cond is None:
            cond_nums = np.argmax(cond_num_score.data.cpu().numpy(), axis=1)
            col_scores = cond_col_score.data.cpu().numpy()
            chosen_col_gt = [list(np.argsort(-col_scores[b])[:cond_nums[b]])
                    for b in range(len(cond_nums))]
        else:
            chosen_col_gt = [ [x[0] for x in one_gt_cond] for 
                    one_gt_cond in gt_cond]

        e_cond_col, _ = col_name_encode(col_inp_var, col_name_len,
                col_len, self.cond_op_name_enc)
        col_emb = []
        for b in range(B):
            cur_col_emb = torch.stack([e_cond_col[b, x] 
                for x in chosen_col_gt[b]] + [e_cond_col[b, 0]] *
                (4 - len(chosen_col_gt[b])))  # Pad the columns to maximum (4)
            col_emb.append(cur_col_emb)
        col_emb = torch.stack(col_emb)

        h_op_enc, _ = run_lstm(self.cond_op_lstm, x_emb_var, x_len)
        if self.use_ca:
            op_att_val = torch.matmul(self.cond_op_att(h_op_enc).unsqueeze(1),
                    col_emb.unsqueeze(3)).squeeze()
            for idx, num in enumerate(x_len):
                if num < max_x_len:
                    op_att_val[idx, :, num:] = -100
            op_att = self.softmax(op_att_val.view(B*4, -1)).view(B, 4, -1)
            K_cond_op = (h_op_enc.unsqueeze(1) * op_att.unsqueeze(3)).sum(2)
        else:
            op_att_val = self.cond_op_att(h_op_enc).squeeze()
            for idx, num in enumerate(x_len):
                if num < max_x_len:
                    op_att_val[idx, num:] = -100
            op_att = self.softmax(op_att_val)
            K_cond_op = (h_op_enc * op_att.unsqueeze(2)).sum(1).unsqueeze(1)

        cond_op_score = self.cond_op_out(self.cond_op_out_K(K_cond_op) +
                self.cond_op_out_col(col_emb)).squeeze() / self.T3

        #Predict the string of conditions
        h_str_enc, _ = run_lstm(self.cond_str_lstm, x_emb_var, x_len)
        e_cond_col, _ = col_name_encode(col_inp_var, col_name_len,
                col_len, self.cond_str_name_enc)
        col_emb = []
        for b in range(B):
            cur_col_emb = torch.stack([e_cond_col[b, x]
                for x in chosen_col_gt[b]] +
                [e_cond_col[b, 0]] * (4 - len(chosen_col_gt[b])))
            col_emb.append(cur_col_emb)
        col_emb = torch.stack(col_emb)

        if gt_where is not None:
            gt_tok_seq, gt_tok_len = self.gen_gt_batch(gt_where)
            g_str_s_flat, _ = self.cond_str_decoder(
                    gt_tok_seq.view(B*4, -1, self.max_tok_num))
            g_str_s = g_str_s_flat.contiguous().view(B, 4, -1, self.N_h)

            h_ext = h_str_enc.unsqueeze(1).unsqueeze(1)
            g_ext = g_str_s.unsqueeze(3)
            col_ext = col_emb.unsqueeze(2).unsqueeze(2)

            cond_str_score = self.cond_str_out(
                    self.cond_str_out_h(h_ext) + self.cond_str_out_g(g_ext) +
                    self.cond_str_out_col(col_ext)).squeeze() / self.T4
            for b, num in enumerate(x_len):
                if num < max_x_len:
                    cond_str_score[b, :, :, num:] = -100
        else:
            h_ext = h_str_enc.unsqueeze(1).unsqueeze(1)
            col_ext = col_emb.unsqueeze(2).unsqueeze(2)
            scores = []

            t = 0
            init_inp = np.zeros((B*4, 1, self.max_tok_num), dtype=np.float32)
            init_inp[:,0,0] = 1  #Set the <BEG> token
            if self.gpu:
                cur_inp = Variable(torch.from_numpy(init_inp).cuda())
            else:
                cur_inp = Variable(torch.from_numpy(init_inp))
            cur_h = None
            while t < 50:
                if cur_h:
                    g_str_s_flat, cur_h = self.cond_str_decoder(cur_inp, cur_h)
                else:
                    g_str_s_flat, cur_h = self.cond_str_decoder(cur_inp)
                g_str_s = g_str_s_flat.view(B, 4, 1, self.N_h)
                g_ext = g_str_s.unsqueeze(3)

                cur_cond_str_score = self.cond_str_out(
                        self.cond_str_out_h(h_ext) + self.cond_str_out_g(g_ext)
                        + self.cond_str_out_col(col_ext)).squeeze()
                for b, num in enumerate(x_len):
                    if num < max_x_len:
                        cur_cond_str_score[b, :, num:] = -100
                scores.append(cur_cond_str_score)

                _, ans_tok_var = cur_cond_str_score.view(B*4, max_x_len).max(1)
                ans_tok = ans_tok_var.data.cpu()
                data = torch.zeros(B*4, self.max_tok_num).scatter_(
                        1, ans_tok.unsqueeze(1), 1)
                if self.gpu:  #To one-hot
                    cur_inp = Variable(data.cuda())
                else:
                    cur_inp = Variable(data)
                cur_inp = cur_inp.unsqueeze(1)

                t += 1

            cond_str_score = torch.stack(scores, 2)
            for b, num in enumerate(x_len):
                if num < max_x_len:
                    cond_str_score[b, :, :, num:] = -100  #[B, IDX, T, TOK_NUM]

        cond_score = (cond_num_score,
                cond_col_score, cond_op_score, cond_str_score)

        return cond_score
