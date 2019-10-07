import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from net_utils import run_lstm, col_name_encode


class AggPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth, use_ca, dr=0.3, temperature=False):
        super(AggPredictor, self).__init__()
        self.use_ca = use_ca

        self.agg_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=dr, bidirectional=True)
        if use_ca:
            print "Using column attention on aggregator predicting"
            self.agg_col_name_enc = nn.LSTM(input_size=N_word,
                    hidden_size=N_h/2, num_layers=N_depth,
                    batch_first=True, dropout=dr, bidirectional=True)
            self.agg_att = nn.Linear(N_h, N_h)
        else:
            print "Not using column attention on aggregator predicting"
            self.agg_att = nn.Linear(N_h, 1)
        self.agg_out = nn.Sequential(nn.Linear(N_h, N_h),
                nn.Tanh(), nn.Linear(N_h, 6))
        self.softmax = nn.Softmax()

        if temperature:
            self.T = nn.Parameter(torch.FloatTensor([1.]))
        else:
            self.T = 1.

    def forward(self, x_emb_var, x_len, col_inp_var=None, col_name_len=None,
            col_len=None, col_num=None, gt_sel=None, dropout_rate=0.):
        B = len(x_emb_var)
        max_x_len = max(x_len)

        h_enc, _ = run_lstm(self.agg_lstm, x_emb_var, x_len, dropout_rate=dropout_rate)
        if self.use_ca:
            e_col, _ = col_name_encode(col_inp_var, col_name_len, 
                    col_len, self.agg_col_name_enc, dropout_rate=dropout_rate)
            chosen_sel_idx = torch.LongTensor(gt_sel)
            aux_range = torch.LongTensor(range(len(gt_sel)))
            if x_emb_var.is_cuda:
                chosen_sel_idx = chosen_sel_idx.cuda()
                aux_range = aux_range.cuda()
            chosen_e_col = e_col[aux_range, chosen_sel_idx]
            att_val = torch.bmm(self.agg_att(h_enc), 
                    chosen_e_col.unsqueeze(2)).squeeze(-1) #squeeze dim=-1
        else:
            att_val = self.agg_att(h_enc).squeeze()

        for idx, num in enumerate(x_len):
            if num < max_x_len:
                att_val[idx, num:] = -100
        att = self.softmax(att_val)

        K_agg = (h_enc * att.unsqueeze(2).expand_as(h_enc)).sum(1)
        if dropout_rate > 0.:
            # K_agg: [batch_size, hid_size]
            K_agg_mask = torch.FloatTensor(K_agg.size()[-1]).view(1, -1)\
                .fill_(1. - dropout_rate).bernoulli().div_(1. - dropout_rate)
            if K_agg.is_cuda:
                K_agg_mask = K_agg_mask.cuda()
            K_agg.data = K_agg.data * K_agg_mask

        agg_score = self.agg_out(K_agg) / self.T
        return agg_score
