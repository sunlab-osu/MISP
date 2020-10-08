# Copyright 2019-present NAVER Corp.
# Apache License v2.0

# Wonseok Hwang
# Sep30, 2018

# Revision by Ziyu:
# 1. preprocessing step is added to save time in batch training.
# 2. fast training/testing

import os, sys, argparse, re, json, datetime, pickle

from matplotlib.pylab import *
import torch.nn as nn
import torch
import torch.nn.functional as F
import random as python_random
# import torchvision.datasets as dsets

# BERT
import SQLova_model.bert.tokenization as tokenization
from SQLova_model.bert.modeling import BertConfig, BertModel
from SQLova_model.sqlova.utils.utils_wikisql import *
from SQLova_model.sqlova.model.nl2sql.wikisql_models import *
from SQLova_model.sqlnet.dbengine import DBEngine

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def construct_hyper_param(parser):
    parser.add_argument('--tepoch', default=200, type=int)
    parser.add_argument("--bS", default=32, type=int,
                        help="Batch size")
    parser.add_argument("--accumulate_gradients", default=1, type=int,
                        help="The number of accumulation of backpropagation to effectivly increase the batch size.")
    parser.add_argument('--fine_tune',
                        default=False,
                        action='store_true',
                        help="If present, BERT is trained.")
    parser.add_argument('--fix_sp',
                        default=False,
                        action='store_true',
                        help='If present, parsing model is not fixed.')

    parser.add_argument("--model_type", default='Seq2SQL_v1', type=str,
                        help="Type of model.")
    # added by Ziyu
    parser.add_argument('--output_dir', type=str, help='Output directory.')
    parser.add_argument('--setting', type=str, choices=['online_pretrain_1p', 'online_pretrain_5p',
                                                        'online_pretrain_10p', 'full_train'])
    parser.add_argument('--job', type=str, help='What to do?',
                        choices=['data_preprocess', 'train', 'dev-test', 'test-test'])
    parser.add_argument('--load_checkpoint_dir', type=str,
                        help='Where to load models for testing? This will overwrite other model paths.')

    # 1.2 BERT Parameters
    parser.add_argument("--vocab_file",
                        default='vocab.txt', type=str,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--max_seq_length",
                        default=222, type=int, # Set based on maximum length of input tokens.
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--num_target_layers",
                        default=2, type=int,
                        help="The Number of final layers of BERT to be used in downstream task.")
    parser.add_argument('--lr_bert', default=1e-5, type=float, help='BERT model learning rate.')
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--no_pretraining', action='store_true', help='Use BERT pretrained model')
    parser.add_argument("--bert_type_abb", default='uS', type=str,
                        help="Type of BERT model to load. e.g., uS, uL, cS, cL, and mcS")

    # 1.3 Seq-to-SQL module parameters
    parser.add_argument('--lS', default=2, type=int, help="The number of LSTM layers.")
    parser.add_argument('--dr', default=0.3, type=float, help="Dropout rate.")
    parser.add_argument('--lr', default=1e-3, type=float, help="Learning rate.")
    parser.add_argument("--hS", default=100, type=int, help="The dimension of hidden vector in the seq-to-SQL module.")

    # 1.4 Execution-guided decoding beam-size. It is used only in test.py
    parser.add_argument('--EG',
                        default=False,
                        action='store_true',
                        help="If present, Execution guided decoding is used in test.")
    parser.add_argument('--beam_size',
                        type=int,
                        default=4,
                        help="The size of beam for smart decoding")

    args = parser.parse_args()

    map_bert_type_abb = {'uS': 'uncased_L-12_H-768_A-12',
                         'uL': 'uncased_L-24_H-1024_A-16',
                         'cS': 'cased_L-12_H-768_A-12',
                         'cL': 'cased_L-24_H-1024_A-16',
                         'mcS': 'multi_cased_L-12_H-768_A-12'}
    args.bert_type = map_bert_type_abb[args.bert_type_abb]
    print(f"BERT-type: {args.bert_type}")

    # Decide whether to use lower_case.
    if args.bert_type_abb == 'cS' or args.bert_type_abb == 'cL' or args.bert_type_abb == 'mcS':
        args.do_lower_case = False
    else:
        args.do_lower_case = True

    # Seeds for random number generation
    python_random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(args.seed)

    #args.toy_model = not torch.cuda.is_available()
    args.toy_model = False
    args.toy_size = 12

    return args


def get_bert(BERT_PT_PATH, bert_type, do_lower_case, no_pretraining):
    bert_config_file = os.path.join(BERT_PT_PATH, f'bert_config_{bert_type}.json')
    vocab_file = os.path.join(BERT_PT_PATH, f'vocab_{bert_type}.txt')
    init_checkpoint = os.path.join(BERT_PT_PATH, f'pytorch_model_{bert_type}.bin')

    bert_config = BertConfig.from_json_file(bert_config_file)
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)
    bert_config.print_status()

    model_bert = BertModel(bert_config)
    if no_pretraining:
        pass
    else:
        model_bert.load_state_dict(torch.load(init_checkpoint, map_location='cpu'))
        print("Load pre-trained parameters.")
    model_bert.to(device)

    return model_bert, tokenizer, bert_config


def get_opt(model, model_bert, fine_tune, train_sp=True):
    if train_sp:
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=args.lr, weight_decay=0)
    else:
        opt = None

    if fine_tune:
        opt_bert = torch.optim.Adam(filter(lambda p: p.requires_grad, model_bert.parameters()),
                                    lr=args.lr_bert, weight_decay=0)
    else:
        opt_bert = None

    return opt, opt_bert


def get_models(args, BERT_PT_PATH, trained=False, path_model_bert=None, path_model=None):
    # some constants
    agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    cond_ops = ['=', '>', '<', 'OP']  # do not know why 'OP' required. Hence,

    print(f"Batch_size = {args.bS * args.accumulate_gradients}")
    print(f"BERT parameters:")
    print(f"learning rate: {args.lr_bert}")
    print(f"Fine-tune BERT: {args.fine_tune}")

    # Get BERT
    model_bert, tokenizer, bert_config = get_bert(BERT_PT_PATH, args.bert_type, args.do_lower_case,
                                                  args.no_pretraining)
    args.iS = bert_config.hidden_size * args.num_target_layers  # Seq-to-SQL input vector dimenstion

    # Get Seq-to-SQL

    n_cond_ops = len(cond_ops)
    n_agg_ops = len(agg_ops)
    print(f"Seq-to-SQL: the number of final BERT layers to be used: {args.num_target_layers}")
    print(f"Seq-to-SQL: the size of hidden dimension = {args.hS}")
    print(f"Seq-to-SQL: LSTM encoding layer size = {args.lS}")
    print(f"Seq-to-SQL: dropout rate = {args.dr}")
    print(f"Seq-to-SQL: learning rate = {args.lr}")
    model = Seq2SQL_v1(args.iS, args.hS, args.lS, args.dr, n_cond_ops, n_agg_ops)
    model = model.to(device)

    if trained:
        assert path_model_bert is not None
        assert path_model is not None

        if torch.cuda.is_available():
            res = torch.load(path_model_bert)
        else:
            res = torch.load(path_model_bert, map_location='cpu')
        model_bert.load_state_dict(res['model_bert'])
        model_bert.to(device)

        if torch.cuda.is_available():
            res = torch.load(path_model)
        else:
            res = torch.load(path_model, map_location='cpu')

        model.load_state_dict(res['model'])

    return model, model_bert, tokenizer, bert_config

def get_data(path_wikisql, args, online_setup=None):
    train_data, train_table, dev_data, dev_table, _, _ = load_wikisql(
        path_wikisql, args.toy_model, args.toy_size, no_w2i=True, no_hs_tok=True)
    if online_setup is not None:
        train_data = [item for idx, item in enumerate(train_data) if idx in set(online_setup['train'])]
        print("## Initial train data size: %d" % len(train_data))
    train_loader, dev_loader = get_loader_wikisql(train_data, dev_data, args.bS, shuffle_train=True)

    return train_data, train_table, dev_data, dev_table, train_loader, dev_loader

def get_processed_data(path_wikisql, args, online_indices=None, num_workers=4, bool_test=False):
    train_data = pickle.load(open(os.path.join(path_wikisql, 'train_tok_processed.pkl'), 'rb'))
    if online_indices is not None:
        train_data = [item for idx, item in enumerate(train_data) if idx in set(online_indices['train'])]
        print("## Initial train data size: %d" % len(train_data))
    train_data = [item for item in train_data if item is not None]

    train_loader = torch.utils.data.DataLoader(
        batch_size=args.bS,
        dataset=train_data,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda x: x  # now dictionary values are not merged!
    )

    filename = 'dev_tok_processed.pkl'
    if bool_test:
        filename = 'test_tok_processed.pkl'
    dev_data = pickle.load(open(os.path.join(path_wikisql, filename), 'rb'))
    dev_data = [item for item in dev_data if item is not None]

    dev_loader = torch.utils.data.DataLoader(
        batch_size=args.bS,
        dataset=dev_data,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda x: x  # now dictionary values are not merged!
    )

    return train_data, None, dev_data, None, train_loader, dev_loader


def report_detail(hds, nlu,
                  g_sc, g_sa, g_wn, g_wc, g_wo, g_wv, g_wv_str, g_sql_q, g_ans,
                  pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wv_str, pr_sql_q, pr_ans,
                  cnt_list, current_cnt):
    cnt_tot, cnt, cnt_sc, cnt_sa, cnt_wn, cnt_wc, cnt_wo, cnt_wv, cnt_wvi, cnt_lx, cnt_x = current_cnt

    print(f'cnt = {cnt} / {cnt_tot} ===============================')

    print(f'headers: {hds}')
    print(f'nlu: {nlu}')

    # print(f's_sc: {s_sc[0]}')
    # print(f's_sa: {s_sa[0]}')
    # print(f's_wn: {s_wn[0]}')
    # print(f's_wc: {s_wc[0]}')
    # print(f's_wo: {s_wo[0]}')
    # print(f's_wv: {s_wv[0][0]}')
    print(f'===============================')
    print(f'g_sc : {g_sc}')
    print(f'pr_sc: {pr_sc}')
    print(f'g_sa : {g_sa}')
    print(f'pr_sa: {pr_sa}')
    print(f'g_wn : {g_wn}')
    print(f'pr_wn: {pr_wn}')
    print(f'g_wc : {g_wc}')
    print(f'pr_wc: {pr_wc}')
    print(f'g_wo : {g_wo}')
    print(f'pr_wo: {pr_wo}')
    print(f'g_wv : {g_wv}')
    # print(f'pr_wvi: {pr_wvi}')
    print('g_wv_str:', g_wv_str)
    print('p_wv_str:', pr_wv_str)
    print(f'g_sql_q:  {g_sql_q}')
    print(f'pr_sql_q: {pr_sql_q}')
    print(f'g_ans: {g_ans}')
    print(f'pr_ans: {pr_ans}')
    print(f'--------------------------------')

    print(cnt_list)

    print(f'acc_lx = {cnt_lx/cnt:.3f}, acc_x = {cnt_x/cnt:.3f}\n',
          f'acc_sc = {cnt_sc/cnt:.3f}, acc_sa = {cnt_sa/cnt:.3f}, acc_wn = {cnt_wn/cnt:.3f}\n',
          f'acc_wc = {cnt_wc/cnt:.3f}, acc_wo = {cnt_wo/cnt:.3f}, acc_wv = {cnt_wv/cnt:.3f}')
    print(f'===============================')


def print_result(epoch, acc, dname):
    ave_loss, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_wvi, acc_wv, acc_lx, acc_x = acc

    print(f'{dname} results ------------')
    print(
        f" Epoch: {epoch}, ave loss: {ave_loss}, acc_sc: {acc_sc:.3f}, acc_sa: {acc_sa:.3f}, acc_wn: {acc_wn:.3f},\
        acc_wc: {acc_wc:.3f}, acc_wo: {acc_wo:.3f}, acc_wvi: {acc_wvi:.3f}, acc_wv: {acc_wv:.3f}, acc_lx: {acc_lx:.3f}, acc_x: {acc_x:.3f}"
    )


def train_fast(train_loader, train_table, model, model_bert, opt, bert_config, tokenizer,
          max_seq_length, num_target_layers, accumulate_gradients=1, check_grad=True,
          st_pos=0, opt_bert=None, path_db=None, dset_name='train'):
    model.train()
    model_bert.train()

    ave_loss = 0
    cnt = 0  # count the # of examples
    cnt_sc = 0  # count the # of correct predictions of select column
    cnt_sa = 0  # of selectd aggregation
    cnt_wn = 0  # of where number
    cnt_wc = 0  # of where column
    cnt_wo = 0  # of where operator
    cnt_wv = 0  # of where-value
    cnt_wvi = 0  # of where-value index (on question tokens)
    cnt_lx = 0  # of logical form acc
    cnt_x = 0  # of execution acc

    # # Engine for SQL querying.
    # engine = DBEngine(os.path.join(path_db, f"{dset_name}.db"))

    for iB, t in enumerate(train_loader):
        cnt += len(t)

        if cnt < st_pos:
            continue

        input_ids, input_mask, segment_ids, tokens, tb, sql_i, hds, i_nlu, i_hds, l_n, l_hpu_batch, l_hs, \
        nlu, nlu_t, nlu_tt, t_to_tt_idx, tt_to_t_idx, g_sc, g_sa, g_wn, g_wc, g_wo, g_wv, g_wvi, g_wvi_corenlp = \
            list(zip(*t))
        l_hpu = [hpu1 for l_hpu1 in l_hpu_batch for hpu1 in l_hpu1]

        # bert encoding
        all_input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
        all_input_mask = torch.tensor(input_mask, dtype=torch.long).to(device)
        all_segment_ids = torch.tensor(segment_ids, dtype=torch.long).to(device)
        wemb_n, wemb_h = get_wemb_bert_fast(bert_config, model_bert, i_hds, l_n, l_hpu, l_hs,
                                            all_input_ids, all_segment_ids, all_input_mask,
                                            num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers)

        # score
        s_sc, s_sa, s_wn, s_wc, s_wo, s_wv = model(wemb_n, l_n, wemb_h, l_hpu, l_hs,
                                                   g_sc=g_sc, g_sa=g_sa, g_wn=g_wn, g_wc=g_wc, g_wvi=g_wvi)

        # Calculate loss & step
        loss = Loss_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi)

        # Calculate gradient
        if iB % accumulate_gradients == 0:  # mode
            # at start, perform zero_grad
            if opt:
                opt.zero_grad()
            if opt_bert:
                opt_bert.zero_grad()
            loss.backward()
            if accumulate_gradients == 1:
                if opt:
                    opt.step()
                if opt_bert:
                    opt_bert.step()
        elif iB % accumulate_gradients == (accumulate_gradients - 1):
            # at the final, take step with accumulated graident
            loss.backward()
            if opt:
                opt.step()
            if opt_bert:
                opt_bert.step()
        else:
            # at intermediate stage, just accumulates the gradients
            loss.backward()

        # Prediction
        pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi = pred_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, )
        pr_wv_str, pr_wv_str_wp = convert_pr_wvi_to_string(pr_wvi, nlu_t, nlu_tt, tt_to_t_idx, nlu)

        # Sort pr_wc:
        #   Sort pr_wc when training the model as pr_wo and pr_wvi are predicted using ground-truth where-column (g_wc)
        #   In case of 'dev' or 'test', it is not necessary as the ground-truth is not used during inference.
        pr_wc_sorted = sort_pr_wc(pr_wc, g_wc)
        pr_sql_i = generate_sql_i(pr_sc, pr_sa, pr_wn, pr_wc_sorted, pr_wo, pr_wv_str, nlu)

        # Cacluate accuracy
        cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, \
        cnt_wc1_list, cnt_wo1_list, \
        cnt_wvi1_list, cnt_wv1_list = get_cnt_sw_list(g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi,
                                                      pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi,
                                                      sql_i, pr_sql_i,
                                                      mode='train')

        cnt_lx1_list = get_cnt_lx_list(cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list,
                                       cnt_wo1_list, cnt_wv1_list)
        # lx stands for logical form accuracy

        # # Execution accuracy test.
        # cnt_x1_list, g_ans, pr_ans = get_cnt_x_list(engine, tb, g_sc, g_sa, sql_i, pr_sc, pr_sa, pr_sql_i)

        # statistics
        ave_loss += loss.item()

        # count
        cnt_sc += sum(cnt_sc1_list)
        cnt_sa += sum(cnt_sa1_list)
        cnt_wn += sum(cnt_wn1_list)
        cnt_wc += sum(cnt_wc1_list)
        cnt_wo += sum(cnt_wo1_list)
        cnt_wvi += sum(cnt_wvi1_list)
        cnt_wv += sum(cnt_wv1_list)
        cnt_lx += sum(cnt_lx1_list)
        # cnt_x += sum(cnt_x1_list)

    ave_loss /= cnt
    acc_sc = cnt_sc / cnt
    acc_sa = cnt_sa / cnt
    acc_wn = cnt_wn / cnt
    acc_wc = cnt_wc / cnt
    acc_wo = cnt_wo / cnt
    acc_wvi = cnt_wv / cnt
    acc_wv = cnt_wv / cnt
    acc_lx = cnt_lx / cnt
    acc_x = cnt_x / cnt

    acc = [ave_loss, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_wvi, acc_wv, acc_lx, acc_x]

    aux_out = 1

    return acc, aux_out


def test_fast(data_loader, data_table, model, model_bert, bert_config, tokenizer, max_seq_length,
              num_target_layers, detail=False, st_pos=0, cnt_tot=1, EG=False, beam_size=4,
              path_db=None, dset_name='test'):
    model.eval()
    model_bert.eval()

    ave_loss = 0
    cnt = 0
    cnt_sc = 0
    cnt_sa = 0
    cnt_wn = 0
    cnt_wc = 0
    cnt_wo = 0
    cnt_wv = 0
    cnt_wvi = 0
    cnt_lx = 0
    cnt_x = 0

    cnt_list = []

    engine = DBEngine(os.path.join(path_db, f"{dset_name}.db"))
    results = []
    for iB, t in enumerate(data_loader):

        cnt += len(t)
        if cnt < st_pos:
            continue

        input_ids, input_mask, segment_ids, tokens, tb, sql_i, hds, i_nlu, i_hds, l_n, l_hpu_batch, l_hs, \
        nlu, nlu_t, nlu_tt, t_to_tt_idx, tt_to_t_idx, g_sc, g_sa, g_wn, g_wc, g_wo, g_wv, g_wvi, g_wvi_corenlp = \
            list(zip(*t))
        l_hpu = [hpu1 for l_hpu1 in l_hpu_batch for hpu1 in l_hpu1]

        try:
            g_wv_str, g_wv_str_wp = convert_pr_wvi_to_string(g_wvi, nlu_t, nlu_tt, tt_to_t_idx, nlu)

        except:
            # Exception happens when where-condition is not found in nlu_tt.
            # In this case, that train example is not used.
            # During test, that example considered as wrongly answered.
            for b in range(len(nlu)):
                results1 = {}
                results1["error"] = "Skip happened"
                results1["nlu"] = nlu[b]
                results1["table_id"] = tb[b]["id"]
                results.append(results1)
            continue

        # bert encoding
        all_input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
        all_input_mask = torch.tensor(input_mask, dtype=torch.long).to(device)
        all_segment_ids = torch.tensor(segment_ids, dtype=torch.long).to(device)
        wemb_n, wemb_h = get_wemb_bert_fast(bert_config, model_bert, i_hds, l_n, l_hpu, l_hs,
                                            all_input_ids, all_segment_ids, all_input_mask,
                                            num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers)

        # model specific part
        # score
        if not EG:
            # No Execution guided decoding
            s_sc, s_sa, s_wn, s_wc, s_wo, s_wv = model(wemb_n, l_n, wemb_h, l_hpu, l_hs)

            # get loss & step
            loss = Loss_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi)

            # prediction
            pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi = pred_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, )
            pr_wv_str, pr_wv_str_wp = convert_pr_wvi_to_string(pr_wvi, nlu_t, nlu_tt, tt_to_t_idx, nlu)
            # g_sql_i = generate_sql_i(g_sc, g_sa, g_wn, g_wc, g_wo, g_wv_str, nlu)
            pr_sql_i = generate_sql_i(pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wv_str, nlu)
        else:
            # Execution guided decoding
            prob_sca, prob_w, prob_wn_w, pr_sc, pr_sa, pr_wn, pr_sql_i = model.beam_forward(wemb_n, l_n, wemb_h, l_hpu,
                                                                                            l_hs, engine, tb,
                                                                                            nlu_t, nlu_tt,
                                                                                            tt_to_t_idx, nlu,
                                                                                            beam_size=beam_size)
            # sort and generate
            pr_wc, pr_wo, pr_wv, pr_sql_i = sort_and_generate_pr_w(pr_sql_i)

            # Follosing variables are just for the consistency with no-EG case.
            pr_wvi = None  # not used
            pr_wv_str = None
            pr_wv_str_wp = None
            loss = torch.tensor([0])

        g_sql_q = generate_sql_q(sql_i, tb)
        pr_sql_q = generate_sql_q(pr_sql_i, tb)

        # Saving for the official evaluation later.
        for b, pr_sql_i1 in enumerate(pr_sql_i):
            results1 = {}
            results1["query"] = pr_sql_i1
            results1["table_id"] = tb[b]["id"]
            results1["nlu"] = nlu[b]
            results.append(results1)

        cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, \
        cnt_wc1_list, cnt_wo1_list, \
        cnt_wvi1_list, cnt_wv1_list = get_cnt_sw_list(g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi,
                                                      pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi,
                                                      sql_i, pr_sql_i,
                                                      mode='test')

        cnt_lx1_list = get_cnt_lx_list(cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list,
                                       cnt_wo1_list, cnt_wv1_list)

        # Execution accura y test
        cnt_x1_list = []
        g_ans = pr_ans = None
        # lx stands for logical form accuracy

        # Execution accuracy test.
        cnt_x1_list, g_ans, pr_ans = get_cnt_x_list(engine, tb, g_sc, g_sa, sql_i, pr_sc, pr_sa, pr_sql_i)

        # stat
        ave_loss += loss.item()

        # count
        cnt_sc += sum(cnt_sc1_list)
        cnt_sa += sum(cnt_sa1_list)
        cnt_wn += sum(cnt_wn1_list)
        cnt_wc += sum(cnt_wc1_list)
        cnt_wo += sum(cnt_wo1_list)
        cnt_wv += sum(cnt_wv1_list)
        cnt_wvi += sum(cnt_wvi1_list)
        cnt_lx += sum(cnt_lx1_list)
        cnt_x += sum(cnt_x1_list)

        current_cnt = [cnt_tot, cnt, cnt_sc, cnt_sa, cnt_wn, cnt_wc, cnt_wo, cnt_wv, cnt_wvi, cnt_lx, cnt_x]
        cnt_list1 = [cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list, cnt_wo1_list, cnt_wv1_list, cnt_lx1_list,
                     cnt_x1_list]
        cnt_list.append(cnt_list1)
        # report
        if detail:
            report_detail(hds, nlu,
                          g_sc, g_sa, g_wn, g_wc, g_wo, g_wv, g_wv_str, g_sql_q, g_ans,
                          pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wv_str, pr_sql_q, pr_ans,
                          cnt_list1, current_cnt)

    ave_loss /= cnt
    acc_sc = cnt_sc / cnt
    acc_sa = cnt_sa / cnt
    acc_wn = cnt_wn / cnt
    acc_wc = cnt_wc / cnt
    acc_wo = cnt_wo / cnt
    acc_wvi = cnt_wvi / cnt
    acc_wv = cnt_wv / cnt
    acc_lx = cnt_lx / cnt
    acc_x = cnt_x / cnt

    acc = [ave_loss, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_wvi, acc_wv, acc_lx, acc_x]
    return acc, results, cnt_list


def run_data_preprocessing(args):
    # data preprocessing
    vocab_file = os.path.join(BERT_PT_PATH, f'vocab_{args.bert_type}.txt')
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=args.do_lower_case)

    for set_name in ['dev', 'test', 'train']:
        new_filename = path_wikisql + '%s_tok_processed.pkl' % (set_name)
        data, tables = load_wikisql_data(path_wikisql, mode=set_name, toy_model=args.toy_model, toy_size=args.toy_size,
                                         no_hs_tok=True, aug=False)

        processed_data = data_preprocessing(tokenizer, data, tables, args.max_seq_length)
        pickle.dump(processed_data, open(new_filename, 'wb'))


if __name__ == '__main__':

    ## 1. Hyper parameters
    parser = argparse.ArgumentParser()
    args = construct_hyper_param(parser)

    ## 2. Paths
    # path_h = '/home/'
    # path_wikisql = os.path.join(path_h, 'data', 'wikisql_tok')
    path_wikisql = 'SQLova_model/download/data/'
    # BERT_PT_PATH = path_wikisql
    BERT_PT_PATH = 'SQLova_model/download/bert/'

    # 3.0 Data preprocessing:
    if args.job == "data_preprocess":
        run_data_preprocessing(args)
        sys.exit()

    # 3. Build & Load models
    if args.load_checkpoint_dir: # load previous checkpoints
        print("Loading models from '%s'..." % args.load_checkpoint_dir)
        path_model_bert = os.path.join(args.load_checkpoint_dir, 'model_bert_best.pt')
        path_model = os.path.join(args.load_checkpoint_dir, 'model_best.pt')
        model, model_bert, tokenizer, bert_config = get_models(
            args, BERT_PT_PATH, trained=True, path_model_bert=path_model_bert, path_model=path_model)
    else: # new models
        model, model_bert, tokenizer, bert_config = get_models(args, BERT_PT_PATH)

    # 4.0. Load data
    if args.setting == 'full_train':
        online_indices = None
    else:
        print("Online pretraining setting: {}".format(args.setting))
        if args.setting == "online_pretrain_1p":
            online_indices = json.load(open(path_wikisql + 'online_setup_1p.json', 'r'))
        elif args.setting == "online_pretrain_5p":
            online_indices = json.load(open(path_wikisql + 'online_setup_5p.json', 'r'))
        else:
            assert args.setting == "online_pretrain_10p"
            online_indices = json.load(open(path_wikisql + 'online_setup_10p.json', 'r'))

    if args.job == "dev-test":
        train_data, train_table, dev_data, dev_table, train_loader, dev_loader = get_processed_data(
            path_wikisql, args, online_indices=online_indices)

        with torch.no_grad():
            acc_dev, results_dev, cnt_list = test_fast(dev_loader,
                                                       dev_table,
                                                       model,
                                                       model_bert,
                                                       bert_config,
                                                       tokenizer,
                                                       args.max_seq_length,
                                                       args.num_target_layers,
                                                       detail=False,
                                                       path_db=path_wikisql,
                                                       st_pos=0,
                                                       dset_name='dev', EG=args.EG)

        print_result(0, acc_dev, 'dev')
        print(acc_dev)

    elif args.job == "test-test":
        train_data, train_table, dev_data, dev_table, train_loader, dev_loader = get_processed_data(
            path_wikisql, args, online_indices=online_indices, bool_test=True)

        with torch.no_grad():
            acc_dev, results_dev, cnt_list = test_fast(dev_loader,
                                                       dev_table,
                                                       model,
                                                       model_bert,
                                                       bert_config,
                                                       tokenizer,
                                                       args.max_seq_length,
                                                       args.num_target_layers,
                                                       detail=False,
                                                       path_db=path_wikisql,
                                                       st_pos=0,
                                                       dset_name='test', EG=args.EG)

        print_result(0, acc_dev, 'test')

    elif args.job == "train":
        model_path = path_save_for_evaluation = args.output_dir
        if not os.path.isdir(path_save_for_evaluation):
            os.mkdir(path_save_for_evaluation)

        ## 4. Load data
        train_data, train_table, dev_data, dev_table, train_loader, dev_loader = get_processed_data(
            path_wikisql, args, online_indices=online_indices)

        ## 5. Get optimizers
        opt, opt_bert = get_opt(model, model_bert, args.fine_tune, not args.fix_sp)
        print("## fix_sp: {}".format(args.fix_sp))

        starting_time = datetime.datetime.now()
        print("## Starting time: {}".format(starting_time))
        ## 6. Train
        acc_lx_t_best = -1
        epoch_best = -1

        early_stop_ep = 10
        patience_counter = 0
        print("## early_stop_ep: {}".format(early_stop_ep))
        for epoch in range(args.tepoch):
            # train
            acc_train, aux_out_train = train_fast(train_loader,
                                                 train_table,
                                                 model,
                                                 model_bert,
                                                 opt,
                                                 bert_config,
                                                 tokenizer,
                                                 args.max_seq_length,
                                                 args.num_target_layers,
                                                 args.accumulate_gradients,
                                                 opt_bert=opt_bert,
                                                 st_pos=0,
                                                 path_db=path_wikisql,
                                                 dset_name='train')

            # check DEV
            with torch.no_grad():
                acc_dev, results_dev, cnt_list = test_fast(dev_loader,
                                                            dev_table,
                                                            model,
                                                            model_bert,
                                                            bert_config,
                                                            tokenizer,
                                                            args.max_seq_length,
                                                            args.num_target_layers,
                                                            detail=False,
                                                            path_db=path_wikisql,
                                                            st_pos=0,
                                                            dset_name='dev', EG=args.EG)


            print_result(epoch, acc_train, 'train')
            print_result(epoch, acc_dev, 'dev')

            # save results for the official evaluation
            save_for_evaluation(path_save_for_evaluation, results_dev, 'dev')

            # save best model
            # Based on Dev Set logical accuracy lx
            acc_lx_t = acc_dev[-2]
            if acc_lx_t > acc_lx_t_best:
                acc_lx_t_best = acc_lx_t
                epoch_best = epoch
                # save best model
                if opt:
                    state = {'model': model.state_dict()}
                    torch.save(state, os.path.join(model_path, 'model_best.pt'))
                    # if datetime.datetime.now() - starting_time > datetime.timedelta(hours=10):
                    #     torch.save(state, os.path.join(model_path, 'model_best_ep%d.pt' % epoch))

                if opt_bert:
                    state = {'model_bert': model_bert.state_dict()}
                    torch.save(state, os.path.join(model_path, 'model_bert_best.pt'))
                    # if datetime.datetime.now() - starting_time > datetime.timedelta(hours=10):
                    #     torch.save(state, os.path.join(model_path, 'model_bert_best_ep%d.pt' % epoch))

                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter == early_stop_ep:
                    print("Early stop after 10 epochs!")
                    break

            print(f" Best Dev lx acc: {acc_lx_t_best} at epoch: {epoch_best}")
            print("## Time spent: {}".format(datetime.datetime.now() - starting_time))
