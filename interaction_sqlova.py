# Adapted from SQLova script for interaction.
# @author: Ziyu Yao
# Oct 7th, 2020
#
import os, sys, argparse, re, json, pickle, math
from copy import deepcopy
from matplotlib.pylab import *
import torch.nn as nn
import torch
import torch.nn.functional as F
import random as python_random
# import torchvision.datasets as dsets
import numpy as np
import time, datetime, pytimeparse

import SQLova_model.bert.tokenization as tokenization
from SQLova_model.bert.modeling import BertConfig, BertModel
from SQLova_model.sqlova.utils.utils_wikisql import *
from SQLova_model.sqlova.model.nl2sql.wikisql_models import *
from SQLova_model.sqlnet.dbengine import DBEngine
from SQLova_model.agent import Agent
from SQLova_model.world_model import WorldModel
from SQLova_model.error_detector import *
from MISP_SQL.question_gen import QuestionGenerator
from SQLova_model.environment import UserSim, RealUser, ErrorEvaluator, GoldUserSim
from user_study_utils import *

np.set_printoptions(precision=3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EARLY_STOP_EPOCH_STAGE1=10
EARLY_STOP_EPOCH_STAGE2=5
EARLY_THRESHOLD=30000


def construct_hyper_param(parser):
    parser.add_argument("--bS", default=1, type=int, help="Batch size")
    parser.add_argument("--model_type", default='Seq2SQL_v1', type=str,
                        help="Type of model.")
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--model_dir', type=str, required=True, help='Which folder to save the model checkpoints.')

    # 1.2 BERT Parameters
    parser.add_argument("--vocab_file",
                        default='vocab.txt', type=str,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--max_seq_length",
                        default=222, type=int,  # Set based on maximum length of input tokens.
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--num_target_layers",
                        default=2, type=int,
                        help="The Number of final layers of BERT to be used in downstream task.")
    parser.add_argument('--lr_bert', default=1e-5, type=float, help='BERT model learning rate.')
    parser.add_argument('--no_pretraining', action='store_true', help='Use BERT pretrained model')
    parser.add_argument("--bert_type_abb", default='uS', type=str,
                        help="Type of BERT model to load. e.g.) uS, uL, cS, cL, and mcS")

    # 1.3 Seq-to-SQL module parameters
    parser.add_argument('--lS', default=2, type=int, help="The number of LSTM layers.")
    parser.add_argument('--dr', default=0.3, type=float, help="Dropout rate.")
    parser.add_argument('--lr', default=1e-3, type=float, help="Learning rate.")
    parser.add_argument("--hS", default=100, type=int, help="The dimension of hidden vector in the seq-to-SQL module.")

    # 1.4 Execution-guided decoding beam-size. It is used only in test.py
    # parser.add_argument('--EG',
    #                     default=False,
    #                     action='store_true',
    #                     help="If present, Execution guided decoding is used in test.")
    # parser.add_argument('--beam_size', # used for non-interactive decoding only
    #                     type=int,
    #                     default=4,
    #                     help="The size of beam for smart decoding")

    # Job setting
    parser.add_argument('--job', default='test_w_interaction', choices=['test_w_interaction', 'online_learning'],
                        help='Set the job. For parser pretraining, see other scripts.')

    # Data setting
    parser.add_argument('--data', default='dev', choices=['dev', 'test', 'user_study', 'online'],
                        help='which dataset to test.')
    parser.add_argument('--data_seed', type=int, default=0, choices=[0, 10, 100],
                        help='Seed for simulated online data order.')

    # Model (initialization/testing) setting
    parser.add_argument('--setting', default='full_train',
                        choices=['full_train', 'online_pretrain_1p', 'online_pretrain_5p', 'online_pretrain_10p'],
                        help='Model setting; checkpoints will be loaded accordingly.')

    # for interaction
    parser.add_argument('--num_options', type=str, default='3',
                        help='[INTERACTION] Number of options (inf or an int number).')
    parser.add_argument('--user', type=str, default='sim', choices=['sim', 'gold_sim', 'real'],
                        help='[INTERACTION] The user setting.')
    parser.add_argument('--err_detector', type=str, default='any',
                        help='[INTERACTION] The error detector: '
                             '(1) prob=x for using policy probability threshold;'
                             '(2) stddev=x for using Bayesian dropout threshold (need to set --dropout and --passes);'
                             '(3) any for querying about every policy action;'
                             '(4) perfect for using a simulated perfect detector.')
    parser.add_argument('--friendly_agent', type=int, default=0, choices=[0, 1],
                        help='[INTERACTION] If 1, the agent will not trigger further interactions '
                             'if any wrong decision is not resolved during parsing.')
    parser.add_argument('--output_path', type=str, default='temp', help='[INTERACTION] Where to save outputs.')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='[INTERACTION] Dropout rate for Bayesian dropout-based uncertainty analysis.')
    parser.add_argument('--passes', type=int, default=1,
                        help='[INTERACTION] Number of decoding passes for Bayesian dropout-based uncertainty analysis.')
    parser.add_argument('--ask_structure', type=int, default=0, choices=[0, 1],
                        help='[INTERACTION] Set to True to allow questions about query structure (WHERE clause).')

    # for online learning
    parser.add_argument('--update_iter', default=1000, type=int,
                        help="[LEARNING] Number of iterations per update.")
    parser.add_argument('--supervision', default='misp_neil',
                        choices=['full_expert', 'misp_neil', 'misp_neil_pos', 'misp_neil_perfect',
                                 'bin_feedback', 'bin_feedback_expert',
                                 'self_train', 'self_train_0.5'],
                        help='[LEARNING] Online learning supervision based on different algorithms.')
    parser.add_argument('--start_iter', default=0, type=int, help='[LEARNING] Iteration to start.')
    parser.add_argument('--end_iter', default=-1, type=int, help='[LEARNING] Iteration to end.')
    parser.add_argument('--auto_iter', default=0, type=int, choices=[0, 1],
                        help='[LEARNING] If 1, unless args.start_iter > 0 is specified, the system will automatically '
                             'search for `start_iter` given the aggregated training data. '
                             'Only applies to args.supervision = misp_neil/bin_feedback(_expert).')

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
    if args.data == "online":
        print("## online data seed: %d" % args.data_seed)
    print("## random seed: %d" % args.seed)
    python_random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # args.toy_model = not torch.cuda.is_available()
    args.toy_model = False
    args.toy_size = 12

    print("Testing data: {}".format(args.data))

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

def get_models(args, BERT_PT_PATH, trained=False, path_model_bert=None, path_model=None):
    # some constants
    agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    cond_ops = ['=', '>', '<', 'OP']  # do not know why 'OP' required. Hence,

    print(f"Batch_size = {args.bS}")
    # print(f"Batch_size = {args.bS * args.accumulate_gradients}")
    print(f"BERT parameters:")
    print(f"learning rate: {args.lr_bert}")
    # print(f"Fine-tune BERT: {args.fine_tune}")

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
        assert path_model_bert != None
        assert path_model != None

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

def get_opt(model, model_bert, fine_tune):
    if fine_tune:
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=args.lr, weight_decay=0)

        opt_bert = torch.optim.Adam(filter(lambda p: p.requires_grad, model_bert.parameters()),
                                    lr=args.lr_bert, weight_decay=0)
    else:
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=args.lr, weight_decay=0)
        opt_bert = None

    return opt, opt_bert

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
        f" Epoch: {epoch}, ave loss: {ave_loss}, acc_sc: {acc_sc:.3f}, acc_sa: {acc_sa:.3f}, acc_wn: {acc_wn:.3f}, \
        acc_wc: {acc_wc:.3f}, acc_wo: {acc_wo:.3f}, acc_wvi: {acc_wvi:.3f}, acc_wv: {acc_wv:.3f}, acc_lx: {acc_lx:.3f}, acc_x: {acc_x:.3f}"
    )


def real_user_interaction(data_loader, data_table, user, agent, tokenizer,
                          max_seq_length, num_target_layers, path_db, save_path):
    dset_name = "test"

    if os.path.isfile(save_path):
        saved_results = json.load(open(save_path, "r"))
        interaction_records = saved_results['interaction_records']
        count_exit = saved_results['count_exit']
        time_spent = datetime.timedelta(seconds=pytimeparse.parse(saved_results['time_spent']))
        st_pos = saved_results['st']
        current_cnt = eval(saved_results['current_cnt'])
        [cnt_tot, cnt_sc, cnt_sa, cnt_wn, cnt_wc, cnt_wo, cnt_wv, cnt_wvi, cnt_lx, cnt_x] = current_cnt
    else:
        cnt_sc = 0
        cnt_sa = 0
        cnt_wn = 0
        cnt_wc = 0
        cnt_wo = 0
        cnt_wv = 0
        cnt_wvi = 0
        cnt_lx = 0
        cnt_x = 0

        interaction_records = {}
        count_exit = 0
        time_spent = datetime.timedelta()
        st_pos = 0
        cnt_tot = 1

    cnt = 0
    engine = DBEngine(os.path.join(path_db, f"{dset_name}.db"))
    for iB, t in enumerate(data_loader):
        assert len(t) == 1
        if cnt < st_pos:
            cnt += 1
            continue
        # Get fields
        nlu, nlu_t, sql_i, sql_q, sql_t, tb, hs_t, hds = get_fields(t, data_table, no_hs_t=True, no_sql_t=True)
        g_sql_q = generate_sql_q(sql_i, tb)

        g_sc, g_sa, g_wn, g_wc, g_wo, g_wv = get_g(sql_i)
        g_wvi_corenlp = get_g_wvi_corenlp(t)

        wemb_n, wemb_h, l_n, l_hpu, l_hs, \
        nlu_tt, t_to_tt_idx, tt_to_t_idx \
            = get_wemb_bert(agent.world_model.bert_config, agent.world_model.model_bert, tokenizer,
                            nlu_t, hds, max_seq_length,
                            num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers)
        g_wvi = get_g_wvi_bert_from_g_wvi_corenlp(t_to_tt_idx, g_wvi_corenlp)
        g_wv_str, g_wv_str_wp = convert_pr_wvi_to_string(g_wvi, nlu_t, nlu_tt, tt_to_t_idx, nlu)

        os.system('clear')  # clear screen
        print_header(len(data_loader.dataset) - cnt)  # interface header

        print(bcolors.BOLD + "Suppose you are given a table with the following " +
              bcolors.BLUE + "header" + bcolors.ENDC +
              bcolors.BOLD + ":" + bcolors.ENDC)
        user.show_table(t[0]['table_id'])  # print table

        print(bcolors.BOLD + "\nAnd you want to answer the following " +
              bcolors.PINK + "question" + bcolors.ENDC +
              bcolors.BOLD + " based on this table:" + bcolors.ENDC)
        print(bcolors.PINK + bcolors.BOLD + t[0]['question'] + bcolors.ENDC + "\n")

        print(bcolors.BOLD + "To help you get the answer automatically,"
                             " the system has the following yes/no questions for you."
                             "\n(When no question prompts, please " +
              bcolors.GREEN + "continue" + bcolors.ENDC +
              bcolors.BOLD + " to the next case)\n" + bcolors.ENDC)

        start_signal = input(bcolors.BOLD + "Ready? please press '" +
                             bcolors.GREEN + "Enter" + bcolors.ENDC + bcolors.BOLD + "' to start!" + bcolors.ENDC)
        while start_signal != "":
            start_signal = input(bcolors.BOLD + "Ready? please press '" +
                                 bcolors.GREEN + "Enter" + bcolors.ENDC + bcolors.BOLD + "' to start!" + bcolors.ENDC)

        start_time = datetime.datetime.now()
        # init decode
        if isinstance(agent.error_detector, ErrorDetectorBayesDropout):
            input_item = [tb, nlu_t, nlu, hds]
        else:
            input_item = [wemb_n, l_n, wemb_h, l_hpu, l_hs, tb, nlu_t, nlu_tt, tt_to_t_idx, nlu]
        init_hyp = agent.world_model.decode(input_item, dec_beam_size=1, bool_verbal=False)[0]

        # interaction
        g_sql = sql_i[0]
        g_sql["g_wvi"] = g_wvi[0]
        hyp, bool_exit = agent.real_user_interactive_parsing_session(
            user, input_item, g_sql, init_hyp, bool_verbal=False)
        print("\nPredicted SQL: {}\n".format(hyp.sql))

        per_time_spent = datetime.datetime.now() - start_time
        time_spent += per_time_spent
        print("Your time spent: {}".format(per_time_spent))

        if bool_exit:
            count_exit += 1

        # post survey
        print("-" * 50)
        print("Post-study Survey: ")
        bool_unclear = input("Is the " + bcolors.BOLD + bcolors.PINK + "initial question" +
                                 bcolors.ENDC + " clear?\nPlease enter y/n: ")
        while bool_unclear not in {'y', 'n'}:
            bool_unclear = input("Is the " + bcolors.BOLD + bcolors.PINK + "initial question" +
                                     bcolors.ENDC + " clear?\nPlease enter y/n: ")
        print("-" * 50)

        pr_sc = [hyp.sql_i['sel']]
        pr_sa = [hyp.sql_i['agg']]
        pr_wn = [len(hyp.sql_i['conds'])]
        pr_wc = [[col for col, _, _ in hyp.sql_i['conds']]]
        pr_wo = [[op for _, op, _ in hyp.sql_i['conds']]]
        pr_sql_i = [hyp.sql_i]
        pr_sql_q = [hyp.sql]

        # Follosing variables are just for the consistency with no-EG case.
        pr_wvi = None  # not used
        pr_wv_str = None
        pr_wv_str_wp = None

        cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list, cnt_wo1_list, cnt_wv1_list, cnt_wvi1_list, \
        cnt_lx1_list, cnt_x1_list, cnt_list1, g_ans, pr_ans = agent.evaluation(
        [pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi, pr_sql_i], [g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi, sql_i], engine, tb)

        # save interaction records
        record = {'nl': t[0]['question'], 'true_sql': g_sql_q[0], 'true_sql_i': "{}".format(sql_i[0]),
                  'init_sql': init_hyp.sql, 'init_sql_i': "{}".format(init_hyp.sql_i),
                  'sql': hyp.sql, 'sql_i': "{}".format(hyp.sql_i),
                  'dec_seq': "{}".format(hyp.dec_seq), 'tag_seq': "{}".format(hyp.tag_seq),
                  'logprob': "{}".format(hyp.logprob), #test time without dropout
                  'lx_correct': int(sum(cnt_lx1_list)), 'x_correct': int(sum(cnt_x1_list)),
                  'exit': bool_exit, 'q_counter': user.q_counter,
                  'questioned_indices': user.questioned_pointers,
                  'questioned_tags': "{}".format(user.questioned_tags),
                  'per_time_spent': str(per_time_spent), 'bool_unclear':bool_unclear,
                  'feedback_records': "{}".format(user.feedback_records),
                  'undo_semantic_units': "{}".format(user.undo_semantic_units),
                  'idx': iB}
        if isinstance(agent.error_detector, ErrorDetectorBayesDropout):
            record.update({'logprob_list': "{}".format(hyp.logprob_list),
                           'test_tag_seq': "{}".format(hyp.test_tag_seq)})
        # interaction_records.append(record)
        interaction_records[cnt] = record

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

        current_cnt = [cnt_tot, cnt_sc, cnt_sa, cnt_wn, cnt_wc, cnt_wo, cnt_wv, cnt_wvi, cnt_lx, cnt_x]
        cnt += 1

        print("Saving records...")
        json.dump({'interaction_records': interaction_records,
                   'current_cnt': "{}".format(current_cnt),
                   'st': cnt, 'time_spent': str(time_spent),
                   'count_exit': count_exit},
                   open(save_path, "w"), indent=4)

        end_signal = input(bcolors.GREEN + bcolors.BOLD +
                               "Next? Press 'Enter' to continue, Ctrl+C to quit." + bcolors.ENDC)
        if end_signal != "":
            return

    print(bcolors.RED + bcolors.BOLD + "Congratulations! You have completed all your task!" + bcolors.ENDC)
    print("Your average time spent: {}".format((time_spent / len(interaction_records))))
    print("You exited %d times." % count_exit)


def interaction(data_loader, data_table, user, agent, tokenizer,
                max_seq_length, num_target_layers, detail=False, st_pos=0, cnt_tot=1,
                path_db=None, dset_name='test', wikisql_sample_ids=None, bool_interaction=True):

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
    results = []
    interaction_records = []
    count_exit = 0
    time_spent = 0.
    count_failure = 0

    engine = DBEngine(os.path.join(path_db, f"{dset_name}.db"))
    for iB, t in enumerate(data_loader):
        if wikisql_sample_ids is not None and iB not in wikisql_sample_ids:
            continue

        cnt += len(t)
        if cnt < st_pos:
            continue
        # Get fields
        nlu, nlu_t, sql_i, sql_q, sql_t, tb, hs_t, hds = get_fields(t, data_table, no_hs_t=True, no_sql_t=True)
        g_sql_q = generate_sql_q(sql_i, tb)

        g_sc, g_sa, g_wn, g_wc, g_wo, g_wv = get_g(sql_i)
        g_wvi_corenlp = get_g_wvi_corenlp(t)

        start_time = time.time()
        wemb_n, wemb_h, l_n, l_hpu, l_hs, \
        nlu_tt, t_to_tt_idx, tt_to_t_idx \
            = get_wemb_bert(agent.world_model.bert_config, agent.world_model.model_bert, tokenizer,
                            nlu_t, hds, max_seq_length,
                            num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers)
        try:
            g_wvi = get_g_wvi_bert_from_g_wvi_corenlp(t_to_tt_idx, g_wvi_corenlp)
            g_wv_str, g_wv_str_wp = convert_pr_wvi_to_string(g_wvi, nlu_t, nlu_tt, tt_to_t_idx, nlu)

        except:
            # Exception happens when where-condition is not found in nlu_tt.
            # In this case, that train example is not used.
            # During test, that example considered as wrongly answered.
            count_failure += 1
            results1 = {}
            results1["error"] = "Skip happened"
            results1["nlu"] = nlu[0]
            results1["table_id"] = tb[0]["id"]
            results.append(results1)
            print("## Failure %d" % count_failure)
            interaction_records.append({'nl': t[0]['question'], 'true_sql': g_sql_q[0],
                                        'true_sql_i': "{}".format(sql_i[0]), "q_counter": 0,
                                        "questioned_indices": []})
            continue

        print("\n" + "#" * 50)
        print("NL input: {}\nTrue SQL: {}".format(t[0]['question'], g_sql_q[0]))
        # init decode
        if isinstance(agent.error_detector, ErrorDetectorBayesDropout):
            input_item = [tb, nlu_t, nlu, hds]
        else:
            input_item = [wemb_n, l_n, wemb_h, l_hpu, l_hs, tb, nlu_t, nlu_tt, tt_to_t_idx, nlu]
        hyp = agent.world_model.decode(input_item, dec_beam_size=1, bool_verbal=False)[0]
        print("## time spent per decode: {:.3f}".format(time.time() - start_time))

        print("-" * 50 + "\nBefore interaction: \ninitial SQL: {}".format(hyp.sql))
        Hypothesis.print_hypotheses([hyp])
        pr_sc = [hyp.sql_i['sel']]
        pr_sa = [hyp.sql_i['agg']]
        pr_wn = [len(hyp.sql_i['conds'])]
        pr_wc = [[col for col, _, _ in hyp.sql_i['conds']]]
        pr_wo = [[op for _, op, _ in hyp.sql_i['conds']]]
        pr_sql_i = [hyp.sql_i]
        pr_wvi = None  # not used
        print("initial evaluation: ")
        cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list, cnt_wo1_list, cnt_wv1_list, cnt_wvi1_list, \
        cnt_lx1_list, cnt_x1_list, cnt_list1, g_ans, pr_ans = agent.evaluation(
            [pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi, pr_sql_i],
            [g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi, sql_i], engine, tb, bool_verbal=True)

        if not bool_interaction:
            record = {'nl': t[0]['question'], 'true_sql': g_sql_q[0], 'true_sql_i': "{}".format(sql_i[0]),
                      'sql': "{}".format(hyp.sql), 'sql_i': "{}".format(hyp.sql_i),
                      'dec_seq': "{}".format(hyp.dec_seq), 'tag_seq': "{}".format(hyp.tag_seq),
                      'logprob': "{}".format(hyp.logprob),
                      'lx_correct': int(sum(cnt_lx1_list)), 'x_correct': int(sum(cnt_x1_list)),
                      "q_counter": 0, "questioned_indices": []}
            if isinstance(agent.error_detector, ErrorDetectorBayesDropout):
                record.update({'logprob_list': "{}".format(hyp.logprob_list),
                              'test_tag_seq': "{}".format(hyp.test_tag_seq)})
            interaction_records.append(record)

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

            # report
            if detail:
                pr_wv_str = None
                current_cnt = [cnt_tot, cnt, cnt_sc, cnt_sa, cnt_wn, cnt_wc, cnt_wo, cnt_wv, cnt_wvi, cnt_lx, cnt_x]
                report_detail(hds, nlu,
                              g_sc, g_sa, g_wn, g_wc, g_wo, g_wv, g_wv_str, g_sql_q, g_ans,
                              pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wv_str, pr_sql_i, pr_ans,
                              cnt_list1, current_cnt)
            continue

        # interaction
        g_sql = sql_i[0]
        g_sql["g_wvi"] = g_wvi[0]
        hyp, bool_exit = agent.interactive_parsing_session(user, input_item, g_sql, hyp, bool_verbal=False)
        print("-" * 50 + "\nAfter interaction:\nfinal SQL: {}".format(hyp.sql))
        Hypothesis.print_hypotheses([hyp])
        print("final evaluation: ")

        # Saving for the official evaluation later.
        results1 = {}
        results1["query"] = hyp.sql_i
        results1["table_id"] = tb[0]["id"]
        results1["nlu"] = nlu[0]
        results.append(results1)

        pr_sc = [hyp.sql_i['sel']]
        pr_sa = [hyp.sql_i['agg']]
        pr_wn = [len(hyp.sql_i['conds'])]
        pr_wc = [[col for col, _, _ in hyp.sql_i['conds']]]
        pr_wo = [[op for _, op, _ in hyp.sql_i['conds']]]
        pr_sql_i = [hyp.sql_i]
        pr_sql_q = [hyp.sql]

        # Follosing variables are just for the consistency with no-EG case.
        pr_wvi = None  # not used
        pr_wv_str = None
        pr_wv_str_wp = None

        cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list, cnt_wo1_list, cnt_wv1_list, cnt_wvi1_list, \
        cnt_lx1_list, cnt_x1_list, cnt_list1, g_ans, pr_ans = agent.evaluation(
            [pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi, pr_sql_i],
            [g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi, sql_i], engine, tb, bool_verbal=True)

        # save interaction records
        record = {'nl': t[0]['question'], 'true_sql': g_sql_q[0], 'true_sql_i': "{}".format(sql_i[0]),
                  'sql': hyp.sql, 'sql_i': "{}".format(hyp.sql_i),
                  'dec_seq': "{}".format(hyp.dec_seq), 'tag_seq': "{}".format(hyp.tag_seq),
                  'logprob': "{}".format(hyp.logprob), #test time without dropout
                  'lx_correct': int(sum(cnt_lx1_list)), 'x_correct': int(sum(cnt_x1_list)),
                  'exit': bool_exit, 'q_counter': user.q_counter,
                  'questioned_indices': user.questioned_pointers,
                  'questioned_tags': "{}".format(user.questioned_tags)}
        if isinstance(agent.error_detector, ErrorDetectorBayesDropout):
            record.update({'logprob_list': "{}".format(hyp.logprob_list),
                           'test_tag_seq': "{}".format(hyp.test_tag_seq)})
        interaction_records.append(record)

        time_spent += (time.time() - start_time)

        if bool_exit:
            count_exit += 1

        # stat
        ave_loss += 0.

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

    if not bool_interaction:
        acc = [ave_loss, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_wvi, acc_wv, acc_lx, acc_x]
        return acc, results, cnt_list, interaction_records

    # stats
    q_count = sum([item['q_counter'] for item in interaction_records])
    dist_q_count = sum([len(set(item['questioned_indices'])) for item in interaction_records])
    print("#questions: {}, #questions per example: {:.3f} (exclude options: {:.3f}).".format(
        q_count, q_count * 1.0 / len(interaction_records), dist_q_count * 1.0 / len(interaction_records)))
    print("#exit: {}".format(count_exit))
    print("Avg time spent: {:.3f}".format((time_spent / len(interaction_records))))

    acc = [ave_loss, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_wvi, acc_wv, acc_lx, acc_x]
    return acc, results, cnt_list, interaction_records


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

        if len(t[0]) == 25:
            input_ids, input_mask, segment_ids, tokens, tb, sql_i, hds, i_nlu, i_hds, l_n, l_hpu_batch, l_hs, \
            nlu, nlu_t, nlu_tt, t_to_tt_idx, tt_to_t_idx, g_sc, g_sa, g_wn, g_wc, g_wo, g_wv, g_wvi, g_wvi_corenlp = \
                list(zip(*t))
            weight_sa = weight_sc = weight_wn = weight_wc = weight_wo = weight_wvi = None
        else:
            assert len(t[0]) == 31
            input_ids, input_mask, segment_ids, tokens, tb, sql_i, hds, i_nlu, i_hds, l_n, l_hpu_batch, l_hs, \
            nlu, nlu_t, nlu_tt, t_to_tt_idx, tt_to_t_idx, g_sc, g_sa, g_wn, g_wc, g_wo, g_wv, g_wvi, g_wvi_corenlp, \
            weight_sc, weight_sa, weight_wn, weight_wc, weight_wo, weight_wvi = \
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
        loss = Loss_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi,
                          weight_sc=weight_sc, weight_sa=weight_sa, weight_wn=weight_wn,
                          weight_wc=weight_wc, weight_wo=weight_wo, weight_wvi=weight_wvi)

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
              path_db=None, dset_name='test', bool_ex=False):
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
        if bool_ex:
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


def run_epochs(model, model_bert, opt, opt_bert, bert_config, tokenizer, path_wikisql, model_path,
               train_loader, train_table, dev_loader, dev_table, test_loader, test_table,
               early_stop_ep=None, bool_eval=True, startime_time=None):
    # some args
    tepoch = 100
    accumulate_gradients = 4
    assert bool_eval
    print("## Actual tepoch %d, accumulate_gradients %d " % (tepoch, accumulate_gradients))
    print("## Early stop epoch: {}".format(early_stop_ep))

    max_seq_length = 222
    num_target_layers = 2

    acc_lx_t_best = -1
    acc_ex_t_best = -1
    epoch_best = -1
    patience_counter = 0
    for epoch in range(tepoch):
        # train
        acc_train, aux_out_train = train_fast(train_loader,
                                             train_table,
                                             model,
                                             model_bert,
                                             opt,
                                             bert_config,
                                             tokenizer,
                                             max_seq_length,
                                             num_target_layers,
                                             accumulate_gradients,
                                             opt_bert=opt_bert,
                                             st_pos=0,
                                             path_db=path_wikisql,
                                             dset_name='train')
        print_result(epoch, acc_train, 'train')
        # check DEV
        if bool_eval:
            with torch.no_grad():
                acc_dev, results_dev, cnt_list = test_fast(dev_loader,
                                                          dev_table,
                                                          model,
                                                          model_bert,
                                                          bert_config,
                                                          tokenizer,
                                                          max_seq_length,
                                                          num_target_layers,
                                                          detail=False,
                                                          path_db=path_wikisql,
                                                          st_pos=0,
                                                          dset_name='dev', EG=False, bool_ex=False)

            print_result(epoch, acc_dev, 'dev')

            # save best model
            # Based on Dev Set logical accuracy lx
            acc_lx_t = acc_dev[-2]
            acc_ex_t = acc_dev[-1]
            if acc_lx_t > acc_lx_t_best:
                acc_lx_t_best = acc_lx_t
                acc_ex_t_best = acc_ex_t
                epoch_best = epoch
                patience_counter = 0
                # save best model
                state = {'model': model.state_dict()}
                torch.save(state, os.path.join(model_path, 'model_best.pt'))

                state = {'model_bert': model_bert.state_dict()}
                torch.save(state, os.path.join(model_path, 'model_bert_best.pt'))
            else:
                patience_counter += 1
                if early_stop_ep is not None and patience_counter == early_stop_ep:
                    print("  Early stop!")
                    break

            print(f" Best Dev lx acc: {acc_lx_t_best} at epoch: {epoch_best}")

        print("  Time stamp: {}".format(datetime.datetime.now()))
        if startime_time is not None:
            print(" Time spent: {}".format(datetime.datetime.now() - startime_time))

        sys.stdout.flush()

    # load back the best model checkpoint
    print("Loading back best checkpoints...")
    if torch.cuda.is_available():
        res = torch.load(os.path.join(model_path, 'model_bert_best.pt'))
    else:
        res = torch.load(os.path.join(model_path, 'model_bert_best.pt'), map_location='cpu')
    model_bert.load_state_dict(res['model_bert'])
    model_bert.to(device)

    if torch.cuda.is_available():
        res = torch.load(os.path.join(model_path, 'model_best.pt'))
    else:
        res = torch.load(os.path.join(model_path, 'model_best.pt'), map_location='cpu')
    model.load_state_dict(res['model'])

    # evaluate: dev lx/ex acc, test lx/ex acc
    with torch.no_grad():
        acc_dev, results_dev, cnt_list = test_fast(dev_loader,
                                                   dev_table,
                                                   model,
                                                   model_bert,
                                                   bert_config,
                                                   tokenizer,
                                                   max_seq_length,
                                                   num_target_layers,
                                                   detail=False,
                                                   path_db=path_wikisql,
                                                   st_pos=0,
                                                   dset_name='dev', EG=False, bool_ex=True)
        print_result(-1, acc_dev, 'dev')
        dev_acc_lx_t_best = acc_dev[-2]
        dev_acc_ex_t_best = acc_dev[-1]

        acc_test, results_test, cnt_list = test_fast(test_loader,
                                                     test_table,
                                                     model,
                                                     model_bert,
                                                     bert_config,
                                                     tokenizer,
                                                     max_seq_length,
                                                     num_target_layers,
                                                     detail=False,
                                                     path_db=path_wikisql,
                                                     st_pos=0,
                                                     dset_name='test', EG=False, bool_ex=True)
        print_result(-1, acc_test, 'test')
        test_acc_lx_t_best = acc_test[-2]
        test_acc_ex_t_best = acc_test[-1]

    return dev_acc_lx_t_best, dev_acc_ex_t_best, test_acc_lx_t_best, test_acc_ex_t_best


def online_learning_full_expert(agent, init_train_data, online_train_data, train_table,
                                val_data, val_table, test_data, test_table,
                                path_db, model_save_path, update_iter, model_renew_fn,
                                start_idx=0, end_idx=-1, batch_size=16):
    # online learning with full supervision (complete SQL query annotation)

    num_total_examples = len(online_train_data)
    print("## data size: %d " % num_total_examples)
    print("## update_iter: %d " % update_iter)
    print("## start_idx: %d" % start_idx)
    print("## end_idx: %d" % end_idx)

    learning_start_time = datetime.datetime.now()
    print("## Online starting time: {}".format(learning_start_time))

    annotation_costs = []
    # pre-calculate annotation cost
    for item in online_train_data:
        query = item[5] # 'query'
        cost = 2 + len(query["conds"]) * 3
        annotation_costs.append(cost)

    for st in np.arange(start_idx, num_total_examples, update_iter):
        annotation_buffer = online_train_data[0: st+update_iter]
        iter_annotation_buffer = online_train_data[st: st+update_iter]
        count_iter = len(annotation_buffer)
        print("~~~\nUpdating base semantic parser at iter {}".format(count_iter))

        # print information about buffer
        for item in iter_annotation_buffer:
            print("NL input: {}".format(item[12])) # 'question'

        model = agent.world_model.semparser
        model_bert = agent.world_model.model_bert
        print("Retraining from scratch...")
        update_buffer = init_train_data + annotation_buffer
        model_renew_fn(model, model_bert)

        print("Train data size: %d" % len(update_buffer))
        opt, opt_bert = get_opt(model, model_bert, True)
        train_loader, dev_loader = get_loader_wikisql(update_buffer, val_data, batch_size, shuffle_train=True)
        test_loader = get_loader_wikisql_v2(test_data, batch_size, False)

        # train
        print("## Starting update at iter {}, anno_cost {}...time spent {}".format(
            count_iter, sum(annotation_costs[0:st+update_iter]), datetime.datetime.now() - learning_start_time))
        model_dir = os.path.join(model_save_path, '%d/' % count_iter)
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)

        dev_acc_lx_t_best, dev_acc_ex_t_best, test_acc_lx_t_best, test_acc_ex_t_best = run_epochs(
            model, model_bert, opt, opt_bert, bert_config, tokenizer, path_db,
            model_dir, train_loader, train_table, dev_loader, val_table, test_loader, test_table,
            early_stop_ep=EARLY_STOP_EPOCH_STAGE1 if count_iter <= EARLY_THRESHOLD else EARLY_STOP_EPOCH_STAGE2,
            bool_eval=True, startime_time=learning_start_time)
        print("## Ending update at iter {}, anno_cost {}, dev acc_lx {}, dev acc_ex {}, test acc_lx {}, "
              "test acc_ex {}...time spent {}\n".format(
            count_iter, sum(annotation_costs[0:st+update_iter]), dev_acc_lx_t_best, dev_acc_ex_t_best,
            test_acc_lx_t_best, test_acc_ex_t_best,
            datetime.datetime.now() - learning_start_time))

        sys.stdout.flush()

        if end_idx != -1 and count_iter == end_idx:
            print("## Ending online learning at iter {}\n".format(end_idx))
            break

    print("## End full training at time {}...time spent {}\n".format(
        datetime.datetime.now(), datetime.datetime.now() - learning_start_time))


def extract_weighted_example(source_t, tt_to_t_idx, gen_sql_i, gen_tag_seq, feedback_records=None, 
                             weight_mode='pos,neg,conf', conf_threshold=None):

    def check_invalidity(weight_sql):
        return weight_sql['sel'] or weight_sql['agg'] or sum([sum(cond) for cond in weight_sql['conds']])

    annotated_example = deepcopy(source_t)
    annotated_example["sql"] = gen_sql_i
    annotated_example["query"] = gen_sql_i
    annotated_example["wvi_corenlp"] = []
    for _cond in gen_sql_i["conds"]:
        for su in gen_tag_seq:
            if su[0] == WHERE_VAL and _cond[0] == su[1][0][-1] and _cond[1] == su[2][-1] and \
                _cond[2] == su[3][-1]:
                annotated_example["wvi_corenlp"].append([tt_to_t_idx[su[3][0]], tt_to_t_idx[su[3][1]]])
    assert len(annotated_example["wvi_corenlp"]) == len(gen_sql_i["conds"])
    # get weights
    if weight_mode == "pos" or weight_mode == "pos,conf":
        annotated_example["weight_sql"] = {'sel': 0.0, 'agg': 0.0,
                                           'conds': [[0.0, 0.0, 0.0] for _ in range(len(gen_sql_i["conds"]))]}
        # add pos
        for su, label in feedback_records:
            if label == 'no':
                continue

            seg_id = su[0]
            if seg_id == SELECT_COL and annotated_example["sql"]["sel"] == su[1][-1]:
                annotated_example["weight_sql"]["sel"] = 1.0
            elif seg_id == SELECT_AGG and annotated_example["sql"]["agg"] == su[2][-1]:
                annotated_example["weight_sql"]["agg"] = 1.0
            elif seg_id == WHERE_COL:
                col_idx = su[1][-1]
                for idx in range(len(annotated_example["sql"]["conds"])):
                    if annotated_example["sql"]["conds"][idx][0] == col_idx:
                        annotated_example["weight_sql"]["conds"][idx][0] = 1.0
                        break
            elif seg_id == WHERE_OP:
                col_idx = su[1][0][-1]
                op_idx = su[2][-1]
                for idx in range(len(annotated_example["sql"]["conds"])):
                    if annotated_example["sql"]["conds"][idx][0] == col_idx and \
                            annotated_example["sql"]["conds"][idx][1] == op_idx:
                        annotated_example["weight_sql"]["conds"][idx][1] = 1.0
                        break
            elif seg_id == WHERE_VAL:
                col_idx = su[1][0][-1]
                op_idx = su[2][-1]
                val_str = su[3][-1]
                for idx in range(len(annotated_example["sql"]["conds"])):
                    if annotated_example["sql"]["conds"][idx][0] == col_idx and \
                            annotated_example["sql"]["conds"][idx][1] == op_idx and \
                            annotated_example["sql"]["conds"][idx][2] == val_str:
                        annotated_example["weight_sql"]["conds"][idx][2] = 1.0
                        break

    if weight_mode == "pos,conf":

        # add confident decisions
        for su in gen_tag_seq:
            prob = su[-2]
            if prob is None or prob < conf_threshold:
                continue

            seg_id = su[0]
            if seg_id == SELECT_COL and annotated_example["sql"]["sel"] == su[1][-1]:
                annotated_example["weight_sql"]["sel"] = 1.0
            elif seg_id == SELECT_AGG and annotated_example["sql"]["agg"] == su[2][-1]:
                annotated_example["weight_sql"]["agg"] = 1.0
            elif seg_id == WHERE_COL:
                col_idx = su[1][-1]
                for idx in range(len(annotated_example["sql"]["conds"])):
                    if annotated_example["sql"]["conds"][idx][0] == col_idx:
                        annotated_example["weight_sql"]["conds"][idx][0] = 1.0
                        break
            elif seg_id == WHERE_OP:
                col_idx = su[1][0][-1]
                op_idx = su[2][-1]
                for idx in range(len(annotated_example["sql"]["conds"])):
                    if annotated_example["sql"]["conds"][idx][0] == col_idx and \
                            annotated_example["sql"]["conds"][idx][1] == op_idx:
                        annotated_example["weight_sql"]["conds"][idx][1] = 1.0
                        break
            elif seg_id == WHERE_VAL:
                col_idx = su[1][0][-1]
                op_idx = su[2][-1]
                val_str = su[3][-1]
                for idx in range(len(annotated_example["sql"]["conds"])):
                    if annotated_example["sql"]["conds"][idx][0] == col_idx and \
                            annotated_example["sql"]["conds"][idx][1] == op_idx and \
                            annotated_example["sql"]["conds"][idx][2] == val_str:
                        annotated_example["weight_sql"]["conds"][idx][2] = 1.0
                        break

    if "weight_sql" not in annotated_example or check_invalidity(annotated_example["weight_sql"]):
        return annotated_example
    else:
        return None


def online_learning(supervision, user, agent, init_train_data, online_data_loader, train_table,
                    val_data, val_table, test_data, test_table, update_iter, model_save_path, record_save_path,
                    model_renew_fn, max_seq_length=222, num_target_layers=2, detail=False,
                    st_pos=0, end_pos=-1, cnt_tot=1, path_db=None, batch_size=16):

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
    results = []
    interaction_records_dict = {'records': [], 'start_iter': 0}
    interaction_records = interaction_records_dict['records']
    count_exit = 0
    count_failure = 0

    count_iter = 0 # online iteration
    num_total_examples = len(online_data_loader.dataset)
    annotation_buffer = [] # processed
    iter_annotation_buffer = [] # processed

    assert supervision.startswith('misp_neil')
    weight_mode = "pos,conf" # misp_neil
    if supervision == "misp_neil_pos":
        weight_mode = "pos"
    print("## supervision: %s, weight_mode: %s " % (supervision, weight_mode))
    print("## data size: %d " % num_total_examples)
    print("## update_iter: %d " % update_iter)
    print("## st_pos: %d " % st_pos)

    # preprocessing initial training data
    init_train_data = data_preprocessing(agent.world_model.tokenizer, init_train_data, train_table,
                                         max_seq_length, bool_remove_none=True,
                                         bool_loss_weight=weight_mode != "pos,neg,conf")

    if st_pos > 0:
        print("## WARNING: inaccurate interaction performance report...")
        print("Loading interaction records from %s..." % record_save_path)
        interaction_records_dict = json.load(open(record_save_path, 'r'))
        interaction_records = interaction_records_dict['records']
        print("Record item size: %d " % len(interaction_records))

    learning_start_time = datetime.datetime.now()
    print("## Online starting time: {}".format(learning_start_time))

    dset_name = 'train'
    engine = DBEngine(os.path.join(path_db, f"{dset_name}.db"))
    for iB, t in enumerate(online_data_loader):
        cnt += len(t)
        assert len(t) == 1
        # if cnt <= st_pos:
        #     count_iter += 1
        #     continue

        # Get fields
        nlu, nlu_t, sql_i, sql_q, sql_t, tb, hs_t, hds = get_fields(t, train_table, no_hs_t=True, no_sql_t=True)
        g_sql_q = generate_sql_q(sql_i, tb)

        g_sc, g_sa, g_wn, g_wc, g_wo, g_wv = get_g(sql_i)
        g_wvi_corenlp = get_g_wvi_corenlp(t)

        # if the record has contained this piece
        if len(interaction_records) >= cnt:
            record = interaction_records[cnt - 1]
            if 'sql_i' not in record:  # failure case
                continue

            gen_sql_i = eval(record['sql_i'])
            gen_tag_seq = eval(record['tag_seq'])
            assert g_sql_q[0] == record['true_sql']

            # BERT processing: 2nd tokenization using WordPiece
            tt_to_t_idx1 = []  # number indicates where sub-token belongs to in 1st-level-tokens (here, CoreNLP).
            for (i, token) in enumerate(nlu_t[0]):
                sub_tokens = agent.world_model.tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    tt_to_t_idx1.append(i)

            if 'feedback_records' in record:
                feedback_records = eval(record['feedback_records'])
            else:
                feedback_records = None
                assert weight_mode == "pos,neg,conf"

            # extract example and add to annotation buffer
            annotated_example = extract_weighted_example(t[0], tt_to_t_idx1, gen_sql_i, gen_tag_seq,
                                                         feedback_records, weight_mode,
                                                         agent.error_detector.prob_threshold)
            if annotated_example is not None:
                iter_annotation_buffer.append(annotated_example)

            count_iter += 1
            if count_iter % update_iter == 0:
                print("  count_iter %d, nl %s" % (count_iter, record['nl']))
                print("  Time stamp: {}".format(datetime.datetime.now()))

        else:
            wemb_n, wemb_h, l_n, l_hpu, l_hs, \
            nlu_tt, t_to_tt_idx, tt_to_t_idx \
                = get_wemb_bert(agent.world_model.bert_config, agent.world_model.model_bert,
                                agent.world_model.tokenizer, nlu_t, hds, max_seq_length,
                                num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers)

            try:
                g_wvi = get_g_wvi_bert_from_g_wvi_corenlp(t_to_tt_idx, g_wvi_corenlp)
                g_wv_str, g_wv_str_wp = convert_pr_wvi_to_string(g_wvi, nlu_t, nlu_tt, tt_to_t_idx, nlu)

            except:
                # Exception happens when where-condition is not found in nlu_tt.
                # In this case, that train example is not used.
                # During test, that example considered as wrongly answered.
                count_failure += 1
                results1 = {}
                results1["error"] = "Skip happened"
                results1["nlu"] = nlu[0]
                results1["table_id"] = tb[0]["id"]
                results.append(results1)
                print("## Failure %d" % count_failure)
                interaction_records.append({'nl': t[0]['question'], 'true_sql': g_sql_q[0],
                                            'true_sql_i': "{}".format(sql_i[0]),
                                            "questioned_indices": [], 'q_counter': 0})
                continue

            print("\n" + "#" * 50)
            print("NL input: {}\nTrue SQL: {}".format(t[0]['question'], g_sql_q[0]))
            # init decode
            if isinstance(agent.error_detector, ErrorDetectorBayesDropout):
                input_item = [tb, nlu_t, nlu, hds]
            else:
                input_item = [wemb_n, l_n, wemb_h, l_hpu, l_hs, tb, nlu_t, nlu_tt, tt_to_t_idx, nlu]
            hyp = agent.world_model.decode(input_item, dec_beam_size=1, bool_verbal=False)[0]

            print("-" * 50 + "\nBefore interaction: \ninitial SQL: {}".format(hyp.sql))
            Hypothesis.print_hypotheses([hyp])
            pr_sc = [hyp.sql_i['sel']]
            pr_sa = [hyp.sql_i['agg']]
            pr_wn = [len(hyp.sql_i['conds'])]
            pr_wc = [[col for col, _, _ in hyp.sql_i['conds']]]
            pr_wo = [[op for _, op, _ in hyp.sql_i['conds']]]
            pr_sql_i = [hyp.sql_i]
            pr_wvi = None  # not used
            print("initial evaluation: ")
            cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list, cnt_wo1_list, cnt_wv1_list, cnt_wvi1_list, \
            cnt_lx1_list, cnt_x1_list, cnt_list1, g_ans, pr_ans = agent.evaluation(
                [pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi, pr_sql_i],
                [g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi, sql_i], engine, tb, bool_verbal=True)

            g_sql = sql_i[0]
            g_sql["g_wvi"] = g_wvi[0]
            hyp, bool_exit = agent.interactive_parsing_session(user, input_item, g_sql, hyp, bool_verbal=False)
            print("-" * 50 + "\nAfter interaction:\nfinal SQL: {}".format(hyp.sql))
            Hypothesis.print_hypotheses([hyp])
            print("final evaluation: ")

            # Saving for the official evaluation later.
            results1 = {}
            results1["query"] = hyp.sql_i
            results1["table_id"] = tb[0]["id"]
            results1["nlu"] = nlu[0]
            results.append(results1)

            pr_sc = [hyp.sql_i['sel']]
            pr_sa = [hyp.sql_i['agg']]
            pr_wn = [len(hyp.sql_i['conds'])]
            pr_wc = [[col for col, _, _ in hyp.sql_i['conds']]]
            pr_wo = [[op for _, op, _ in hyp.sql_i['conds']]]
            pr_sql_i = [hyp.sql_i]
            pr_sql_q = [hyp.sql]

            # Follosing variables are just for the consistency with no-EG case.
            pr_wvi = None  # not used
            pr_wv_str = None
            pr_wv_str_wp = None

            cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list, cnt_wo1_list, cnt_wv1_list, cnt_wvi1_list, \
            cnt_lx1_list, cnt_x1_list, cnt_list1, g_ans, pr_ans = agent.evaluation(
                [pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi, pr_sql_i],
                [g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi, sql_i], engine, tb, bool_verbal=True)

            # save interaction records
            record = {'nl': t[0]['question'], 'true_sql': g_sql_q[0], 'true_sql_i': "{}".format(sql_i[0]),
                      'sql': hyp.sql, 'sql_i': "{}".format(hyp.sql_i),
                      'dec_seq': "{}".format(hyp.dec_seq), 'tag_seq': "{}".format(hyp.tag_seq),
                      'logprob': "{}".format(hyp.logprob),  # test time without dropout
                      'lx_correct': int(sum(cnt_lx1_list)), 'x_correct': int(sum(cnt_x1_list)),
                      'exit': bool_exit, 'q_counter': user.q_counter,
                      'questioned_indices': user.questioned_pointers,
                      'questioned_tags': "{}".format(user.questioned_tags),
                      'feedback_records': "{}".format(user.feedback_records)}
            if isinstance(agent.error_detector, ErrorDetectorBayesDropout):
                record.update({'logprob_list': "{}".format(hyp.logprob_list),
                               'test_tag_seq': "{}".format(hyp.test_tag_seq)})
            interaction_records.append(record)

            # extract example and add to annotation buffer
            annotated_example = extract_weighted_example(t[0], tt_to_t_idx[0], hyp.sql_i, hyp.tag_seq,
                                                         user.feedback_records, weight_mode,
                                                         agent.error_detector.prob_threshold)

            if annotated_example is not None:
                iter_annotation_buffer.append(annotated_example)

            if bool_exit:
                count_exit += 1

            # stat
            ave_loss += 0.

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
            cnt_list.append(cnt_list1)
            # report
            if detail:
                report_detail(hds, nlu,
                              g_sc, g_sa, g_wn, g_wc, g_wo, g_wv, g_wv_str, g_sql_q, g_ans,
                              pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wv_str, pr_sql_q, pr_ans,
                              cnt_list1, current_cnt)
            count_iter += 1
            del wemb_n, wemb_h  # garbage collecting

        if count_iter % update_iter == 0 or count_iter == num_total_examples:  # update model
            if count_iter <= st_pos:
                # preprocessing
                iter_annotation_buffer = data_preprocessing(agent.world_model.tokenizer,
                                                            iter_annotation_buffer, train_table,
                                                            max_seq_length, bool_remove_none=True,
                                                            bool_loss_weight=weight_mode != "pos,neg,conf")
                annotation_buffer.extend(iter_annotation_buffer)
                iter_annotation_buffer = []
                continue

            print("\n~~~\nCurrent interaction performance (iter {}): ".format(count_iter))  # interaction so far
            _ave_loss = ave_loss / cnt
            _acc_sc = cnt_sc / cnt
            _acc_sa = cnt_sa / cnt
            _acc_wn = cnt_wn / cnt
            _acc_wc = cnt_wc / cnt
            _acc_wo = cnt_wo / cnt
            _acc_wvi = cnt_wvi / cnt
            _acc_wv = cnt_wv / cnt
            _acc_lx = cnt_lx / cnt
            _acc_x = cnt_x / cnt
            _acc = [_ave_loss, _acc_sc, _acc_sa, _acc_wn, _acc_wc, _acc_wo, _acc_wvi, _acc_wv, _acc_lx, _acc_x]
            print("Interaction acc: {}".format(_acc))

            q_count = sum([item['q_counter'] for item in interaction_records])
            dist_q_count = sum([len(set(item['questioned_indices'])) for item in interaction_records])
            print("Interaction #questions: {}, #questions per example: {:.3f} (exclude options: {:.3f}).".format(
                q_count, q_count * 1.0 / len(interaction_records), dist_q_count * 1.0 / len(interaction_records)))
            print("Interaction #exit: {}".format(count_exit))
            print("~~~\n")

            print("Saving interaction records to %s..." % record_save_path)
            json.dump(interaction_records_dict, open(record_save_path, 'w'), indent=4)
            # preprocessing
            iter_annotation_buffer = data_preprocessing(agent.world_model.tokenizer,
                                                        iter_annotation_buffer, train_table,
                                                        max_seq_length, bool_remove_none=True,
                                                        bool_loss_weight=weight_mode != "pos,neg,conf")
            annotation_buffer.extend(iter_annotation_buffer)

            # parser update
            print("~~~\nUpdating base semantic parser at iter {}".format(count_iter))
            model = agent.world_model.semparser
            model_bert = agent.world_model.model_bert


            print("Retraining from scratch...")
            update_buffer = init_train_data + annotation_buffer
            # reset parameters
            model_renew_fn(model, model_bert)

            print("Train data size: %d " % len(update_buffer))
            opt, opt_bert = get_opt(model, model_bert, True)
            train_loader, dev_loader = get_loader_wikisql(update_buffer, val_data, batch_size, shuffle_train=True)
            test_loader = get_loader_wikisql_v2(test_data, batch_size, False)

            # train
            print("## Starting update at iter {}, anno_cost {}...time spent {}".format(
                count_iter, sum([item['q_counter'] for item in interaction_records]),
                datetime.datetime.now() - learning_start_time))
            model_dir = os.path.join(model_save_path, '%d/' % count_iter)
            if not os.path.isdir(model_dir):
                os.mkdir(model_dir)

            dev_acc_lx_t_best, dev_acc_ex_t_best, test_acc_lx_t_best, test_acc_ex_t_best = run_epochs(
                model, model_bert, opt, opt_bert, agent.world_model.bert_config,
                agent.world_model.tokenizer, path_db, model_dir, train_loader,
                train_table, dev_loader, val_table, test_loader, test_table,
                early_stop_ep=EARLY_STOP_EPOCH_STAGE1 if count_iter <= EARLY_THRESHOLD else EARLY_STOP_EPOCH_STAGE2,
                bool_eval=True, startime_time=learning_start_time)
            print("## Ending update at iter {}, anno_cost {}, dev acc_lx {}, dev acc_ex {}, test acc_lx {},"
                  "test acc_ex {}...time spent {}\n".format(
                count_iter, sum([item['q_counter'] for item in interaction_records]),
                dev_acc_lx_t_best, dev_acc_ex_t_best, test_acc_lx_t_best, test_acc_ex_t_best,
                datetime.datetime.now() - learning_start_time))

            print("Update interaction_records_dict: start_iter = %d." % count_iter)
            interaction_records_dict['start_iter'] = count_iter
            print("Saving interaction records to %s..." % record_save_path)
            json.dump(interaction_records_dict, open(record_save_path, 'w'), indent=4)

            # clean
            iter_annotation_buffer = []

            # check end_pos
            if end_pos != -1 and count_iter == end_pos:
                print("## Ending online learning at iter {}\n".format(end_pos))
                break

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

    print("## End online learning at time {}...time spent {}\n".format(
        datetime.datetime.now(), datetime.datetime.now() - learning_start_time))

    # stats
    q_count = sum([item['q_counter'] for item in interaction_records])
    dist_q_count = sum([len(set(item['questioned_indices'])) for item in interaction_records])
    print("#questions: {}, #questions per example: {:.3f} (exclude options: {:.3f}).".format(
        q_count, q_count * 1.0 / len(interaction_records), dist_q_count * 1.0 / len(interaction_records)))
    print("#exit: {}".format(count_exit))

    acc = [ave_loss, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_wvi, acc_wv, acc_lx, acc_x]
    return acc, results, cnt_list, interaction_records_dict


def online_learning_self_train(supervision, agent, init_train_data, online_data_loader, train_table,
                               val_data, val_table, test_data, test_table, update_iter, model_save_path, record_save_path,
                               model_renew_fn, max_seq_length=222, num_target_layers=2, detail=False,
                               st_pos=0, end_pos=-1, cnt_tot=1, path_db=None, batch_size=16):
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
    results = []
    interaction_records_dict = {'records': [], 'start_iter': 0}
    interaction_records = interaction_records_dict['records']
    count_exit = 0
    count_failure = 0

    count_iter = 0  # online iteration
    num_total_examples = len(online_data_loader.dataset)
    annotation_buffer = []  # processed
    iter_annotation_buffer = []  # processed
    print("## supervision:", supervision)
    print("## data size: %d " % num_total_examples)
    print("## update_iter: %d " % update_iter)
    conf_threshold = None
    if supervision == 'self_train_0.5':
        conf_threshold = 0.5
    print("## conf_threshold:", str(conf_threshold))
    print("## st_pos: %d " % st_pos)

    # preprocessing initial training data
    init_train_data = data_preprocessing(agent.world_model.tokenizer, init_train_data, train_table,
                                         max_seq_length, bool_remove_none=True,
                                         bool_loss_weight=False)

    if st_pos > 0:
        print("## WARNING: inaccurate interaction performance report...")
        print("Loading interaction records from %s..." % record_save_path)
        interaction_records_dict = json.load(open(record_save_path, 'r'))
        interaction_records = interaction_records_dict['records']
        print("Record item size: %d " % len(interaction_records))

    learning_start_time = datetime.datetime.now()
    print("## Online starting time: {}".format(learning_start_time))

    dset_name = 'train'
    engine = DBEngine(os.path.join(path_db, f"{dset_name}.db"))
    for iB, t in enumerate(online_data_loader):
        cnt += len(t)
        assert len(t) == 1
        # if cnt <= st_pos:
        #     count_iter += 1
        #     continue

        # Get fields
        nlu, nlu_t, sql_i, sql_q, sql_t, tb, hs_t, hds = get_fields(t, train_table, no_hs_t=True, no_sql_t=True)
        g_sql_q = generate_sql_q(sql_i, tb)

        g_sc, g_sa, g_wn, g_wc, g_wo, g_wv = get_g(sql_i)
        g_wvi_corenlp = get_g_wvi_corenlp(t)

        # if the record has contained this piece
        if len(interaction_records) >= cnt:
            record = interaction_records[cnt - 1]
            if 'sql_i' not in record:  # failure case
                continue

            if conf_threshold is None or float(record['logprob']) > np.log(conf_threshold):
                gen_sql_i = eval(record['sql_i'])
                gen_tag_seq = eval(record['tag_seq'])
                assert g_sql_q[0] == record['true_sql']

                # BERT processing: 2nd tokenization using WordPiece
                tt_to_t_idx1 = []  # number indicates where sub-token belongs to in 1st-level-tokens (here, CoreNLP).
                for (i, token) in enumerate(nlu_t[0]):
                    sub_tokens = agent.world_model.tokenizer.tokenize(token)
                    for sub_token in sub_tokens:
                        tt_to_t_idx1.append(i)

                # extract example and add to annotation buffer
                annotated_example = extract_weighted_example(t[0], tt_to_t_idx1, gen_sql_i, gen_tag_seq)
                if annotated_example is not None:
                    iter_annotation_buffer.append(annotated_example)

            count_iter += 1
            if count_iter % update_iter == 0:
                print("  count_iter %d, nl %s" % (count_iter, record['nl']))
                print("  Time stamp: {}".format(datetime.datetime.now()))

        else:
            wemb_n, wemb_h, l_n, l_hpu, l_hs, \
            nlu_tt, t_to_tt_idx, tt_to_t_idx \
                = get_wemb_bert(agent.world_model.bert_config, agent.world_model.model_bert,
                                agent.world_model.tokenizer, nlu_t, hds, max_seq_length,
                                num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers)

            try:
                g_wvi = get_g_wvi_bert_from_g_wvi_corenlp(t_to_tt_idx, g_wvi_corenlp)
                g_wv_str, g_wv_str_wp = convert_pr_wvi_to_string(g_wvi, nlu_t, nlu_tt, tt_to_t_idx, nlu)

            except:
                # Exception happens when where-condition is not found in nlu_tt.
                # In this case, that train example is not used.
                # During test, that example considered as wrongly answered.
                count_failure += 1
                results1 = {}
                results1["error"] = "Skip happened"
                results1["nlu"] = nlu[0]
                results1["table_id"] = tb[0]["id"]
                results.append(results1)
                print("## Failure %d" % count_failure)
                interaction_records.append({'nl': t[0]['question'], 'true_sql': g_sql_q[0],
                                            'true_sql_i': "{}".format(sql_i[0]),
                                            "questioned_indices": [], 'q_counter': 0})
                continue

            print("\n" + "#" * 50)
            print("NL input: {}\nTrue SQL: {}".format(t[0]['question'], g_sql_q[0]))
            # init decode
            if isinstance(agent.error_detector, ErrorDetectorBayesDropout):
                input_item = [tb, nlu_t, nlu, hds]
            else:
                input_item = [wemb_n, l_n, wemb_h, l_hpu, l_hs, tb, nlu_t, nlu_tt, tt_to_t_idx, nlu]
            hyp = agent.world_model.decode(input_item, dec_beam_size=1, bool_verbal=False)[0]

            print("-" * 50 + "\nBefore interaction: \ninitial SQL: {}".format(hyp.sql))
            Hypothesis.print_hypotheses([hyp])
            pr_sc = [hyp.sql_i['sel']]
            pr_sa = [hyp.sql_i['agg']]
            pr_wn = [len(hyp.sql_i['conds'])]
            pr_wc = [[col for col, _, _ in hyp.sql_i['conds']]]
            pr_wo = [[op for _, op, _ in hyp.sql_i['conds']]]
            pr_sql_i = [hyp.sql_i]
            pr_wvi = None  # not used
            print("initial evaluation: ")
            cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list, cnt_wo1_list, cnt_wv1_list, cnt_wvi1_list, \
            cnt_lx1_list, cnt_x1_list, cnt_list1, g_ans, pr_ans = agent.evaluation(
                [pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi, pr_sql_i],
                [g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi, sql_i], engine, tb, bool_verbal=True)

            record = {'nl': t[0]['question'], 'true_sql': g_sql_q[0], 'true_sql_i': "{}".format(sql_i[0]),
                      'sql': "{}".format(hyp.sql), 'sql_i': "{}".format(hyp.sql_i),
                      'dec_seq': "{}".format(hyp.dec_seq), 'tag_seq': "{}".format(hyp.tag_seq),
                      'logprob': "{}".format(hyp.logprob),
                      'lx_correct': int(sum(cnt_lx1_list)), 'x_correct': int(sum(cnt_x1_list)),
                      'q_counter': 0, 'questioned_indices': []}
            if isinstance(agent.error_detector, ErrorDetectorBayesDropout):
                record.update({'logprob_list': "{}".format(hyp.logprob_list),
                               'test_tag_seq': "{}".format(hyp.test_tag_seq)})
            interaction_records.append(record)

            # extract example and add to annotation buffer
            if conf_threshold is None or hyp.logprob > np.log(conf_threshold):
                annotated_example = extract_weighted_example(t[0], tt_to_t_idx[0], hyp.sql_i, hyp.tag_seq)
                if annotated_example is not None:
                    iter_annotation_buffer.append(annotated_example)

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

            cnt_list.append(cnt_list1)
            # report
            if detail:
                pr_wv_str = None
                current_cnt = [cnt_tot, cnt, cnt_sc, cnt_sa, cnt_wn, cnt_wc, cnt_wo, cnt_wv, cnt_wvi, cnt_lx, cnt_x]
                report_detail(hds, nlu,
                              g_sc, g_sa, g_wn, g_wc, g_wo, g_wv, g_wv_str, g_sql_q, g_ans,
                              pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wv_str, pr_sql_i, pr_ans,
                              cnt_list1, current_cnt)

            count_iter += 1
            del wemb_n, wemb_h  # garbage collecting

        if count_iter % update_iter == 0 or count_iter == num_total_examples:  # update model
            if count_iter <= st_pos:
                # preprocessing
                iter_annotation_buffer = data_preprocessing(agent.world_model.tokenizer,
                                                            iter_annotation_buffer, train_table,
                                                            max_seq_length, bool_remove_none=True,
                                                            bool_loss_weight=False)
                annotation_buffer.extend(iter_annotation_buffer)
                iter_annotation_buffer = []
                continue

            print("\n~~~\nCurrent interaction performance (iter {}): ".format(count_iter))  # interaction so far
            _ave_loss = ave_loss / cnt
            _acc_sc = cnt_sc / cnt
            _acc_sa = cnt_sa / cnt
            _acc_wn = cnt_wn / cnt
            _acc_wc = cnt_wc / cnt
            _acc_wo = cnt_wo / cnt
            _acc_wvi = cnt_wvi / cnt
            _acc_wv = cnt_wv / cnt
            _acc_lx = cnt_lx / cnt
            _acc_x = cnt_x / cnt
            _acc = [_ave_loss, _acc_sc, _acc_sa, _acc_wn, _acc_wc, _acc_wo, _acc_wvi, _acc_wv, _acc_lx, _acc_x]
            print("Interaction acc: {}".format(_acc))

            q_count = sum([item['q_counter'] for item in interaction_records])
            dist_q_count = sum([len(set(item['questioned_indices'])) for item in interaction_records])
            print("Interaction #questions: {}, #questions per example: {:.3f} (exclude options: {:.3f}).".format(
                q_count, q_count * 1.0 / len(interaction_records), dist_q_count * 1.0 / len(interaction_records)))
            print("Interaction #exit: {}".format(count_exit))
            print("~~~\n")

            print("Saving interaction records to %s..." % record_save_path)
            json.dump(interaction_records_dict, open(record_save_path, 'w'), indent=4)
            # preprocessing
            iter_annotation_buffer = data_preprocessing(agent.world_model.tokenizer,
                                                        iter_annotation_buffer, train_table,
                                                        max_seq_length, bool_remove_none=True,
                                                        bool_loss_weight=False)
            annotation_buffer.extend(iter_annotation_buffer)

            # parser update
            print("~~~\nUpdating base semantic parser at iter {}".format(count_iter))
            model = agent.world_model.semparser
            model_bert = agent.world_model.model_bert

            print("Retraining from scratch...")
            update_buffer = init_train_data + annotation_buffer
            # reset parameters
            model_renew_fn(model, model_bert)

            print("Train data size: %d " % len(update_buffer))
            opt, opt_bert = get_opt(model, model_bert, True)
            train_loader, dev_loader = get_loader_wikisql(update_buffer, val_data, batch_size, shuffle_train=True)
            test_loader = get_loader_wikisql_v2(test_data, batch_size, False)

            # train
            print("## Starting update at iter {}, anno_cost {}...time spent {}".format(
                count_iter, sum([item['q_counter'] for item in interaction_records]),
                datetime.datetime.now() - learning_start_time))
            model_dir = os.path.join(model_save_path, '%d/' % count_iter)
            if not os.path.isdir(model_dir):
                os.mkdir(model_dir)

            dev_acc_lx_t_best, dev_acc_ex_t_best, test_acc_lx_t_best, test_acc_ex_t_best = run_epochs(
                model, model_bert, opt, opt_bert, agent.world_model.bert_config,
                agent.world_model.tokenizer, path_db, model_dir, train_loader,
                train_table, dev_loader, val_table, test_loader, test_table,
                early_stop_ep=EARLY_STOP_EPOCH_STAGE1 if count_iter <= EARLY_THRESHOLD else EARLY_STOP_EPOCH_STAGE2,
                bool_eval=True, startime_time=learning_start_time)
            print("## Ending update at iter {}, anno_cost {}, dev acc_lx {}, dev acc_ex {}, test acc_lx {},"
                  "test acc_ex {}...time spent {}\n".format(
                count_iter, sum([item['q_counter'] for item in interaction_records]),
                dev_acc_lx_t_best, dev_acc_ex_t_best, test_acc_lx_t_best, test_acc_ex_t_best,
                datetime.datetime.now() - learning_start_time))

            print("Update interaction_records_dict: start_iter = %d." % count_iter)
            interaction_records_dict['start_iter'] = count_iter
            print("Saving interaction records to %s..." % record_save_path)
            json.dump(interaction_records_dict, open(record_save_path, 'w'), indent=4)

            # clean
            iter_annotation_buffer = []

            # check end_pos
            if end_pos != -1 and count_iter == end_pos:
                print("## Ending online learning at iter {}\n".format(end_pos))
                break


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

    print("## End online learning at time {}...time spent {}\n".format(
        datetime.datetime.now(), datetime.datetime.now() - learning_start_time))

    # stats
    q_count = sum([item['q_counter'] for item in interaction_records])
    dist_q_count = sum([len(set(item['questioned_indices'])) for item in interaction_records])
    print("#questions: {}, #questions per example: {:.3f} (exclude options: {:.3f}).".format(
        q_count, q_count * 1.0 / len(interaction_records), dist_q_count * 1.0 / len(interaction_records)))
    print("#exit: {}".format(count_exit))

    acc = [ave_loss, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_wvi, acc_wv, acc_lx, acc_x]
    return acc, results, cnt_list, interaction_records_dict


def online_learning_bin_feedback(supervision, agent, init_train_data, online_data_loader, train_table,
                                 val_data, val_table, test_data, test_table, model_save_path, record_save_path, path_db,
                                 update_iter, model_renew_fn, max_seq_length=222, num_target_layers=2,
                                 detail=False, cnt_tot=1, start_idx=0, end_idx=-1, batch_size=16):
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
    results = []
    interaction_records_dict = {'records': [], 'start_iter': 0}
    interaction_records = interaction_records_dict['records']
    count_exit = 0
    count_failure = 0

    count_iter = 0  # online iteration
    num_total_examples = len(online_data_loader.dataset)
    annotation_buffer = []  # processed
    iter_annotation_buffer = []  # processed

    print("## data size: %d " % num_total_examples)
    print("## update_iter: %d " % update_iter)
    print("## start_idx: %d" % start_idx)
    print("## end_idx: %d" % end_idx)

    # preprocessing initial training data
    init_train_data = data_preprocessing(agent.world_model.tokenizer, init_train_data, train_table,
                                         max_seq_length, bool_remove_none=True)

    annotation_costs = []

    if start_idx > 0:
        print("Loading interaction records from %s..." % record_save_path)
        interaction_records_dict = json.load(open(record_save_path, 'r'))
        interaction_records = interaction_records_dict['records']
        print("Record item size: %d " % len(interaction_records))

    learning_start_time = datetime.datetime.now()
    print("## Online starting time: {}".format(learning_start_time))

    dset_name = 'train'
    engine = DBEngine(os.path.join(path_db, f"{dset_name}.db"))
    for iB, t in enumerate(online_data_loader):
        cnt += len(t)
        assert len(t) == 1
        # if cnt <= st_pos:
        #     count_iter += 1
        #     continue

        # Get fields
        nlu, nlu_t, sql_i, sql_q, sql_t, tb, hs_t, hds = get_fields(t, train_table, no_hs_t=True, no_sql_t=True)
        g_sql_q = generate_sql_q(sql_i, tb)

        g_sc, g_sa, g_wn, g_wc, g_wo, g_wv = get_g(sql_i)
        g_wvi_corenlp = get_g_wvi_corenlp(t)

        if len(interaction_records) >= cnt:
            record = interaction_records[cnt - 1]
            if 'sql_i' not in record:  # failure case
                continue

            assert record['nl'] == t[0]['question']

            x_correct = record['x_correct']
            if x_correct:
                gen_sql_i = eval(record['sql_i'])
                gen_tag_seq = eval(record['tag_seq'])

                # BERT processing: 2nd tokenization using WordPiece
                tt_to_t_idx1 = []  # number indicates where sub-token belongs to in 1st-level-tokens (here, CoreNLP).
                for (i, token) in enumerate(nlu_t[0]):
                    sub_tokens = agent.world_model.tokenizer.tokenize(token)
                    for sub_token in sub_tokens:
                        tt_to_t_idx1.append(i)

                annotated_example = extract_weighted_example(
                    t[0], tt_to_t_idx1, gen_sql_i, gen_tag_seq)
                iter_annotation_buffer.append(annotated_example)
            elif supervision == "bin_feedback_expert":
                iter_annotation_buffer.append(t[0])

            cost = 2 + len(eval(record['true_sql_i'])["conds"]) * 3
            annotation_costs.append(cost)

            count_iter += 1
            if count_iter % update_iter == 0:
                print("  count_iter %d, nl %s" % (count_iter, record['nl']))
                print("  Time stamp: {}".format(datetime.datetime.now()))

        else:

            wemb_n, wemb_h, l_n, l_hpu, l_hs, \
            nlu_tt, t_to_tt_idx, tt_to_t_idx \
                = get_wemb_bert(agent.world_model.bert_config, agent.world_model.model_bert,
                                agent.world_model.tokenizer, nlu_t, hds, max_seq_length,
                                num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers)

            try:
                g_wvi = get_g_wvi_bert_from_g_wvi_corenlp(t_to_tt_idx, g_wvi_corenlp)
                g_wv_str, g_wv_str_wp = convert_pr_wvi_to_string(g_wvi, nlu_t, nlu_tt, tt_to_t_idx, nlu)

            except:
                # Exception happens when where-condition is not found in nlu_tt.
                # In this case, that train example is not used.
                # During test, that example considered as wrongly answered.
                count_failure += 1
                results1 = {}
                results1["error"] = "Skip happened"
                results1["nlu"] = nlu[0]
                results1["table_id"] = tb[0]["id"]
                results.append(results1)
                print("## Failure %d" % count_failure)
                interaction_records.append({'nl': t[0]['question'], 'true_sql': g_sql_q[0],
                                            'true_sql_i': "{}".format(sql_i[0]),
                                            "questioned_indices": [], 'q_counter': 0})
                continue

            print("\n" + "#" * 50)
            print("NL input: {}\nTrue SQL: {}".format(t[0]['question'], g_sql_q[0]))
            # init decode
            if isinstance(agent.error_detector, ErrorDetectorBayesDropout):
                input_item = [tb, nlu_t, nlu, hds]
            else:
                input_item = [wemb_n, l_n, wemb_h, l_hpu, l_hs, tb, nlu_t, nlu_tt, tt_to_t_idx, nlu]
            hyp = agent.world_model.decode(input_item, dec_beam_size=1, bool_verbal=False)[0]

            print("-" * 50 + "\nBefore interaction: \ninitial SQL: {}".format(hyp.sql))
            Hypothesis.print_hypotheses([hyp])
            pr_sc = [hyp.sql_i['sel']]
            pr_sa = [hyp.sql_i['agg']]
            pr_wn = [len(hyp.sql_i['conds'])]
            pr_wc = [[col for col, _, _ in hyp.sql_i['conds']]]
            pr_wo = [[op for _, op, _ in hyp.sql_i['conds']]]
            pr_sql_i = [hyp.sql_i]
            pr_wvi = None  # not used
            print("initial evaluation: ")
            cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list, cnt_wo1_list, cnt_wv1_list, cnt_wvi1_list, \
            cnt_lx1_list, cnt_x1_list, cnt_list1, g_ans, pr_ans = agent.evaluation(
                [pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi, pr_sql_i],
                [g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi, sql_i], engine, tb, bool_verbal=True)

            record = {'nl': t[0]['question'], 'true_sql': g_sql_q[0], 'true_sql_i': "{}".format(sql_i[0]),
                      'sql': "{}".format(hyp.sql), 'sql_i': "{}".format(hyp.sql_i),
                      'dec_seq': "{}".format(hyp.dec_seq), 'tag_seq': "{}".format(hyp.tag_seq),
                      'logprob': "{}".format(hyp.logprob),
                      'lx_correct': int(sum(cnt_lx1_list)), 'x_correct': int(sum(cnt_x1_list))}
            if isinstance(agent.error_detector, ErrorDetectorBayesDropout):
                record.update({'logprob_list': "{}".format(hyp.logprob_list),
                               'test_tag_seq': "{}".format(hyp.test_tag_seq)})
            interaction_records.append(record)
            
            if int(sum(cnt_x1_list)) == 1: # execution correct
                # iter_annotation_buffer.append(t[0])
                annotated_example = extract_weighted_example(
                    t[0], tt_to_t_idx[0], hyp.sql_i, hyp.tag_seq)
                iter_annotation_buffer.append(annotated_example)
            elif supervision == "bin_feedback_expert":
                iter_annotation_buffer.append(t[0])

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

            # report
            if detail:
                pr_wv_str = None
                current_cnt = [cnt_tot, cnt, cnt_sc, cnt_sa, cnt_wn, cnt_wc, cnt_wo, cnt_wv, cnt_wvi, cnt_lx, cnt_x]
                report_detail(hds, nlu,
                              g_sc, g_sa, g_wn, g_wc, g_wo, g_wv, g_wv_str, g_sql_q, g_ans,
                              pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wv_str, pr_sql_i, pr_ans,
                              cnt_list1, current_cnt)

            count_iter += 1
            del wemb_n, wemb_h  # garbage collecting

            cost = 2 + len(eval(record['true_sql_i'])["conds"]) * 3
            annotation_costs.append(cost)

        if count_iter % update_iter == 0 or count_iter == num_total_examples:  # update model
            if count_iter <= start_idx:
                # preprocessing
                iter_annotation_buffer = data_preprocessing(agent.world_model.tokenizer,
                                                            iter_annotation_buffer, train_table,
                                                            max_seq_length, bool_remove_none=True)
                annotation_buffer.extend(iter_annotation_buffer)
                iter_annotation_buffer = []
                continue

            print("\n~~~\nCurrent interaction performance (iter {}): ".format(count_iter))  # interaction so far
            _ave_loss = ave_loss / cnt
            _acc_sc = cnt_sc / cnt
            _acc_sa = cnt_sa / cnt
            _acc_wn = cnt_wn / cnt
            _acc_wc = cnt_wc / cnt
            _acc_wo = cnt_wo / cnt
            _acc_wvi = cnt_wvi / cnt
            _acc_wv = cnt_wv / cnt
            _acc_lx = cnt_lx / cnt
            _acc_x = cnt_x / cnt
            _acc = [_ave_loss, _acc_sc, _acc_sa, _acc_wn, _acc_wc, _acc_wo, _acc_wvi, _acc_wv, _acc_lx, _acc_x]
            print("Interaction acc: {}".format(_acc))

            print("Saving interaction records to %s..." % record_save_path)
            json.dump(interaction_records_dict, open(record_save_path, 'w'), indent=4)
            # preprocessing
            iter_annotation_buffer = data_preprocessing(agent.world_model.tokenizer,
                                                        iter_annotation_buffer, train_table,
                                                        max_seq_length, bool_remove_none=True)
            annotation_buffer.extend(iter_annotation_buffer)
            iter_annotation_buffer = []

            print("~~~\nUpdating base semantic parser at iter {}".format(count_iter))

            model = agent.world_model.semparser
            model_bert = agent.world_model.model_bert
            print("Retraining from scratch...")
            update_buffer = init_train_data + annotation_buffer
            model_renew_fn(model, model_bert)

            print("Train data size: %d" % len(update_buffer))
            opt, opt_bert = get_opt(model, model_bert, True)
            train_loader, dev_loader = get_loader_wikisql(update_buffer, val_data, batch_size, shuffle_train=True)
            test_loader = get_loader_wikisql_v2(test_data, batch_size, False)

            # train
            print("## Starting update at iter {}, anno_cost {}...time spent {}".format(
                count_iter, sum(annotation_costs), datetime.datetime.now() - learning_start_time))
            model_dir = os.path.join(model_save_path, '%d/' % count_iter)
            if not os.path.isdir(model_dir):
                os.mkdir(model_dir)

            dev_acc_lx_t_best, dev_acc_ex_t_best, test_acc_lx_t_best, test_acc_ex_t_best = run_epochs(
                model, model_bert, opt, opt_bert, agent.world_model.bert_config,
                agent.world_model.tokenizer, path_db,
                model_dir, train_loader, train_table, dev_loader, val_table,
                test_loader, test_table,
                early_stop_ep=EARLY_STOP_EPOCH_STAGE1 if count_iter <= EARLY_THRESHOLD else EARLY_STOP_EPOCH_STAGE2,
                bool_eval=True, startime_time=learning_start_time)
            print("## Ending update at iter {}, anno_cost {}, dev acc_lx {}, dev acc_ex {}, test acc_lx {}, "
                  "test acc_ex {}...time spent {}\n".format(
                count_iter, sum(annotation_costs), dev_acc_lx_t_best, dev_acc_ex_t_best,
                test_acc_lx_t_best, test_acc_ex_t_best,
                datetime.datetime.now() - learning_start_time))

            print("Update interaction_records_dict: start_iter = %d." % count_iter)
            interaction_records_dict['start_iter'] = count_iter
            print("Saving interaction records to %s..." % record_save_path)
            json.dump(interaction_records_dict, open(record_save_path, 'w'), indent=4)

            sys.stdout.flush()

            if end_idx != -1 and count_iter == end_idx:
                print("## Ending online learning at iter {}\n".format(end_idx))
                break

    print("## End full training at time {}...time spent {}\n".format(
        datetime.datetime.now(), datetime.datetime.now() - learning_start_time))


def online_learning_misp_perfect(user, agent, online_data_loader, train_table,
                                 update_iter, model_save_path, record_save_path,
                                 max_seq_length=222, num_target_layers=2, st_pos=0, end_pos=-1):
    # This function simulates MISP_NEIL^*, i.e., the best version of MISP with a perfect error detector
    # and a perfect interaction design (thus can get gold answers and detect redundant/missing components).
    # The learned parser will be the same as "full expert" parser.

    assert args.ask_structure and args.user == "gold_sim" and args.err_detector == "perfect"

    cnt = 0
    interaction_records = []
    count_exit = 0
    count_failure = 0

    count_iter = 0  # online iteration
    num_total_examples = len(online_data_loader.dataset)

    if st_pos > 0:
        print("Loading interaction records from %s..." % record_save_path)
        interaction_records = json.load(open(record_save_path, 'r'))
        print("Record item size: %d " % len(interaction_records))

    dset_name = 'train'
    for iB, t in enumerate(online_data_loader):
        cnt += len(t)
        assert len(t) == 1

        if len(interaction_records) >= cnt:
            record = interaction_records[cnt - 1]
            if 'sql_i' not in record:  # failure case
                continue

            count_iter += 1

        else:
            # Get fields
            nlu, nlu_t, sql_i, sql_q, sql_t, tb, hs_t, hds = get_fields(t, train_table, no_hs_t=True, no_sql_t=True)
            g_sql_q = generate_sql_q(sql_i, tb)

            g_sc, g_sa, g_wn, g_wc, g_wo, g_wv = get_g(sql_i)
            g_wvi_corenlp = get_g_wvi_corenlp(t)

            wemb_n, wemb_h, l_n, l_hpu, l_hs, \
            nlu_tt, t_to_tt_idx, tt_to_t_idx \
                = get_wemb_bert(agent.world_model.bert_config, agent.world_model.model_bert,
                                agent.world_model.tokenizer, nlu_t, hds, max_seq_length,
                                num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers)

            try:
                g_wvi = get_g_wvi_bert_from_g_wvi_corenlp(t_to_tt_idx, g_wvi_corenlp)
                g_wv_str, g_wv_str_wp = convert_pr_wvi_to_string(g_wvi, nlu_t, nlu_tt, tt_to_t_idx, nlu)

            except:
                # Exception happens when where-condition is not found in nlu_tt.
                # In this case, that train example is not used.
                # During test, that example considered as wrongly answered.
                count_failure += 1
                print("## Failure %d" % count_failure)
                interaction_records.append({'nl': t[0]['question'], 'true_sql': g_sql_q[0],
                                            'true_sql_i': "{}".format(sql_i[0]),
                                            "questioned_indices": [], 'q_counter': 0,
                                            'count_additional_q': 0})
                continue

            print("\n" + "#" * 50)
            print("NL input: {}\nTrue SQL: {}".format(t[0]['question'], g_sql_q[0]))
            # init decode
            if isinstance(agent.error_detector, ErrorDetectorBayesDropout):
                input_item = [tb, nlu_t, nlu, hds]
            else:
                input_item = [wemb_n, l_n, wemb_h, l_hpu, l_hs, tb, nlu_t, nlu_tt, tt_to_t_idx, nlu]
            hyp = agent.world_model.decode(input_item, dec_beam_size=1, bool_verbal=False)[0]

            print("-" * 50 + "\nBefore interaction: \ninitial SQL: {}".format(hyp.sql))

            # interaction
            g_sql = sql_i[0]
            g_sql["g_wvi"] = g_wvi[0]
            hyp, bool_exit = agent.interactive_parsing_session(user, input_item, g_sql, hyp, bool_verbal=False)
            print("-" * 50 + "\nAfter interaction:\nfinal SQL: {}".format(hyp.sql))
            Hypothesis.print_hypotheses([hyp])

            # check missing/redundant part
            assert hyp.sql_i['sel'] == sql_i[0]['sel']
            assert hyp.sql_i['agg'] == sql_i[0]['agg']
            count_additional_q = 0
            if len(hyp.sql_i['conds']) < len(sql_i[0]['conds']): # missing conditions
                count_additional_q += (len(sql_i[0]['conds']) - len(hyp.sql_i['conds'])) * 3
            elif len(hyp.sql_i['conds']) > len(sql_i[0]['conds']):
                for col, op, val in hyp.sql_i['conds']:
                    if col not in [_col for _col, _op, _val in sql_i[0]['conds']]:
                        count_additional_q += 3
                    elif (col, op) not in [(_col, _op) for _col, _op, _val in sql_i[0]['conds']]:
                        count_additional_q += 2
                    elif (col, op, val) not in [(_col, _op, _val) for _col, _op, _val in sql_i[0]['conds']]:
                        count_additional_q += 1

            print("count_additional_q: {}".format(count_additional_q))

            # save interaction records
            record = {'nl': t[0]['question'], 'true_sql': g_sql_q[0], 'true_sql_i': "{}".format(sql_i[0]),
                      'sql': hyp.sql, 'sql_i': "{}".format(hyp.sql_i),
                      'dec_seq': "{}".format(hyp.dec_seq), 'tag_seq': "{}".format(hyp.tag_seq),
                      'logprob': "{}".format(hyp.logprob),  # test time without dropout
                      'exit': bool_exit, 'q_counter': user.q_counter,
                      'count_additional_q': count_additional_q,
                      'questioned_indices': user.questioned_pointers,
                      'questioned_tags': "{}".format(user.questioned_tags),
                      'feedback_records': "{}".format(user.feedback_records)}
            interaction_records.append(record)

            if bool_exit:
                count_exit += 1

            count_iter += 1
            del wemb_n, wemb_h  # garbage collecting

        if count_iter % update_iter == 0 or count_iter == num_total_examples:  # update model
            if count_iter < st_pos:
                continue

            if count_iter > st_pos:
                # report q counts
                q_count = sum([item['q_counter'] + item['count_additional_q'] for item in interaction_records])
                print("## End update at iter {}, anno_cost {}\n".format(count_iter, q_count))

                print("Saving interaction records to %s..." % record_save_path)
                json.dump(interaction_records, open(record_save_path, 'w'), indent=4)

            # check end_pos
            if end_pos != -1 and count_iter == end_pos:
                print("## Ending online learning at iter {}\n".format(end_pos))
                print(datetime.datetime.now())
                break

            # loading models
            model_dir = os.path.join(model_save_path, '%d/' % count_iter)
            print("Loading model from %s..." % model_dir)

            path_model = os.path.join(model_dir, 'model_best.pt')
            path_model_bert = os.path.join(model_dir, 'model_bert_best.pt')
            if torch.cuda.is_available():
                res = torch.load(path_model_bert)
            else:
                res = torch.load(path_model_bert, map_location='cpu')
            agent.world_model.model_bert.load_state_dict(res['model_bert'])
            agent.world_model.model_bert.to(device)

            if torch.cuda.is_available():
                res = torch.load(path_model)
            else:
                res = torch.load(path_model, map_location='cpu')
            agent.world_model.semparser.load_state_dict(res['model'])

            print(datetime.datetime.now())

    print("Saving interaction records to %s..." % record_save_path)
    json.dump(interaction_records, open(record_save_path, 'w'), indent=4)


def load_processed_wikisql_data(path_wikisql, dset_name):
    data = pickle.load(open(os.path.join(path_wikisql, '%s_tok_processed.pkl' % dset_name), 'rb'))

    path_table = os.path.join(path_wikisql, dset_name + '.tables.jsonl')
    table = {}
    with open(path_table) as f:
        for idx, line in enumerate(f):
            t1 = json.loads(line.strip())
            table[t1['id']] = t1

    return data, table


if __name__ == '__main__':

    ## 1. Hyper parameters
    parser = argparse.ArgumentParser()
    args = construct_hyper_param(parser)

    ## 2. Paths
    path_wikisql = 'SQLova_model/download/data/'
    BERT_PT_PATH = 'SQLova_model/download/bert/'
    model_dir = args.model_dir
    
    print("## job: {}".format(args.job))
    print("## setting: {}".format(args.setting))
    print("## model_dir: {}".format(args.model_dir))
    if args.auto_iter:
        print("## auto_iter is on.")
        print("\targs.start_iter=%d, args.end_iter=%d." % (args.start_iter, args.end_iter))

    path_model_bert = os.path.join(model_dir, "model_bert_best.pt")
    path_model = os.path.join(model_dir, "model_best.pt")

    ## 3. Load data
    if args.job == 'online_learning':
        dev_data, dev_table = load_processed_wikisql_data(path_wikisql, 'dev')
        test_data, test_table = load_processed_wikisql_data(path_wikisql, 'test')
        test_data = [item for item in test_data if item is not None]
    else:
        if args.data == "user_study":
            test_data, test_table = load_wikisql_data(path_wikisql, mode="test", toy_model=args.toy_model,
                                                      toy_size=args.toy_size, no_hs_tok=True)
            sampled_ids = json.load(open("SQLova_model/download/data/user_study_ids.json", "r"))
            test_data = [test_data[idx] for idx in sampled_ids]
        else:
            # args.data in ["dev", "test"]
            test_data, test_table = load_wikisql_data(
                path_wikisql, mode=args.data,
                toy_model=args.toy_model, toy_size=args.toy_size, no_hs_tok=True)

    # 4. Build & Load models
    model, model_bert, tokenizer, bert_config = get_models(args, BERT_PT_PATH, trained=True,
                                                           path_model_bert=path_model_bert,
                                                           path_model=path_model)
    model.eval()
    model_bert.eval()

    ## 5. Create ISQL agent
    print("Creating MISP agent...")
    question_generator = QuestionGenerator()
    error_evaluator = ErrorEvaluator()
    print("## user: {}".format(args.user))
    if args.user == "real":
        user = RealUser(error_evaluator, test_table)
    elif args.user == "gold_sim":
        user = GoldUserSim(error_evaluator, bool_structure_question=args.ask_structure)
    else:
        assert not args.ask_structure, "UserSim with ask_struct=1 is not supported!"
        user = UserSim(error_evaluator)

    if args.err_detector == 'any':
        error_detector = ErrorDetectorProbability(1.1)  # ask any SU
    elif args.err_detector.startswith('prob='):
        prob = float(args.err_detector[5:])
        error_detector = ErrorDetectorProbability(prob)
        print("Error Detector: probability threshold = %.3f" % prob)
        assert args.passes == 1, "Error: For prob-based evaluation, set --passes 1."
    elif args.err_detector.startswith('stddev='):
        stddev = float(args.err_detector[7:])
        error_detector = ErrorDetectorBayesDropout(stddev)
        print("Error Detector: Bayesian Dropout Stddev threshold = %.3f" % stddev)
        print("num passes: %d, dropout rate: %.3f" % (args.passes, args.dropout))
        assert args.passes > 1, "Error: For dropout-based evaluation, set --passes 10."
    elif args.err_detector == "perfect":
        error_detector = ErrorDetectorSim()
        print("Error Detector: using a simulated perfect detector.")
    else:
        raise Exception("Invalid error detector setup %s!" % args.err_detector)

    if args.num_options == 'inf':
        print("WARNING: Unlimited options!")
        num_options = np.inf
    else:
        num_options = int(args.num_options)
        print("num_options: {}".format(num_options))

    print("ask_structure: {}".format(args.ask_structure))
    world_model = WorldModel((bert_config, model_bert, tokenizer, args.max_seq_length, args.num_target_layers),
                             model, num_options, num_passes=args.passes, dropout_rate=args.dropout,
                             bool_structure_question=args.ask_structure)

    print("friendly_agent: {}".format(args.friendly_agent))
    agent = Agent(world_model, error_detector, question_generator, bool_mistake_exit=args.friendly_agent,
                  bool_structure_question=args.ask_structure)

    ## 6. Test
    if not os.path.exists(os.path.dirname(args.output_path)):
        os.mkdir(os.path.dirname(args.output_path))

    if args.job == 'online_learning':
        assert args.data == "online"
        print("## supervision: {}".format(args.supervision))
        print("## update_iter: {}".format(args.update_iter))

        if args.setting == "online_pretrain_1p":
            online_setup_indices = json.load(open(path_wikisql + "online_setup_1p.json"))
        elif args.setting == "online_pretrain_5p":
            online_setup_indices = json.load(open(path_wikisql + "online_setup_5p.json"))
        elif args.setting == "online_pretrain_10p":
            online_setup_indices = json.load(open(path_wikisql + "online_setup_10p.json"))
        else:
            raise Exception("Invalid args.setting={}".format(args.setting))

        if args.supervision == 'full_expert':
            train_data, train_table = load_processed_wikisql_data(path_wikisql, "train")  # processed data
        else:
            train_data, train_table = load_wikisql_data(path_wikisql, mode="train", toy_model=args.toy_model,
                                                        toy_size=args.toy_size, no_hs_tok=True) # raw data

        init_train_indices = set(online_setup_indices["train"])
        init_train_data = [train_data[idx] for idx in init_train_indices if train_data[idx] is not None]
        print("## Update init train size %d " % len(init_train_data))

        online_train_indices = online_setup_indices["online_seed%d" % args.data_seed]
        online_train_data = [train_data[idx] for idx in online_train_indices if train_data[idx] is not None]

        print("## Update online train size %d " % len(online_train_data))
        online_data_loader = torch.utils.data.DataLoader(
            batch_size=1, # must be 1
            dataset=online_train_data,
            shuffle=False,
            num_workers=1, # 4
            collate_fn=lambda x: x  # now dictionary values are not merged!
        )

        def create_new_model(model, model_bert):
            # parser
            def param_reset(m):
                if type(m) in {nn.LSTM, nn.Linear}:
                    m.reset_parameters()
            model.apply(param_reset)
            model.eval()

            # bert
            init_checkpoint = os.path.join(BERT_PT_PATH, 'pytorch_model_{}.bin'.format(args.bert_type))
            model_bert.load_state_dict(torch.load(init_checkpoint, map_location='cpu'))
            print("Reload pre-trained BERT parameters.")
            model_bert.to(device)
            model_bert.eval()

        if args.supervision in ("misp_neil", "misp_neil_pos"):
            subdir = "%s_OP%s_ED%s_SETTING%s_ITER%d_DATASEED%d%s%s/" % (
                args.supervision, args.num_options, args.err_detector, args.setting,
                args.update_iter, args.data_seed,
                ("_FRIENDLY" if args.friendly_agent else ""),
                ("_GoldUser" if args.user == "gold_sim" else ""))
            if not os.path.isdir(os.path.join(model_dir, subdir)):
                os.mkdir(os.path.join(model_dir, subdir))

            if args.auto_iter and args.start_iter == 0 and os.path.exists(args.output_path):
                record_save_path = args.output_path
                print("Loading interaction records from %s..." % record_save_path)
                interaction_records_dict = json.load(open(record_save_path, 'r'))
                args.start_iter = interaction_records_dict['start_iter']
                print("AUTO start_iter = %d." % args.start_iter)

            if args.start_iter > 0:
                print("Loading previous checkpoints at iter {}...".format(args.start_iter))
                start_path_model = os.path.join(model_dir, subdir, '%d' % args.start_iter, 'model_best.pt')
                start_path_model_bert = os.path.join(model_dir, subdir, '%d' % args.start_iter, 'model_bert_best.pt')
                if torch.cuda.is_available():
                    res = torch.load(start_path_model_bert)
                else:
                    res = torch.load(start_path_model_bert, map_location='cpu')
                agent.world_model.model_bert.load_state_dict(res['model_bert'])
                agent.world_model.model_bert.to(device)

                if torch.cuda.is_available():
                    res = torch.load(start_path_model)
                else:
                    res = torch.load(start_path_model, map_location='cpu')
                agent.world_model.semparser.load_state_dict(res['model'])

            online_learning(args.supervision, user, agent, init_train_data, online_data_loader,
                            train_table, dev_data, dev_table, test_data, test_table, args.update_iter,
                            os.path.join(model_dir, subdir), args.output_path, create_new_model,
                            max_seq_length=222, num_target_layers=2, detail=False,
                            st_pos=args.start_iter, end_pos=args.end_iter,
                            cnt_tot=1, path_db=path_wikisql, batch_size=args.bS)

        elif args.supervision.startswith('self_train'):
            subdir = "%s_SETTING%s_ITER%d_DATASEED%d/" % (
                args.supervision, args.setting, args.update_iter,
                args.data_seed)
            if not os.path.isdir(os.path.join(model_dir, subdir)):
                os.mkdir(os.path.join(model_dir, subdir))

            if args.auto_iter and args.start_iter == 0 and os.path.exists(args.output_path):
                record_save_path = args.output_path
                print("Loading interaction records from %s..." % record_save_path)
                interaction_records_dict = json.load(open(record_save_path, 'r'))
                args.start_iter = interaction_records_dict['start_iter']
                print("AUTO start_iter = %d." % args.start_iter)

            if args.start_iter > 0:
                print("Loading previous checkpoints at iter {}...".format(args.start_iter))
                start_path_model = os.path.join(model_dir, subdir, '%d' % args.start_iter, 'model_best.pt')
                start_path_model_bert = os.path.join(model_dir, subdir, '%d' % args.start_iter, 'model_bert_best.pt')
                if torch.cuda.is_available():
                    res = torch.load(start_path_model_bert)
                else:
                    res = torch.load(start_path_model_bert, map_location='cpu')
                agent.world_model.model_bert.load_state_dict(res['model_bert'])
                agent.world_model.model_bert.to(device)

                if torch.cuda.is_available():
                    res = torch.load(start_path_model)
                else:
                    res = torch.load(start_path_model, map_location='cpu')
                agent.world_model.semparser.load_state_dict(res['model'])

            online_learning_self_train(args.supervision, agent, init_train_data, online_data_loader, train_table,
                                       dev_data, dev_table, test_data, test_table, args.update_iter,
                                       os.path.join(model_dir, subdir), args.output_path,
                                       create_new_model, max_seq_length=222, num_target_layers=2, detail=False,
                                       st_pos=args.start_iter, end_pos=args.end_iter,
                                       cnt_tot=1, path_db=path_wikisql, batch_size=args.bS)

        elif args.supervision == "full_expert":
            subdir = "full_expert_SETTING%s_ITER%d_DATASEED%d/" % (
                args.setting, args.update_iter, args.data_seed)
            if not os.path.isdir(os.path.join(model_dir, subdir)):
                os.mkdir(os.path.join(model_dir, subdir))

            assert not args.auto_iter, "--auto_iter is not allowed for Full Expert experiments!"

            online_learning_full_expert(agent, init_train_data, online_train_data, train_table,
                                        dev_data, dev_table, test_data, test_table,
                                        path_wikisql, os.path.join(model_dir, subdir), args.update_iter,
                                        create_new_model, start_idx=args.start_iter, end_idx=args.end_iter,
                                        batch_size=args.bS)

        elif args.supervision in {"bin_feedback", "bin_feedback_expert"}:
            subdir = "%s_SETTING%s_ITER%d_DATASEED%d/" % (
                args.supervision, args.setting, args.update_iter, args.data_seed)
            if not os.path.isdir(os.path.join(model_dir, subdir)):
                os.mkdir(os.path.join(model_dir, subdir))

            if args.auto_iter and args.start_iter == 0 and os.path.exists(args.output_path):
                record_save_path = args.output_path
                print("Loading interaction records from %s..." % record_save_path)
                interaction_records_dict = json.load(open(record_save_path, 'r'))
                args.start_iter = interaction_records_dict['start_iter']
                print("AUTO start_iter = %d." % args.start_iter)

            if args.start_iter > 0:
                print("Loading previous checkpoints at iter {}...".format(args.start_iter))
                start_path_model = os.path.join(model_dir, subdir, '%d' % args.start_iter, 'model_best.pt')
                start_path_model_bert = os.path.join(model_dir, subdir, '%d' % args.start_iter, 'model_bert_best.pt')
                if torch.cuda.is_available():
                    res = torch.load(start_path_model_bert)
                else:
                    res = torch.load(start_path_model_bert, map_location='cpu')
                agent.world_model.model_bert.load_state_dict(res['model_bert'])
                agent.world_model.model_bert.to(device)

                if torch.cuda.is_available():
                    res = torch.load(start_path_model)
                else:
                    res = torch.load(start_path_model, map_location='cpu')
                agent.world_model.semparser.load_state_dict(res['model'])

            online_learning_bin_feedback(args.supervision, agent, init_train_data, online_data_loader, train_table,
                                         dev_data, dev_table, test_data, test_table,
                                         os.path.join(model_dir, subdir), args.output_path,
                                         path_wikisql, args.update_iter, create_new_model,
                                         start_idx=args.start_iter, end_idx=args.end_iter, batch_size=args.bS)

        else:
            assert args.supervision == "misp_neil_perfect"
            subdir = "full_expert_SETTING%s_ITER%d_DATASEED%d/" % (
                args.setting, args.update_iter, args.data_seed)

            if args.start_iter > 0:
                print("Loading previous checkpoints at iter {}...".format(args.start_iter))
                start_path_model = os.path.join(model_dir, subdir, '%d' % args.start_iter, 'model_best.pt')
                start_path_model_bert = os.path.join(model_dir, subdir, '%d' % args.start_iter, 'model_bert_best.pt')
                if torch.cuda.is_available():
                    res = torch.load(start_path_model_bert)
                else:
                    res = torch.load(start_path_model_bert, map_location='cpu')
                agent.world_model.model_bert.load_state_dict(res['model_bert'])
                agent.world_model.model_bert.to(device)

                if torch.cuda.is_available():
                    res = torch.load(start_path_model)
                else:
                    res = torch.load(start_path_model, map_location='cpu')
                agent.world_model.semparser.load_state_dict(res['model'])

            online_learning_misp_perfect(user, agent, online_data_loader, train_table,
                                         args.update_iter, os.path.join(model_dir, subdir),
                                         args.output_path, st_pos=args.start_iter, end_pos=args.end_iter)

    else:
        # test_w_interaction
        test_loader = torch.utils.data.DataLoader(
            batch_size=1,  # must be 1
            dataset=test_data,
            shuffle=False,
            num_workers=1,  # 4
            collate_fn=lambda x: x  # now dictionary values are not merged!
        )

        if args.user == "real":
            with torch.no_grad():
                real_user_interaction(test_loader, test_table, user, agent, tokenizer, args.max_seq_length,
                                      args.num_target_layers, path_wikisql, args.output_path)

        else:
            with torch.no_grad():
                acc_test, results_test, cnt_list, interaction_records = interaction(
                    test_loader, test_table, user, agent, tokenizer, args.max_seq_length, args.num_target_layers,
                    detail=True, path_db=path_wikisql, st_pos=0,
                    dset_name="test" if args.data == "user_study" else args.data)
            print(acc_test)

            # save results for the official evaluation
            path_save_for_evaluation = os.path.dirname(args.output_path)
            save_for_evaluation(path_save_for_evaluation, results_test, args.output_path[args.output_path.index('records_'):])
            json.dump(interaction_records, open(args.output_path, "w"), indent=4)
