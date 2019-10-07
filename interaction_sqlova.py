# Copyright 2019-present NAVER Corp.
# Apache License v2.0

# Wonseok Hwang
# Sep30, 2018
#
# Adapted for interaction only. @author: Ziyu Yao
import os, sys, argparse, re, json, pickle
from matplotlib.pylab import *
import torch.nn as nn
import torch
import torch.nn.functional as F
import random as python_random
# import torchvision.datasets as dsets
import numpy as np
import time

import SQLova_model.bert.tokenization as tokenization
from SQLova_model.bert.modeling import BertConfig, BertModel
from SQLova_model.sqlova.utils.utils_wikisql import *
from SQLova_model.sqlova.model.nl2sql.wikisql_models import *
from SQLova_model.sqlnet.dbengine import DBEngine
from SQLova_model.ISQL import ISQLSQLova
from SQLova_model.err_detector import ErrorDetectorEvaluatorSQLova, ErrorDetectorProbability,\
    ErrorDetectorBayDropout, ErrorDetectorLR, ErrorDetectorRespectiveProbability
from interaction_framework.question_gen import QuestionGenerator
# from interaction_framework.user_simulator import RealUser, UserSim
from SQLova_model.user_simulator import UserSim, RealUserSQLova

from user_study_utils import *


np.set_printoptions(precision=3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def construct_hyper_param(parser):
    parser.add_argument("--bS", default=1, type=int, help="Batch size")
    parser.add_argument("--model_type", default='Seq2SQL_v1', type=str,
                        help="Type of model.")

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

    # for interaction
    parser.add_argument('--num_options', type=str, default='3', help='#of options.')
    parser.add_argument('--real_user', action='store_true', help='Real user interaction.')
    parser.add_argument('--err_detector', type=str, default='any', help='the error detector: prob-x.')
    parser.add_argument('--structure', type=int, default=0, help='Whether to change to kw structure.')
    parser.add_argument('--output_path', type=str, default='temp', help='Where to save outputs.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate for uncertainty analysis.')
    parser.add_argument('--passes', type=int, default=1, help='Number of decoding passes.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for dropout evaluation.')

    # test data
    parser.add_argument('--data', default='dev', choices=['dev', 'test', 'user_study'], help='which dataset to test.')

    # temperature
    parser.add_argument('--temperature', default=0, type=int, help='Set to 1 for using softmax temperature.')

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
    print("## seed: %d" % args.seed)
    python_random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(args.seed)

    print("## temperature: %d" % args.temperature)

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
    model = Seq2SQL_v1(args.iS, args.hS, args.lS, args.dr, n_cond_ops, n_agg_ops, temperature=args.temperature)
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


def real_user_interaction(data_loader, data_table, agent, model_bert, bert_config, tokenizer,
                          max_seq_length, num_target_layers, path_db, save_path):
    dset_name = "test"

    if os.path.isfile(save_path):
        saved_results = json.load(open(save_path, "r"))
        interaction_records = saved_results['interaction_records']
        count_exit = saved_results['count_exit']
        time_spent = saved_results['time_spent']
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
        time_spent = 0.
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
            = get_wemb_bert(bert_config, model_bert, tokenizer, nlu_t, hds, max_seq_length,
                            num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers)
        g_wvi = get_g_wvi_bert_from_g_wvi_corenlp(t_to_tt_idx, g_wvi_corenlp)
        g_wv_str, g_wv_str_wp = convert_pr_wvi_to_string(g_wvi, nlu_t, nlu_tt, tt_to_t_idx, nlu)

        os.system('clear')  # clear screen
        print_header(len(data_loader.dataset) - cnt)  # interface header

        print(bcolors.BOLD + "Suppose you are given a table with the following " +
              bcolors.BLUE + "header" + bcolors.ENDC +
              bcolors.BOLD + ":" + bcolors.ENDC)
        agent.user_sim.show_table(t[0]['table_id'])  # print table

        print(bcolors.BOLD + "\nAnd you want to answer the following " +
              bcolors.PINK + "question" + bcolors.ENDC +
              bcolors.BOLD + " based on this table:" + bcolors.ENDC)
        print(bcolors.PINK + bcolors.BOLD + t[0]['question'] + bcolors.ENDC + "\n")

        print(bcolors.BOLD + "To help you get the answer automatically,"
                             " the system has the following yes/no questions for you."
                             "\n(When no question prompts, please " +
              bcolors.GREEN + "continue" + bcolors.ENDC +
              bcolors.BOLD + " to the next case)\n" + bcolors.ENDC)

        # TODO: debug only
        # print("True SQL: {}".format(g_sql_q[0]))

        start_signal = input(bcolors.BOLD + "Ready? please press '" +
                             bcolors.GREEN + "Enter" + bcolors.ENDC + bcolors.BOLD + "' to start!" + bcolors.ENDC)
        while start_signal != "":
            start_signal = input(bcolors.BOLD + "Ready? please press '" +
                                 bcolors.GREEN + "Enter" + bcolors.ENDC + bcolors.BOLD + "' to start!" + bcolors.ENDC)

        start_time = time.time()
        # init decode
        if isinstance(agent.error_detector, ErrorDetectorBayDropout):
            input_item = [tb, nlu_t, nlu, hds]
        else:
            input_item = [wemb_n, l_n, wemb_h, l_hpu, l_hs, tb, nlu_t, nlu_tt, tt_to_t_idx, nlu]
        init_hyp = agent.decode(input_item, dec_beam_size=1, bool_verbal=False)[0]

        # interaction
        hyp, bool_exit = agent.error_detection(input_item, sql_i[0], init_hyp, bool_verbal=False)
        print("\nPredicted SQL: {}\n".format(hyp.sql))

        per_time_spent = time.time() - start_time
        time_spent += per_time_spent
        print("Your time spent: %.3f sec" % per_time_spent)

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
                  'confirmed_indices': "{}".format(agent.user_sim.confirmed_indices),
                  'lx_correct': int(sum(cnt_lx1_list)), 'x_correct': int(sum(cnt_x1_list)),
                  'exit': bool_exit, 'waste_q_counter': agent.user_sim.waste_q_counter,
                  'necessary_q_counter': agent.user_sim.necessary_q_counter,
                  'questioned_indices': agent.user_sim.questioned_indices,
                  'questioned_tags': "{}".format(agent.user_sim.questioned_tags),
                  'per_time_spent': per_time_spent, 'bool_unclear':bool_unclear}
        if isinstance(agent.error_detector, ErrorDetectorBayDropout):
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
                   'st': cnt, 'time_spent': time_spent,
                   'count_exit': count_exit},
                   open(save_path, "w"), indent=4)

        end_signal = input(bcolors.GREEN + bcolors.BOLD +
                               "Next? Press 'Enter' to continue, Ctrl+C to quit." + bcolors.ENDC)
        if end_signal != "":
            return

    print(bcolors.RED + bcolors.BOLD + "Congratulations! You have completed all your task!" + bcolors.ENDC)
    print("Your average time spent: {:.3f}".format((time_spent / len(interaction_records))))
    print("You exited %d times." % count_exit)


def interaction(data_loader, data_table, agent, model_bert, bert_config, tokenizer,
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
            = get_wemb_bert(bert_config, model_bert, tokenizer, nlu_t, hds, max_seq_length,
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
                                        "questioned_indices": []})
            continue

        print("\n" + "#" * 50)
        print("NL input: {}\nTrue SQL: {}".format(t[0]['question'], g_sql_q[0]))
        # init decode
        if isinstance(agent.error_detector, ErrorDetectorBayDropout):
            input_item = [tb, nlu_t, nlu, hds]
        else:
            input_item = [wemb_n, l_n, wemb_h, l_hpu, l_hs, tb, nlu_t, nlu_tt, tt_to_t_idx, nlu]
        hyp = agent.decode(input_item, dec_beam_size=1, bool_verbal=False)[0]
        print("## time spent per decode: {:.3f}".format(time.time() - start_time))

        print("-" * 50 + "\nBefore interaction: \ninitial SQL: {}".format(hyp.sql))
        BasicHypothesis.print_hypotheses([hyp])
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
            [g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi, sql_i], engine, tb)

        if not bool_interaction:
            record = {'nl': t[0]['question'], 'true_sql': g_sql_q[0], 'true_sql_i': "{}".format(sql_i[0]),
                      'sql': "{}".format(hyp.sql), 'sql_i': "{}".format(hyp.sql_i),
                      'dec_seq': "{}".format(hyp.dec_seq), 'tag_seq': "{}".format(hyp.tag_seq),
                      'logprob': "{}".format(hyp.logprob),
                      'lx_correct': int(sum(cnt_lx1_list)), 'x_correct': int(sum(cnt_x1_list))}
            if isinstance(agent.error_detector, ErrorDetectorBayDropout):
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
        hyp, bool_exit = agent.error_detection(input_item, sql_i[0], hyp, bool_verbal=False)
        print("-" * 50 + "\nAfter interaction:\nfinal SQL: {}".format(hyp.sql))
        BasicHypothesis.print_hypotheses([hyp])
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
        [pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi, pr_sql_i], [g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi, sql_i], engine, tb)

        # save interaction records
        record = {'nl': t[0]['question'], 'true_sql': g_sql_q[0], 'true_sql_i': "{}".format(sql_i[0]),
                  'sql': hyp.sql, 'sql_i': "{}".format(hyp.sql_i),
                  'dec_seq': "{}".format(hyp.dec_seq), 'tag_seq': "{}".format(hyp.tag_seq),
                  'logprob': "{}".format(hyp.logprob), #test time without dropout
                  'confirmed_indices': "{}".format(agent.user_sim.confirmed_indices),
                  'lx_correct': int(sum(cnt_lx1_list)), 'x_correct': int(sum(cnt_x1_list)),
                  'exit': bool_exit, 'waste_q_counter': agent.user_sim.waste_q_counter,
                  'necessary_q_counter': agent.user_sim.necessary_q_counter,
                  'questioned_indices': agent.user_sim.questioned_indices,
                  'questioned_tags': "{}".format(agent.user_sim.questioned_tags)}
        if isinstance(agent.error_detector, ErrorDetectorBayDropout):
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
    q_count = agent.necessary_q_counter + agent.waste_q_counter
    dist_q_count = sum([len(set(item['questioned_indices'])) for item in interaction_records])
    print("#questions: {}, #questions per example: {:.3f} (exclude options: {:.3f}).".format(
        q_count, q_count * 1.0 / len(interaction_records), dist_q_count * 1.0 / len(interaction_records)))
    # print("#necessary questions: {} ({:.3f}%), #wasteful questions: {} ({:.3f}%).".format(
    #     agent.necessary_q_counter, agent.necessary_q_counter * 100.0 / q_count,
    #     agent.waste_q_counter, agent.waste_q_counter * 100.0 / q_count))
    print("#exit: {}".format(count_exit))
    # interaction_records.append({'necessary_q': agent.necessary_q_counter, 'wasteful_q': agent.waste_q_counter})
    print("Avg time spent: {:.3f}".format((time_spent / len(interaction_records))))

    acc = [ave_loss, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_wvi, acc_wv, acc_lx, acc_x]
    return acc, results, cnt_list, interaction_records


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

    # path_save_for_evaluation = '/home/yao.470/Projects2/interactive-SQL/SQLova_model/interaction/'
    model_path = 'SQLova_model/checkpoints_0416/'

    ## 3. Load data
    # train_data, train_table, dev_data, dev_table, train_loader, dev_loader = get_data(path_wikisql, args)
    if args.data == "user_study":
        test_data, test_table = load_wikisql_data(path_wikisql, mode="test", toy_model=args.toy_model,
                                                  toy_size=args.toy_size, no_hs_tok=True)
        sampled_ids = json.load(open("SQLova_model/download/data/user_study_ids.json", "r"))
        test_data = [test_data[idx] for idx in sampled_ids]
    else:
        test_data, test_table = load_wikisql_data(path_wikisql, mode=args.data, toy_model=args.toy_model, toy_size=args.toy_size, no_hs_tok=True)
    test_loader = torch.utils.data.DataLoader(
        batch_size=1, # must be 1
        dataset=test_data,
        shuffle=False,
        num_workers=1, # 4
        collate_fn=lambda x: x  # now dictionary values are not merged!
    )

    # 4. Build & Load models
    model_prefix = ""
    if args.temperature:
        model_prefix = "calibr_"
    model, model_bert, tokenizer, bert_config = get_models(args, BERT_PT_PATH, trained=True,
                                                           path_model_bert=model_path + "model_bert_best.pt",
                                                           path_model=model_path + "{}model_best.pt".format(model_prefix))
    model.eval()
    model_bert.eval()

    ## 5. Create ISQL agent
    print("Creating ISQL agent...")
    question_generator = QuestionGenerator()
    err_evaluator = ErrorDetectorEvaluatorSQLova()
    # user_simulator = RealUser(err_evaluator) if args.real_user else UserSim(err_evaluator)
    if args.real_user:
        user_simulator = RealUserSQLova(err_evaluator, test_table)
    else:
        user_simulator = UserSim(err_evaluator)

    if args.err_detector == 'any':
        error_detector = ErrorDetectorProbability(1.1)  # ask any SU
    elif args.err_detector.startswith('prob='):
        prob = float(args.err_detector[5:])
        error_detector = ErrorDetectorProbability(prob)
        print("Error Detector: probability threshold = %.3f" % prob)
        assert args.passes == 1, "Error: For prob-based evaluation, set --passes 1."
    elif args.err_detector.startswith('stddev='):
        stddev = float(args.err_detector[7:])
        error_detector = ErrorDetectorBayDropout(stddev)
        print("Error Detector: Bayesian Dropout Stddev threshold = %.3f" % stddev)
        print("num passes: %d, dropout rate: %.3f" % (args.passes, args.dropout))
        assert args.passes > 1, "Error: For dropout-based evaluation, set --passes 10."
    else:
        raise Exception("Invalid error detector setup %s!" % args.err_detector)

    if args.num_options == 'inf':
        print("WARNING: Unlimited options!")
        num_options = np.inf
    else:
        num_options = int(args.num_options)
        print("num_options: {}".format(num_options))

    print("bool_structure_rev: {}".format(args.structure))

    agent = ISQLSQLova((bert_config, model_bert, tokenizer, args.max_seq_length, args.num_target_layers),
                       model, error_detector, question_generator, user_simulator, num_options,
                       bool_structure_rev=args.structure, num_passes=args.passes, dropout_rate=args.dropout)

    ## 6. Test
    if args.real_user:
        # filename = '/home/yao.470/Projects2/interactive-SQL/SQLova_model/user_study/records_' +\
        #            args.output_path + ".json"
        with torch.no_grad():
            real_user_interaction(test_loader, test_table, agent, model_bert, bert_config,
                                  tokenizer, args.max_seq_length, args.num_target_layers,
                                  path_wikisql, args.output_path)

    else:
        wikisql_sample_ids = None #pickle.load(open("wikisql_analyze_sample_ids.pkl", "rb"))
        with torch.no_grad():
            acc_test, results_test, cnt_list, interaction_records = interaction(
                test_loader, test_table, agent, model_bert, bert_config,
                tokenizer, args.max_seq_length, args.num_target_layers,
                detail=True, path_db=path_wikisql, st_pos=0, dset_name="test" if args.data == "user_study" else args.data,
                wikisql_sample_ids=wikisql_sample_ids, bool_interaction=True)
        print(acc_test)

        # save results for the official evaluation
        path_save_for_evaluation = os.path.dirname(args.output_path)
        save_for_evaluation(path_save_for_evaluation, results_test, args.output_path[args.output_path.index('records_'):])
        json.dump(interaction_records, open(args.output_path, "w"), indent=4)
