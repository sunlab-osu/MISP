import json, pickle
import torch
from SQLNet_model.sqlnet.utils import *
from SQLNet_model.sqlnet.model.seq2sql import Seq2SQL
from SQLNet_model.sqlnet.model.sqlnet import SQLNet, BasicHypothesis
import numpy as np
import time, os
import argparse
import random

import sys
reload(sys)
sys.setdefaultencoding('utf8')

from SQLNet_model.ISQL import ISQLSQLNet
from SQLNet_model.err_detector import *
from SQLNet_model.user_simulator import UserSim, RealUserSQLNet

from interaction_framework.question_gen import QuestionGenerator
# from interaction_framework.user_simulator import RealUser, UserSim
from user_study_utils import *

np.set_printoptions(precision=3)


def load_data(SQL_PATH):
    sql_data = []
    with open(SQL_PATH) as inf:
        for idx, line in enumerate(inf):
            sql = json.loads(line.strip())
            sql_data.append(sql)
    return sql_data


def real_user_interaction(agent, sql_data, table_data, db_path, save_path):
    engine = DBEngine(db_path)

    batch_size = 1
    perm = list(range(len(sql_data)))

    if os.path.isfile(save_path):
        saved_results = json.load(open(save_path, "r"))

        st = saved_results['st']
        qm_one_acc_num = eval(saved_results['qm_one_acc_num'])
        qm_tot_acc_num = saved_results['qm_tot_acc_num']
        exe_tot_acc_num = saved_results['exe_tot_acc_num']
        time_spent = saved_results['time_spent']

        interaction_records = saved_results['interaction_records']
        count_exit = saved_results['count_exit']

    else:
        st = 0
        qm_one_acc_num = 0.0
        qm_tot_acc_num = 0.0
        exe_tot_acc_num = 0.0
        time_spent = 0.

        interaction_records = {}
        count_exit = 0

    while st < len(sql_data):
        ed = st + batch_size if st + batch_size < len(perm) else len(perm)

        q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, raw_data = to_batch_seq(
            sql_data, table_data, perm, st, ed, ret_vis_data=True)
        raw_q_seq = [x[0] for x in raw_data]
        raw_col_seq = [x[1] for x in raw_data]
        query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)
        input_item = [q_seq, col_seq, raw_q_seq, raw_col_seq, col_num]

        os.system('clear')  # clear screen
        print_header(len(sql_data) - st)  # interface header

        print(bcolors.BOLD + "Suppose you are given a table with the following " +
              bcolors.BLUE + "header" + bcolors.ENDC +
              bcolors.BOLD + ":" + bcolors.ENDC)
        agent.user_sim.show_table(table_ids[0])  # print table

        print(bcolors.BOLD + "\nAnd you want to answer the following " +
              bcolors.PINK + "question" + bcolors.ENDC +
              bcolors.BOLD + " based on this table:" + bcolors.ENDC)
        print(bcolors.PINK + bcolors.BOLD + raw_data[0][0] + bcolors.ENDC + "\n")

        print(bcolors.BOLD + "To help you get the answer automatically,"
                             " the system has the following yes/no questions for you."
                             "\n(When no question prompts, please " +
                             bcolors.GREEN + "continue" + bcolors.ENDC +
                             bcolors.BOLD + " to the next case)\n" + bcolors.ENDC)

        start_signal = raw_input(bcolors.BOLD + "Ready? please press '" +
                                 bcolors.GREEN + "Enter" + bcolors.ENDC + bcolors.BOLD + "' to start!" + bcolors.ENDC)
        while start_signal != "":
            start_signal = raw_input(bcolors.BOLD + "Ready? please press '" +
                                     bcolors.GREEN + "Enter" + bcolors.ENDC + bcolors.BOLD + "' to start!" + bcolors.ENDC)

        start_time = time.time()
        init_hyp = agent.decode(input_item, bool_verbal=False)[0]

        hyp, bool_exit = agent.error_detection(input_item, query_gt[0], init_hyp, bool_verbal=False)
        print("\nPredicted SQL: {}\n".format(hyp.sql))

        per_qm_one_acc_num, per_qm_tot_acc_num, per_exe_tot_acc_num = agent.evaluation(
            raw_data, [hyp.sql_i], query_gt, table_ids, engine)
        per_time_spent = time.time() - start_time
        time_spent += per_time_spent
        print("Your time spent: %.3f sec" % per_time_spent)

        if bool_exit:
            count_exit += 1

        # post survey
        print("-" * 50)
        print("Post-study Survey: ")
        bool_unclear = raw_input("Is the " + bcolors.BOLD + bcolors.PINK + "initial question" +
                                 bcolors.ENDC + " clear?\nPlease enter y/n: ")
        while bool_unclear not in {'y', 'n'}:
            bool_unclear = raw_input("Is the " + bcolors.BOLD + bcolors.PINK + "initial question" +
                                     bcolors.ENDC + " clear?\nPlease enter y/n: ")
        print("-" * 50)

        record = {'nl': raw_data[0][0], 'true_sql': raw_data[0][-1], 'true_sql_i': "{}".format(query_gt[0]),
                  'init_sql': "{}".format(init_hyp.sql), 'init_sql_i': "{}".format(init_hyp.sql_i),
                  'sql': "{}".format(hyp.sql), 'sql_i': "{}".format(hyp.sql_i),
                  'dec_seq': "{}".format(hyp.dec_seq), 'tag_seq': "{}".format(hyp.tag_seq),
                  'logprob': "{}".format(hyp.logprob),
                  'lx_correct': per_qm_tot_acc_num, 'x_correct': per_exe_tot_acc_num,
                  'exit': bool_exit, 'waste_q_counter': agent.user_sim.waste_q_counter,
                  'necessary_q_counter': agent.user_sim.necessary_q_counter,
                  'questioned_indices': agent.user_sim.questioned_indices,
                  'questioned_tags': "{}".format(agent.user_sim.questioned_tags),
                  'per_time_spent': per_time_spent, 'bool_unclear': bool_unclear}
        if isinstance(agent.error_detector, ErrorDetectorBayDropout):
            record.update({'logprob_list': "{}".format(hyp.logprob_list),
                           'test_tag_seq': "{}".format(hyp.test_tag_seq)})
        # interaction_records.append(record)
        interaction_records[st] = record

        qm_one_acc_num += per_qm_one_acc_num
        qm_tot_acc_num += per_qm_tot_acc_num
        exe_tot_acc_num += per_exe_tot_acc_num

        st = ed # the next start

        print("Saving records...")
        json.dump({'interaction_records': interaction_records,
                   'qm_one_acc_num': "{}".format(list(qm_one_acc_num)), 'qm_tot_acc_num': qm_tot_acc_num,
                   'exe_tot_acc_num': exe_tot_acc_num, 'st': st, 'time_spent': time_spent,
                   'count_exit': count_exit},
                  open(save_path, "w"), indent=4)

        end_signal = raw_input(bcolors.GREEN + bcolors.BOLD +
                               "Next? Press 'Enter' to continue, Ctrl+C to quit." + bcolors.ENDC)
        if end_signal != "":
            return

    print(bcolors.RED + bcolors.BOLD + "Congratulations! You have completed all your task!" + bcolors.ENDC)
    print("Your average time spent: {:.3f}".format((time_spent / len(interaction_records))))
    print("You exited %d times." % count_exit)


def interaction(agent, sql_data, table_data, db_path, wikisql_sample_ids=None, bool_interaction=True):
    engine = DBEngine(db_path)

    batch_size = 1
    perm = list(range(len(sql_data)))
    st = 0
    qm_one_acc_num = 0.0
    qm_tot_acc_num = 0.0
    exe_tot_acc_num = 0.0
    time_spent = 0.

    interaction_records = []
    count_exit = 0
    while st < len(sql_data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)
        if wikisql_sample_ids is not None and st not in wikisql_sample_ids:
            st = ed
            continue

        q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, raw_data = to_batch_seq(
            sql_data, table_data, perm, st, ed, ret_vis_data=True)
        raw_q_seq = [x[0] for x in raw_data]
        raw_col_seq = [x[1] for x in raw_data]
        query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)
        input_item = [q_seq, col_seq, raw_q_seq, raw_col_seq, col_num]

        start_time = time.time()
        print("\n" + "#" * 50)
        print("NL input: {}\nTrue SQL: {}".format(raw_data[0][0], raw_data[0][-1]))

        hyp = agent.decode(input_item, bool_verbal=False)[0]
        print("## time spent per decode: {:.3f}".format(time.time() - start_time))

        print("-" * 50 + "\nBefore interaction: \ninitial SQL: {}".format(hyp.sql))
        BasicHypothesis.print_hypotheses([hyp])
        print("initial evaluation: ")
        per_qm_one_acc_num, per_qm_tot_acc_num, per_exe_tot_acc_num = agent.evaluation(
            raw_data, [hyp.sql_i], query_gt, table_ids, engine)

        if not bool_interaction:
            record = {'nl': raw_data[0][0], 'true_sql': raw_data[0][-1], 'true_sql_i': "{}".format(query_gt[0]),
                      'sql': "{}".format(hyp.sql), 'sql_i': "{}".format(hyp.sql_i),
                      'dec_seq': "{}".format(hyp.dec_seq), 'tag_seq': "{}".format(hyp.tag_seq),
                      'logprob': "{}".format(hyp.logprob),
                      'lx_correct': per_qm_tot_acc_num, 'x_correct': per_exe_tot_acc_num}
            if isinstance(agent.error_detector, ErrorDetectorBayDropout):
                record.update({'logprob_list': "{}".format(hyp.logprob_list),
                               'test_tag_seq': "{}".format(hyp.test_tag_seq)})
            interaction_records.append(record)
            st = ed

            qm_one_acc_num += per_qm_one_acc_num
            qm_tot_acc_num += per_qm_tot_acc_num
            exe_tot_acc_num += per_exe_tot_acc_num

            continue

        hyp, bool_exit = agent.error_detection(input_item, query_gt[0], hyp, bool_verbal=False)
        print("-" * 50 + "\nAfter interaction:\nfinal SQL: {}".format(hyp.sql))
        BasicHypothesis.print_hypotheses([hyp])
        print("final evaluation: ")
        per_qm_one_acc_num, per_qm_tot_acc_num, per_exe_tot_acc_num = agent.evaluation(
            raw_data, [hyp.sql_i], query_gt, table_ids, engine)
        time_spent += (time.time() - start_time)

        if bool_exit:
            count_exit += 1

        record = {'nl': raw_data[0][0], 'true_sql': raw_data[0][-1], 'true_sql_i': "{}".format(query_gt[0]),
                 'sql': "{}".format(hyp.sql), 'sql_i': "{}".format(hyp.sql_i),
                 'dec_seq': "{}".format(hyp.dec_seq), 'tag_seq': "{}".format(hyp.tag_seq),
                  'logprob': "{}".format(hyp.logprob),
                 'lx_correct': per_qm_tot_acc_num, 'x_correct': per_exe_tot_acc_num,
                 'exit': bool_exit, 'waste_q_counter': agent.user_sim.waste_q_counter,
                 'necessary_q_counter': agent.user_sim.necessary_q_counter,
                 'questioned_indices': agent.user_sim.questioned_indices,
                  'questioned_tags': "{}".format(agent.user_sim.questioned_tags)}
        if isinstance(agent.error_detector, ErrorDetectorBayDropout):
            record.update({'logprob_list': "{}".format(hyp.logprob_list),
                           'test_tag_seq': "{}".format(hyp.test_tag_seq)})
        interaction_records.append(record)

        qm_one_acc_num += per_qm_one_acc_num
        qm_tot_acc_num += per_qm_tot_acc_num
        exe_tot_acc_num += per_exe_tot_acc_num

        st = ed

    qm_one_acc = qm_one_acc_num / len(sql_data)
    qm_tot_acc = qm_tot_acc_num / len(sql_data)
    exe_tot_acc = exe_tot_acc_num / len(sql_data)

    if not bool_interaction:
        return qm_one_acc, qm_tot_acc, exe_tot_acc, interaction_records

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

    return qm_one_acc, qm_tot_acc, exe_tot_acc, interaction_records


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--toy', action='store_true',
            help='If set, use small data; used for fast debugging.')
    parser.add_argument('--ca', action='store_true',
            help='Use conditional attention.')
    parser.add_argument('--dataset', type=int, default=0,
            help='0: original dataset, 1: re-split dataset')
    parser.add_argument('--rl', action='store_true',
            help='Use RL for Seq2SQL.')
    parser.add_argument('--baseline', action='store_true',
            help='If set, then test Seq2SQL model; default is SQLNet_model model.')
    parser.add_argument('--train_emb', action='store_true',
            help='Use trained word embedding for SQLNet_model.')
    parser.add_argument('--dr', type=float, default=0.3,
            help='Dropout for SQLNet_model training.')

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

    # model temperature
    parser.add_argument('--temperature', default=0, type=int, help='Use softmax temperature.')

    args = parser.parse_args()

    assert args.train_emb == True

    # Seeds for random number generation
    seed = args.seed
    print("## seed: %d" % seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed)

    print("## temperature: {}".format(args.temperature))

    N_word=300
    B_word=42
    if args.toy:
        USE_SMALL=True
        GPU=True
        BATCH_SIZE=15
    else:
        USE_SMALL=False
        GPU=True
        BATCH_SIZE=64
    TEST_ENTRY=(True, True, True)  # (AGG, SEL, COND)

    sql_data, table_data, val_sql_data, val_table_data, \
            test_sql_data, test_table_data, \
            TRAIN_DB, DEV_DB, TEST_DB = load_dataset(
                    args.dataset, use_small=USE_SMALL, data_dir='SQLNet_model/')

    word_emb = load_word_emb('glove/glove.%dB.%dd.txt'%(B_word,N_word),
        load_used=True, use_small=USE_SMALL, data_dir='SQLNet_model/') # load_used can speed up loading

    if args.baseline:
        model = Seq2SQL(word_emb, N_word=N_word, gpu=GPU, trainable_emb=True)
    else:
        model = SQLNet(word_emb, N_word=N_word, use_ca=args.ca, gpu=GPU,
                trainable_emb = True, dr=args.dr, temperature=args.temperature)
    model.eval()

    print("Creating ISQL agent...")
    question_generator = QuestionGenerator()
    err_evaluator = ErrorDetectorEvaluatorSQLNet()

    if args.real_user:
        user_simulator = RealUserSQLNet(err_evaluator, test_table_data)
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

    agent = ISQLSQLNet(model, error_detector, question_generator, user_simulator, num_options,
                       bool_structure_rev=args.structure, num_passes=args.passes, dropout_rate=args.dropout)

    if args.temperature:
        prefix = 'calibr_'
    else:
        prefix = ''
    if args.train_emb:
        agg_m, sel_m, cond_m, agg_e, sel_e, cond_e = best_model_name(
            args, data_dir='SQLNet_model/', prefix=prefix)
        print "Loading from %s"%agg_m
        model.agg_pred.load_state_dict(torch.load(agg_m))
        print "Loading from %s"%sel_m
        model.sel_pred.load_state_dict(torch.load(sel_m))
        print "Loading from %s"%cond_m
        model.cond_pred.load_state_dict(torch.load(cond_m))
        print "Loading from %s"%agg_e
        model.agg_embed_layer.load_state_dict(torch.load(agg_e))
        print "Loading from %s"%sel_e
        model.sel_embed_layer.load_state_dict(torch.load(sel_e))
        print "Loading from %s"%cond_e
        model.cond_embed_layer.load_state_dict(torch.load(cond_e))
    else:
        agg_m, sel_m, cond_m = best_model_name(
            args, data_dir='SQLNet_model', prefix=prefix)
        print "Loading from %s"%agg_m
        model.agg_pred.load_state_dict(torch.load(agg_m))
        print "Loading from %s"%sel_m
        model.sel_pred.load_state_dict(torch.load(sel_m))
        print "Loading from %s"%cond_m
        model.cond_pred.load_state_dict(torch.load(cond_m))

    wikisql_sample_ids = None #pickle.load(open("wikisql_analyze_sample_ids.pkl", "rb"))
    # filename = '/home/yao.470/Projects2/interactive-SQL/SQLNet_model/interaction/records_' + args.output_path + '.json'

    if args.data == 'dev':
        assert user_simulator.user_type == "sim"
        qm_one_acc, qm_tot_acc, exe_tot_acc, interaction_records = interaction(
            agent, val_sql_data, val_table_data, DEV_DB, wikisql_sample_ids=wikisql_sample_ids)
        print "Dev acc_qm: %s;\n  breakdown on (agg, sel, where): %s\nDec execution acc: %s" % (
            qm_tot_acc, qm_one_acc, exe_tot_acc)
        json.dump(interaction_records, open(args.output_path, 'w'), indent=4)

    elif args.data == 'test':
        assert user_simulator.user_type == "sim"
        qm_one_acc, qm_tot_acc, exe_tot_acc, interaction_records = interaction(
            agent, test_sql_data, test_table_data, TEST_DB)
        print "Test acc_qm: %s;\n  breakdown on (agg, sel, where): %s\nTest execution acc: %s" % (
            qm_tot_acc, qm_one_acc, exe_tot_acc)
        json.dump(interaction_records, open(args.output_path, 'w'), indent=4)

    elif args.data == "user_study":
        # filename = "/home/yao.470/Projects2/interactive-SQL/SQLNet_model/user_study/" \
        #            "records_%s.json" % args.output_path

        sampled_ids = json.load(open("SQLNet_model/data/user_study_ids.json", "r"))
        sql_data = [test_sql_data[idx] for idx in sampled_ids]

        if user_simulator.user_type == "real":
            real_user_interaction(agent, sql_data, test_table_data, TEST_DB, args.output_path)
        else:
            qm_one_acc, qm_tot_acc, exe_tot_acc, interaction_records = interaction(
                agent, sql_data, test_table_data, TEST_DB, bool_interaction=True)
            print "Test acc_qm: %s;\n  breakdown on (agg, sel, where): %s\nTest execution acc: %s" % (
                qm_tot_acc, qm_one_acc, exe_tot_acc)
            json.dump(interaction_records, open(args.output_path, 'w'), indent=4)


