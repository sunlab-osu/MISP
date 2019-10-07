import json
import torch
import time, os
import argparse
import numpy as np
np.set_printoptions(precision=3)
import random

from syntaxSQL.utils import *
from syntaxSQL.supermodel import SuperModel
from syntaxSQL.preprocess_train_dev_data import get_table_dict
from syntaxSQL.agent import Agent
from syntaxSQL.world_model import WorldModel
from syntaxSQL.error_detector import ErrorDetectorProbability, ErrorDetectorBayesDropout
from syntaxSQL.environment import UserSim, RealUser, ErrorEvaluator
from MISP_SQL.question_gen import QuestionGenerator
from MISP_SQL.utils import Hypothesis
from user_study_utils import *


def interaction(user, agent, data, output_path, bool_interaction=True):
    f = open(output_path, "w")  # .txt

    results = []
    count_exit = 0
    count_correct = 0.
    print "Interaction starts...\n"
    for item_idx, item in enumerate(data):
        print("\n" + "#" * 50)
        print("NL input: {}\nTrue SQL: {}".format(item['question'].encode('utf-8'), item['query']))

        # init decode
        start_time = time.time()
        hyp = agent.world_model.decode(item, bool_verbal=False)[0]
        print("\n##Before interaction:\ninitial SQL: {}".format(hyp.sql))
        Hypothesis.print_hypotheses([hyp])
        print("## time spent per decode: {:.3f}".format((time.time() - start_time)))

        print("initial evaluation: ")
        hardness, bool_err, exact_score, partial_scores = agent.evaluation(item, hyp, bool_verbal=True)

        if not bool_interaction:
            record = {'nl': item['question'], 'true_sql': item['query'],
                      'current_sql': "{}".format(hyp.current_sql),
                      'history': "{}".format(hyp.history[0]), 'dec_seq': "{}".format(hyp.dec_seq),
                      'tag_seq': "{}".format(hyp.tag_seq), 'sql': hyp.sql,
                      'logprob': "{}".format(hyp.logprob),
                      'exact_score': exact_score,
                      'partial_scores': "{}".format(partial_scores)}
            if isinstance(agent.error_detector, ErrorDetectorBayesDropout):
                record.update({'logprob_list': "{}".format(hyp.logprob_list),
                               'test_tag_seq': "{}".format(hyp.test_tag_seq)})
            results.append(record)
            f.write(hyp.sql + "\n")
            count_correct += exact_score
            continue

        hyp, bool_exit = agent.interactive_parsing_session(user, item, item['sql'], hyp, bool_verbal=False)
        print("\n##After interaction:\nfinal SQL: {}\nfinal evaluation: ".format(hyp.sql))
        Hypothesis.print_hypotheses([hyp])
        hardness, bool_err, exact_score, partial_scores = agent.evaluation(item, hyp, bool_verbal=True)

        record = {'nl': item['question'], 'true_sql': item['query'], 'current_sql': "{}".format(hyp.current_sql),
                  'history': "{}".format(hyp.history[0]), 'dec_seq': "{}".format(hyp.dec_seq),
                  'tag_seq': "{}".format(hyp.tag_seq), 'sql': hyp.sql,
                  'logprob': "{}".format(hyp.logprob),
                  'exit': bool_exit, 'q_counter': user.q_counter,
                  'questioned_indices': user.questioned_pointers,
                  'questioned_tags': "{}".format(user.questioned_tags),
                  'exact_score': exact_score,
                  'partial_scores': "{}".format(partial_scores)}
        if isinstance(agent.error_detector, ErrorDetectorBayesDropout):
            record.update({'logprob_list': "{}".format(hyp.logprob_list),
                           'test_tag_seq': "{}".format(hyp.test_tag_seq)})
        results.append(record)
        count_correct += exact_score

        if bool_exit:
            count_exit += 1
        f.write(hyp.sql + "\n")

    print("EM Accuracy: %.3f" % (count_correct / len(results)))
    if not bool_interaction:
        f.close()

        f_json = open(output_path[:-4] + ".json", "w")  # .json
        json.dump(results, f_json, indent=4)
        f_json.close()

        return

    # stats
    q_count = sum([item['q_counter'] for item in results])
    dist_q_count = sum([len(set(item['questioned_indices'])) for item in results])
    print("#questions: {}, #questions per example: {:.3f}, (exclude options: {:.3f}).".format(
        q_count, q_count * 1.0 / len(data), dist_q_count * 1.0 / len(data)))
    print("#exit: {}".format(count_exit))

    f.close()

    f_json = open(output_path[:-4] + ".json", "w")  # .json
    json.dump(results, f_json, indent=4)
    f_json.close()


def real_user_interaction(user, agent, data, output_path):
    save_path = output_path[:-4] + ".json"

    if os.path.isfile(save_path):
        saved_results = json.load(open(save_path, "r"))
        st = saved_results['st']
        results = saved_results['results']
        count_exit = saved_results['count_exit']
        time_spent = saved_results['time_spent']

    else:
        st = 0
        results = {}
        count_exit = 0
        time_spent = 0.

    while st < len(data):
        item = data[st]

        os.system('clear')  # clear screen
        print_header(len(data) - st, bool_table_color=True)  # interface header

        print(bcolors.BOLD + "Suppose you are given some tables with the following " +
              bcolors.BLUE + "headers" + bcolors.ENDC +
              bcolors.BOLD + ":" + bcolors.ENDC)
        user.show_table(item['db_id'])  # print table

        print(bcolors.BOLD + "\nAnd you want to answer the following " +
              bcolors.PINK + "question" + bcolors.ENDC +
              bcolors.BOLD + " based on this table:" + bcolors.ENDC)
        print(bcolors.PINK + bcolors.BOLD + item['question'].encode('utf-8') + bcolors.ENDC + "\n")
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
        # init decode
        init_hyp = agent.world_model.decode(item, bool_verbal=False)[0]

        hyp, bool_exit = agent.interactive_parsing_session(user, item, item['sql'], init_hyp, bool_verbal=False)
        print("\nPredicted SQL: {}".format(hyp.sql))
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

        record = {'nl': item['question'], 'true_sql': item['query'],
                  'init_current_sql': "{}".format(init_hyp.current_sql), 'init_sql': init_hyp.sql,
                  'current_sql': "{}".format(hyp.current_sql),
                  'history': "{}".format(hyp.history[0]), 'dec_seq': "{}".format(hyp.dec_seq),
                  'tag_seq': "{}".format(hyp.tag_seq), 'sql': hyp.sql,
                  'logprob': "{}".format(hyp.logprob),
                  'exit': bool_exit, 'q_counter': user.q_counter,
                  'questioned_indices': user.questioned_pointers,
                  'questioned_tags': "{}".format(user.questioned_tags),
                  'per_time_spent': per_time_spent, 'bool_unclear': bool_unclear}
        if isinstance(agent.error_detector, ErrorDetectorBayesDropout):
            record.update({'logprob_list': "{}".format(hyp.logprob_list),
                           'test_tag_seq': "{}".format(hyp.test_tag_seq)})
        # results.append(record)
        results[st] = record

        st += 1

        print("Saving records...")
        json.dump({'results': results,
                   'st': st, 'time_spent': time_spent,
                   'count_exit': count_exit},
                  open(save_path, "w"), indent=4)

        end_signal = raw_input(bcolors.GREEN + bcolors.BOLD +
                               "Next? Press 'Enter' to continue, Ctrl+C to quit." + bcolors.ENDC)
        if end_signal != "":
            return

    print(bcolors.RED + bcolors.BOLD + "Congratulations! You have completed all your task!" + bcolors.ENDC)
    print("Your average time spent: {:.3f}".format((time_spent / len(data))))
    print("You exited %d times." % count_exit)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_emb', action='store_true',
                        help='Train word embedding.')
    parser.add_argument('--toy', action='store_true',
                        help='If set, use small data; used for fast debugging.')
    parser.add_argument('--models', type=str, help='path to saved model')
    parser.add_argument('--test_data_path',type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--history_type', type=str, default='full', choices=['full','part','no'],
                        help='full, part, or no history')
    parser.add_argument('--table_type', type=str, default='std', choices=['std','hier','no'],
                        help='standard, hierarchical, or no table info')
    parser.add_argument('--dr', type=float, default=0.3, help='Dropout rate for training.')
    # for interaction
    parser.add_argument('--num_options', type=str, default='3', help='#of options.')
    parser.add_argument('--real_user', action='store_true', help='Real user interaction.')
    parser.add_argument('--err_detector', type=str, default='any', help='the error detector: prob-x.')
    parser.add_argument('--structure', type=int, default=1, help='Whether to change to kw structure.')
    parser.add_argument('--seek', type=int, default=0, help='Whether to seek the next available one when user '
                                                            'negates all options.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate for uncertainty analysis.')
    parser.add_argument('--passes', type=int, default=1, help='Number of decoding passes.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for dropout evaluation.')
    parser.add_argument('--data', type=str, default='dev', choices=['dev', 'user_study'], help='Testing data.')

    # temperature
    parser.add_argument('--temperature', type=int, default=0, help='Set to 1 for using softmax temperature.')

    args = parser.parse_args()
    use_hs = True
    if args.history_type == "no":
        args.history_type = "full"
        use_hs = False

    # Seeds for random number generation
    print("## seed: %d" % args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        GPU = True
        device = None
    else:
        GPU = False
        device = lambda storage, loc: storage #torch.device('cpu')

    print("## temperature: %d" % args.temperature)

    N_word=300
    B_word=42
    N_h = 300
    N_depth=2
    # if args.part:
    #     part = True
    # else:
    #     part = False
    if args.toy:
        USE_SMALL=True
        BATCH_SIZE=2 #20
    else:
        USE_SMALL=False
        BATCH_SIZE=2 #64
    # TRAIN_ENTRY=(False, True, False)  # (AGG, SEL, COND)
    # TRAIN_AGG, TRAIN_SEL, TRAIN_COND = TRAIN_ENTRY
    learning_rate = 1e-4

    data = json.load(open(args.test_data_path))
    # dev_data = load_train_dev_dataset(args.train_component, "dev", args.history)
    if args.data == "user_study":
        sampled_ids = json.load(open("syntaxSQL/data/user_study_ids.json", "r"))
        data = [data[idx] for idx in sampled_ids]

    word_emb = load_word_emb('syntaxSQL/glove/glove.%dB.%dd.txt'%(B_word,N_word),
                             load_used=args.train_emb, use_small=USE_SMALL)
    # dev_data = load_train_dev_dataset(args.train_component, "dev", args.history)
    #word_emb = load_concat_wemb('glove/glove.42B.300d.txt', "/data/projects/paraphrase/generation/para-nmt-50m/data/paragram_sl999_czeng.txt")

    print "Creating parser..."
    model = SuperModel(word_emb, N_word=N_word, gpu=GPU, trainable_emb = args.train_emb, table_type=args.table_type,
                       use_hs=use_hs, dr=args.dr, temperature=args.temperature)

    # agg_m, sel_m, cond_m = best_model_name(args)
    # torch.save(model.state_dict(), "saved_models/{}_models.dump".format(args.train_component))

    if args.temperature:
        prefix = "calibr_"
    else:
        prefix = ""
    print "Loading from modules..."
    model.multi_sql.load_state_dict(torch.load("{}/{}multi_sql_models.dump".format(args.models, prefix), map_location=device))
    model.key_word.load_state_dict(torch.load("{}/{}keyword_models.dump".format(args.models, prefix), map_location=device))
    model.col.load_state_dict(torch.load("{}/{}col_models.dump".format(args.models, prefix), map_location=device))
    model.op.load_state_dict(torch.load("{}/{}op_models.dump".format(args.models, prefix), map_location=device))
    model.agg.load_state_dict(torch.load("{}/{}agg_models.dump".format(args.models, prefix), map_location=device))
    model.root_teminal.load_state_dict(torch.load("{}/{}root_tem_models.dump".format(args.models, prefix), map_location=device))
    model.des_asc.load_state_dict(torch.load("{}/{}des_asc_models.dump".format(args.models, prefix), map_location=device))
    model.having.load_state_dict(torch.load("{}/{}having_models.dump".format(args.models, prefix), map_location=device))
    model.andor.load_state_dict(torch.load("{}/{}andor_models.dump".format(args.models, prefix), map_location=device))

    print "Creating MISP agent..."
    # question generator
    question_generator = QuestionGenerator()

    # environment setup: user simulator
    error_evaluator = ErrorEvaluator()
    if args.real_user:
        user = RealUser(error_evaluator, get_table_dict("syntaxSQL/data/tables.json"))
    else:
        user = UserSim(error_evaluator)

    if args.err_detector == 'any':
        error_detector = ErrorDetectorProbability(1.1) #ask any SU
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
    else:
        raise Exception("Invalid error detector setup %s!" % args.err_detector)

    if args.num_options == 'inf':
        print("WARNING: Unlimited options!")
        num_options = np.inf
    else:
        num_options = int(args.num_options)
        print("num_options: {}".format(num_options))

    print("bool_structure_rev: {}".format(args.structure))
    print("bool_seek: {}".format(args.seek))
    world_model = WorldModel(model, num_options, args.structure, args.seek, args.passes, args.dropout)
    agent = Agent(world_model, error_detector, question_generator)

    if not os.path.exists(os.path.dirname(args.output_path)):
        os.mkdir(os.path.dirname(args.output_path))
    if args.real_user:
        real_user_interaction(user, agent, data, args.output_path)
    else:
        interaction(user, agent, data, args.output_path, bool_interaction=True)



