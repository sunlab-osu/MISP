""" The main function for interactive semantic parsing based on EditSQL. Dataset: Spider. """

import os
import sys
import numpy as np
import random
import argparse
import torch
import datetime, pytimeparse
import json
import pickle
import traceback
import subprocess
import copy
import re
from collections import defaultdict, Counter

from EditSQL.postprocess_eval import read_schema, read_prediction, postprocess, write_and_evaluate, postprocess_one
from EditSQL.eval_scripts.evaluation import evaluate_single, build_foreign_key_map_from_json, evaluate
from EditSQL.eval_scripts.evaluation import WHERE_OPS, AGG_OPS
from EditSQL.data_util import dataset_split as ds
from EditSQL.data_util.interaction import load_function
from EditSQL.data_util.utterance import Utterance

from EditSQL.logger import Logger
from EditSQL.data_util import atis_data
from EditSQL.model.schema_interaction_model import SchemaInteractionATISModel
from EditSQL.model_util import Metrics, get_progressbar, write_prediction, update_sums, construct_averages,\
    evaluate_interaction_sample, evaluate_utterance_sample, train_epoch_with_interactions, train_epoch_with_utterances
from EditSQL.world_model import WorldModel
from EditSQL.error_detector import ErrorDetectorProbability, ErrorDetectorBayesDropout, ErrorDetectorSim
from EditSQL.environment import ErrorEvaluator, UserSim, RealUser, GoldUserSim
from EditSQL.agent import Agent
from EditSQL.question_gen import QuestionGenerator
from MISP_SQL.utils import SELECT_AGG_v2, WHERE_COL, WHERE_OP, WHERE_ROOT_TERM, GROUP_COL, HAV_AGG_v2, \
    HAV_OP_v2, HAV_ROOT_TERM_v2, ORDER_AGG_v2, ORDER_DESC_ASC, ORDER_LIMIT, IUEN_v2, OUTSIDE
from user_study_utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VALID_EVAL_METRICS = [Metrics.LOSS, Metrics.TOKEN_ACCURACY, Metrics.STRING_ACCURACY]
TRAIN_EVAL_METRICS = [Metrics.LOSS, Metrics.TOKEN_ACCURACY, Metrics.STRING_ACCURACY]
FINAL_EVAL_METRICS = [Metrics.STRING_ACCURACY, Metrics.TOKEN_ACCURACY]


def interpret_args():
    """ Interprets the command line arguments, and returns a dictionary. """
    parser = argparse.ArgumentParser()

    ### Data parameters
    parser.add_argument(
        '--raw_train_filename',
        type=str,
        default='../atis_data/data/resplit/processed/train_with_tables.pkl')
    parser.add_argument(
        '--raw_dev_filename',
        type=str,
        default='../atis_data/data/resplit/processed/dev_with_tables.pkl')
    parser.add_argument(
        '--raw_validation_filename',
        type=str,
        default='../atis_data/data/resplit/processed/valid_with_tables.pkl')
    parser.add_argument(
        '--raw_test_filename',
        type=str,
        default='../atis_data/data/resplit/processed/test_with_tables.pkl')

    parser.add_argument('--data_directory', type=str, default='processed_data')

    parser.add_argument('--processed_train_filename', type=str, default='train.pkl')
    parser.add_argument('--processed_dev_filename', type=str, default='dev.pkl')
    parser.add_argument('--processed_validation_filename', type=str, default='validation.pkl')
    parser.add_argument('--processed_test_filename', type=str, default='test.pkl')

    parser.add_argument('--database_schema_filename', type=str, default=None)
    parser.add_argument('--embedding_filename', type=str, default=None)

    parser.add_argument('--input_vocabulary_filename', type=str, default='input_vocabulary.pkl')
    parser.add_argument('--output_vocabulary_filename',
                        type=str,
                        default='output_vocabulary.pkl')

    parser.add_argument('--input_key', type=str, default='nl_with_dates')

    parser.add_argument('--anonymize', type=bool, default=False)
    parser.add_argument('--anonymization_scoring', type=bool, default=False)
    parser.add_argument('--use_snippets', type=bool, default=False)

    parser.add_argument('--use_previous_query', type=bool, default=False)
    parser.add_argument('--maximum_queries', type=int, default=1)
    parser.add_argument('--use_copy_switch', type=bool, default=False)
    parser.add_argument('--use_query_attention', type=bool, default=False)

    parser.add_argument('--use_utterance_attention', type=bool, default=False)

    parser.add_argument('--freeze', type=bool, default=False)
    parser.add_argument('--scheduler', type=bool, default=False)

    parser.add_argument('--use_bert', type=bool, default=False)
    parser.add_argument("--bert_type_abb", type=str, help="Type of BERT model to load. e.g.) uS, uL, cS, cL, and mcS")
    parser.add_argument("--bert_input_version", type=str, default='v1')
    parser.add_argument('--fine_tune_bert', type=bool, default=False)
    parser.add_argument('--lr_bert', default=1e-5, type=float, help='BERT model learning rate.')

    ### Debugging/logging parameters
    parser.add_argument('--logdir', type=str, default='logs')
    parser.add_argument('--deterministic', type=bool, default=False)
    parser.add_argument('--num_train', type=int, default=-1)

    parser.add_argument('--logfile', type=str, default='log.txt')
    parser.add_argument('--results_file', type=str, default='results.txt')

    ### Model architecture
    parser.add_argument('--input_embedding_size', type=int, default=300)
    parser.add_argument('--output_embedding_size', type=int, default=300)

    parser.add_argument('--encoder_state_size', type=int, default=300)
    parser.add_argument('--decoder_state_size', type=int, default=300)

    parser.add_argument('--encoder_num_layers', type=int, default=1)
    parser.add_argument('--decoder_num_layers', type=int, default=2)
    parser.add_argument('--snippet_num_layers', type=int, default=1)

    parser.add_argument('--maximum_utterances', type=int, default=5)
    parser.add_argument('--state_positional_embeddings', type=bool, default=False)
    parser.add_argument('--positional_embedding_size', type=int, default=50)

    parser.add_argument('--snippet_age_embedding', type=bool, default=False)
    parser.add_argument('--snippet_age_embedding_size', type=int, default=64)
    parser.add_argument('--max_snippet_age_embedding', type=int, default=4)
    parser.add_argument('--previous_decoder_snippet_encoding', type=bool, default=False)

    parser.add_argument('--discourse_level_lstm', type=bool, default=False)

    parser.add_argument('--use_schema_attention', type=bool, default=False)
    parser.add_argument('--use_encoder_attention', type=bool, default=False)

    parser.add_argument('--use_schema_encoder', type=bool, default=False)
    parser.add_argument('--use_schema_self_attention', type=bool, default=False)
    parser.add_argument('--use_schema_encoder_2', type=bool, default=False)

    ### Training parameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--train_maximum_sql_length', type=int, default=200)
    parser.add_argument('--train_evaluation_size', type=int, default=100)

    parser.add_argument('--dropout_amount', type=float, default=0.5)

    parser.add_argument('--initial_patience', type=float, default=10.)
    parser.add_argument('--patience_ratio', type=float, default=1.01)

    parser.add_argument('--initial_learning_rate', type=float, default=0.001)
    parser.add_argument('--learning_rate_ratio', type=float, default=0.8)

    parser.add_argument('--interaction_level', type=bool, default=True)
    parser.add_argument('--reweight_batch', type=bool, default=False)

    ### Setting
    # parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--train', type=int, choices=[0,1], default=0)
    parser.add_argument('--debug', type=bool, default=False)

    parser.add_argument('--evaluate', type=bool, default=False)
    parser.add_argument('--attention', type=bool, default=False)
    parser.add_argument('--enable_testing', type=bool, default=False)
    parser.add_argument('--use_predicted_queries', type=bool, default=False)
    parser.add_argument('--evaluate_split', type=str, default='dev')
    parser.add_argument('--evaluate_with_gold_forcing', type=bool, default=False)
    parser.add_argument('--eval_maximum_sql_length', type=int, default=1000)
    parser.add_argument('--results_note', type=str, default='')
    parser.add_argument('--compute_metrics', type=bool, default=False)

    parser.add_argument('--reference_results', type=str, default='')

    parser.add_argument('--interactive', type=bool, default=False)

    parser.add_argument('--database_username', type=str, default="aviarmy")
    parser.add_argument('--database_password', type=str, default="aviarmy")
    parser.add_argument('--database_timeout', type=int, default=2)

    # interaction params - Ziyu
    parser.add_argument('--job', default='test_w_interaction', choices=['test_w_interaction', 'online_learning'],
                        help='Set the job. For parser pretraining, see other scripts.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--raw_data_directory', type=str, help='The data directory of the raw spider data.')

    parser.add_argument('--num_options', type=str, default='3', help='[INTERACTION] Number of options.')
    parser.add_argument('--user', type=str, default='sim', choices=['sim', 'gold_sim', 'real'],
                        help='[INTERACTION] User type.')
    parser.add_argument('--err_detector', type=str, default='any',
                        help='[INTERACTION] The error detector: '
                             '(1) prob=x for using policy probability threshold;'
                             '(2) stddev=x for using Bayesian dropout threshold (need to set --dropout and --passes);'
                             '(3) any for querying about every policy action;'
                             '(4) perfect for using a simulated perfect detector.')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='[INTERACTION] Dropout rate for Bayesian dropout-based uncertainty analysis. '
                             'This does NOT change the dropout rate in training.')
    parser.add_argument('--passes', type=int, default=1,
                        help='[INTERACTION] Number of decoding passes for Bayesian dropout-based uncertainty analysis.')
    parser.add_argument('--friendly_agent', type=int, default=0, choices=[0, 1],
                        help='[INTERACTION] If 1, the agent will not trigger further interactions '
                             'if any wrong decision is not resolved during parsing.')
    parser.add_argument('--ask_structure', type=int, default=0, choices=[0, 1],
                        help='[INTERACTION] Set to True to allow questions about query structure '
                             '(WHERE/GROUP_COL, ORDER/HAV_AGG_v2) in NL.')
    parser.add_argument('--output_path', type=str, default='temp', help='[INTERACTION] Where to save outputs.')

    # online learning
    parser.add_argument('--setting', type=str, default='', choices=['online_pretrain_10p', 'full_train'],
                        help='Model setting; checkpoints will be loaded accordingly.')
    parser.add_argument('--supervision', type=str, default='full_expert',
                        choices=['full_expert', 'misp_neil', 'misp_neil_perfect', 'misp_neil_pos',
                                 'bin_feedback', 'bin_feedback_expert',
                                 'self_train', 'self_train_0.5'],
                        help='[LEARNING] Online learning supervision based on different algorithms.')
    parser.add_argument('--data_seed', type=int, choices=[0, 10, 100],
                        help='[LEARNING] Seed for online learning data.')
    parser.add_argument('--start_iter', type=int, default=0, help='[LEARNING] Starting iteration in online learing.')
    parser.add_argument('--end_iter', type=int, default=-1, help='[LEARNING] Ending iteration in online learing.')
    parser.add_argument('--update_iter', type=int, default=1000,
                        help='[LEARNING] Number of iterations per parser update.')

    args = parser.parse_args()

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    if not (args.train or args.evaluate or args.interactive or args.attention):
        raise ValueError('You need to be training or evaluating')
    if args.enable_testing and not args.evaluate:
        raise ValueError('You should evaluate the model if enabling testing')

    # Seeds for random number generation
    print("## seed: %d" % args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    return args


def evaluation(model, data, eval_fn, valid_pred_path):
    valid_examples = data.get_all_interactions(data.valid_data)
    valid_eval_results = eval_fn(valid_examples,
                                model,
                                name=valid_pred_path,
                                metrics=FINAL_EVAL_METRICS,
                                total_num=atis_data.num_utterances(data.valid_data),
                                database_username=params.database_username,
                                database_password=params.database_password,
                                database_timeout=params.database_timeout,
                                use_predicted_queries=True,
                                max_generation_length=params.eval_maximum_sql_length,
                                write_results=True,
                                use_gpu=True,
                                compute_metrics=params.compute_metrics,
                                bool_progressbar=False)[0]
    token_accuracy = valid_eval_results[Metrics.TOKEN_ACCURACY]
    string_accuracy = valid_eval_results[Metrics.STRING_ACCURACY]

    print("## postprocess_eval...")

    database_schema = read_schema(table_schema_path)
    predictions = read_prediction(valid_pred_path + "_predictions.json")
    postprocess_db_sqls = postprocess(predictions, database_schema, True)

    postprocess_sqls = []
    for db in db_list:
        for postprocess_sql, interaction_id, turn_id in postprocess_db_sqls[db]:
            postprocess_sqls.append([postprocess_sql])

    eval_acc = evaluate(gold_path, postprocess_sqls, db_path, "match",
                        kmaps, bool_verbal=False, bool_predict_file=False)['all']['exact']
    eval_acc = float(eval_acc) * 100  # percentage

    return eval_acc, token_accuracy, string_accuracy


def train(model, data, params, model_save_dir, patience=5):
    """ Trains a model.

    Inputs:
        model (ATISModel): The model to train.
        data (ATISData): The data that is used to train.
        params (namespace): Training parameters.
    """

    save_path = os.path.join(model_save_dir, "model_best.pt")
    train_pred_path = os.path.join(model_save_dir, "train-eval")
    valid_pred_path = os.path.join(model_save_dir, "valid-eval")
    valid_pred_write_path = os.path.join(model_save_dir, "valid_use_predicted_queries")

    # Get the training batches.
    num_train_original = atis_data.num_utterances(data.train_data)
    print("Original number of training utterances:\t" + str(num_train_original))

    eval_fn = evaluate_utterance_sample
    trainbatch_fn = data.get_utterance_batches
    trainsample_fn = data.get_random_utterances
    validsample_fn = data.get_all_utterances
    batch_size = params.batch_size
    if params.interaction_level:
        batch_size = 1
        eval_fn = evaluate_interaction_sample
        trainbatch_fn = data.get_interaction_batches
        trainsample_fn = data.get_random_interactions
        validsample_fn = data.get_all_interactions

    maximum_output_length = params.train_maximum_sql_length
    train_batches = trainbatch_fn(batch_size,
                                  max_output_length=maximum_output_length,
                                  randomize=not params.deterministic)

    if params.num_train >= 0:
        train_batches = train_batches[:params.num_train]

    training_sample = trainsample_fn(params.train_evaluation_size,
                                     max_output_length=maximum_output_length)
    valid_examples = validsample_fn(data.valid_data,
                                    max_output_length=maximum_output_length)

    num_train_examples = sum([len(batch) for batch in train_batches])
    num_steps_per_epoch = len(train_batches)

    print("Actual number of used training examples:\t" +
          str(num_train_examples))
    print("(Shortened by output limit of " +
          str(maximum_output_length) + ")")
    print("Number of steps per epoch:\t" + str(num_steps_per_epoch))
    print("Batch size:\t" + str(batch_size))

    print(
        "Kept " +
        str(num_train_examples) +
        "/" +
        str(num_train_original) +
        " examples")
    print(
        "Batch size of " +
        str(batch_size) +
        " gives " +
        str(num_steps_per_epoch) +
        " steps per epoch")

    # Keeping track of things during training.
    epochs = 0
    # patience = params.initial_patience
    learning_rate_coefficient = 1.
    previous_epoch_loss = float('inf')
    maximum_eval_acc = 0.0
    maximum_token_accuracy = 0.0
    maximum_string_accuracy = 0.0

    countdown = int(patience)

    if params.scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model.trainer, mode='min', )

    print("Time stamp: {}".format(datetime.datetime.now()))
    keep_training = True
    while keep_training:
        print("Epoch:\t" + str(epochs))
        model.set_dropout(params.dropout_amount)

        if not params.scheduler:
            model.set_learning_rate(learning_rate_coefficient * params.initial_learning_rate)

        # Run a training step.
        if params.interaction_level:
            epoch_loss = train_epoch_with_interactions(
                train_batches,
                params,
                model,
                randomize=not params.deterministic,
                bool_progressbar=False)
        else:
            epoch_loss = train_epoch_with_utterances(
                train_batches,
                model,
                randomize=not params.deterministic)

        print("train epoch loss:\t" + str(epoch_loss))

        model.set_dropout(0.)

        # Run an evaluation step on a sample of the training data.
        train_eval_results = eval_fn(training_sample,
                                     model,
                                     params.train_maximum_sql_length,
                                     name=train_pred_path,
                                     write_results=True,
                                     gold_forcing=True,
                                     metrics=TRAIN_EVAL_METRICS,
                                     bool_progressbar=False)[0]

        for name, value in train_eval_results.items():
            print(
                "train final gold-passing " +
                name.name +
                ":\t" +
                "%.2f" %
                value)

        # Run an evaluation step on the validation set. - WITH GOLD FEED
        valid_eval_results = eval_fn(valid_examples,
                                     model,
                                     params.eval_maximum_sql_length,
                                     name=valid_pred_path,
                                     write_results=True,
                                     gold_forcing=True,
                                     metrics=VALID_EVAL_METRICS,
                                     bool_progressbar=False)[0]
        for name, value in valid_eval_results.items():
            print("valid gold-passing " + name.name + ":\t" + "%.2f" % value)

        valid_loss = valid_eval_results[Metrics.LOSS]
        # token_accuracy = valid_eval_results[Metrics.TOKEN_ACCURACY]
        # string_accuracy = valid_eval_results[Metrics.STRING_ACCURACY]

        if params.scheduler:
            scheduler.step(valid_loss)

        if valid_loss > previous_epoch_loss:
            learning_rate_coefficient *= params.learning_rate_ratio
            print(
                "learning rate coefficient:\t" +
                str(learning_rate_coefficient))

        previous_epoch_loss = valid_loss
        saved = False

        # measuring the actual accuracy
        eval_acc, actual_token_accuracy, actual_string_accuracy = evaluation(
            model, data, eval_fn, valid_pred_write_path)
        for name, value in zip(["eval accuracy", "actual token accuracy", "actual string accuracy"],
                               [eval_acc, actual_token_accuracy, actual_string_accuracy]):
            print("valid post-eval " + name + ":\t" + "%.2f" % value)

        if not saved and eval_acc > maximum_eval_acc:
            maximum_eval_acc = eval_acc
            countdown = int(patience)
            model.save(save_path)

            maximum_token_accuracy = actual_token_accuracy
            maximum_string_accuracy = actual_string_accuracy

            print("maximum eval accuracy:\t" + str(maximum_eval_acc))
            print("actual token accuracy:\t" + str(actual_token_accuracy))
            print("actual string accuracy:\t" + str(actual_string_accuracy))
            print("patience:\t" + str(patience))
            print("save file:\t" + str(save_path))

        if countdown <= 0:
            keep_training = False

        countdown -= 1
        print("countdown:\t" + str(countdown))
        print("")

        epochs += 1
        print("Time stamp: {}".format(datetime.datetime.now()))
        sys.stdout.flush()

    print("Finished training!")

    # loading the best checkpoint
    print("## Loading the best checkpoint...")
    model.load(save_path)

    return maximum_eval_acc, maximum_token_accuracy, maximum_string_accuracy


def extract_weighted_example(old_example, generated_sql, utter_revision_fn,
                             feedback_records=None, gen_tag_seq=None, dec_seq=None, conf_threshold=None,
                             complete_vocab=None, weight_mode="pos,neg,conf", g_tag_seq=None, bool_verbal=True):
    # should return a list like: [(['select', 'count', '(', 'head.*', ')', 'where', 'head.age', '>', 'value'], [])]

    assert weight_mode in ("pos,neg,conf", "pos,conf", "pos", "pos2", "gold-assessment")

    if generated_sql[-1] == "_EOS":
        generated_sql = generated_sql[:-1]
        if dec_seq is not None:
            dec_seq = dec_seq[:-1]

    true_sql = old_example.interaction.utterances[0].original_gold_query
    new_example = copy.deepcopy(old_example)

    new_output_sequences = [(generated_sql, [])]
    new_utterance = utter_revision_fn(new_example.interaction.utterances[0], new_output_sequences)

    # calculate weights
    if weight_mode in ("pos,conf", "pos", "pos2", "gold-assessment"):
        gold_query_to_use = new_utterance.gold_query_to_use

        kw_indices = [complete_vocab.index(kw) for kw in ['select', 'where', 'group_by', 'having', 'order_by',
                                                          'intersect', 'union', 'except']]

        if weight_mode == "pos,conf":
            # collect positive or confident semantic units
            su_list = [su for su, label in feedback_records if label == 'yes'] + \
                      [su for su in gen_tag_seq if su[-2] >= conf_threshold]
        else:
            # weight_mode in ("pos", "pos2", "gold-assessment")
            su_list = [su for su, label in feedback_records if label == 'yes']

        dec_seq_weights = [0.0] * len(dec_seq)
        for su in su_list:
            dec_idx = su[-1]
            # dec_seq_weights[dec_idx] = 1.0

            # revised 0206: there could be cases where dec_idx refers to an inaccurate position in multi-choice setting.
            semantic_tag = su[0]
            if semantic_tag in {SELECT_AGG_v2, ORDER_AGG_v2, HAV_AGG_v2}:
                new_decision_item = []
                col, agg, bool_distinct = su[1:4]
                if agg[-1] > 0:
                    agg_name = AGG_OPS[agg[-1]]
                    new_decision_item.append(complete_vocab.index(agg_name))
                    new_decision_item.append(complete_vocab.index('('))
                    if bool_distinct:
                        new_decision_item.append(complete_vocab.index('distinct'))
                    new_decision_item.append(col[-1])
                    new_decision_item.append(complete_vocab.index(')'))
                else:
                    if bool_distinct:
                        new_decision_item.append(complete_vocab.index('distinct'))
                    new_decision_item.append(col[-1])

                if dec_seq[dec_idx] == new_decision_item:
                    dec_seq_weights[dec_idx] = 1.0
                else:
                    bool_found = False
                    st_idx = dec_idx + 1
                    while st_idx < len(dec_seq) and (isinstance(dec_seq[st_idx], list) or dec_seq[st_idx] not in kw_indices):
                        if isinstance(dec_seq[st_idx], list) and dec_seq[st_idx] == new_decision_item:
                            dec_seq_weights[st_idx] = 1.0
                            bool_found = True
                            break
                        st_idx += 1

                    if not bool_found:
                        st_idx = dec_idx - 1
                        while st_idx >= 0 and (isinstance(dec_seq[st_idx], list) or dec_seq[st_idx] not in kw_indices):
                            if isinstance(dec_seq[st_idx], list) and dec_seq[st_idx] == new_decision_item:
                                bool_found = True
                                break
                            st_idx -= 1

                        if not bool_found: # likely appears before dec_idx
                            print("Exception in extract_weighted_example: su = {}, new_dec_item = {}\ndec_seq = {}\n".format(
                                su, new_decision_item, dec_seq))

            elif semantic_tag in {WHERE_COL, GROUP_COL}:
                new_decision_item = su[1][-1]

                if not isinstance(dec_seq[dec_idx], list) and dec_seq[dec_idx] == new_decision_item:
                    dec_seq_weights[dec_idx] = 1.0
                else:
                    bool_found = False
                    st_idx = dec_idx + 1
                    while st_idx < len(dec_seq) and (isinstance(dec_seq[st_idx], list) or dec_seq[st_idx] not in kw_indices):
                        if not isinstance(dec_seq[st_idx], list) and dec_seq[st_idx] == new_decision_item:
                            dec_seq_weights[st_idx] = 1.0
                            bool_found = True
                            break
                        st_idx += 1

                    if not bool_found:
                        st_idx = dec_idx - 1
                        while st_idx >= 0 and (isinstance(dec_seq[st_idx], list) or dec_seq[st_idx] not in kw_indices):
                            if not isinstance(dec_seq[st_idx], list) and dec_seq[st_idx] == new_decision_item:
                                bool_found = True
                                break
                            st_idx -= 1

                        if not bool_found:  # likely appears before dec_idx
                            print("Exception in extract_weighted_example: su = {}, new_dec_item = {}\ndec_seq = {}\n".format(
                                  su, new_decision_item, dec_seq))

            else:
                dec_seq_weights[dec_idx] = 1.0

        gold_query_weights = []
        gold_query_idx = 0

        for dec_idx, dec_item in enumerate(dec_seq):
            if dec_seq_weights[dec_idx] == 1.0:
                if isinstance(dec_item, list):
                    gold_query_weights.extend([1.0] * len(dec_item))
                    gold_query_idx += len(dec_item)
                else:
                    gold_query_weights.append(1.0)
                    gold_query_idx += 1
            elif gold_query_to_use[gold_query_idx] in {'select', 'order_by', 'having', 'where', 'group_by',
                                                       'distinct', ',', 'and', 'or'}:
                if weight_mode == "pos": # TODO: is this right?
                    gold_query_weights.append(0.0)
                elif weight_mode == "pos2":
                    if gold_query_to_use[gold_query_idx] == 'distinct':  # MISP does not verify it
                        gold_query_weights.append(0.0)
                    else:
                        gold_query_weights.append(None)  # need further process
                elif weight_mode == "gold-assessment":
                    if gold_query_to_use[gold_query_idx] == 'distinct' and gold_query_idx == 1:
                        gold_query_weights.append(float(true_sql[1] == 'distinct'))
                    else:
                        gold_query_weights.append(float(gold_query_to_use[gold_query_idx] in true_sql))
                else:
                    gold_query_weights.append(1.0)
                gold_query_idx += 1
            elif gold_query_to_use[gold_query_idx] == 'value' and \
                    gold_query_to_use[gold_query_idx-3:gold_query_idx+1] == ['between', 'value', 'and', 'value'] and\
                    gold_query_weights[gold_query_idx-2] == 1:
                gold_query_weights.append(1.0)
                gold_query_idx += 1
            else:
                if isinstance(dec_item, list):
                    gold_query_weights.extend([0.0] * len(dec_item))
                    gold_query_idx += len(dec_item)
                else:
                    gold_query_weights.append(0.0)
                    gold_query_idx += 1

        if weight_mode == "pos2":
            count_valid, count_valid_clause = 0, 0
            for gold_query_idx in range(len(gold_query_weights))[::-1]:
                if gold_query_weights[gold_query_idx] is None:
                    if gold_query_to_use[gold_query_idx] in {'select', 'order_by', 'having', 'where', 'group_by'}:
                        gold_query_weights[gold_query_idx] = float(count_valid_clause > 0)
                        count_valid_clause = 0
                        count_valid = 0
                    else:
                        assert gold_query_to_use[gold_query_idx] in {',', 'and', 'or'}
                        gold_query_weights[gold_query_idx] = float(count_valid > 0)
                        count_valid = 0
                else:
                    count_valid += gold_query_weights[gold_query_idx]
                    count_valid_clause += gold_query_weights[gold_query_idx]

        if weight_mode in ("pos", "pos2"):
            if sum(gold_query_weights) == 0.0:
                print("Example skipped with invalid weights!")
                return None
            else:
                gold_query_weights += [0.0]
                new_utterance.set_gold_query_weights(gold_query_weights)
        elif weight_mode == "gold-assessment":
            gold_query_weights += [float(len(g_tag_seq) == len(gen_tag_seq))] # no missing/redundant
            new_utterance.set_gold_query_weights(gold_query_weights)
        else:
            gold_query_weights += [1.0]  # with EOS
            new_utterance.set_gold_query_weights(gold_query_weights)

        if bool_verbal:
            print("FINAL: gold_query_weights = {}".format(gold_query_weights))

    new_example.interaction.utterances = [new_utterance]

    return new_example.interaction


def online_learning(online_train_data_examples, init_train_data_examples, online_train_raw_examples, data,
                    user, agent, max_generation_length, model_save_dir, record_save_path,
                    update_iter, model_renew_fn, utter_revision_fn, start_idx=0, end_idx=-1,
                    metrics=None, database_username=None, database_password=None, database_timeout=None,
                    bool_interaction=True, supervision="misp_neil"):
    """ Online learning with MISP. """

    class pseudoDatasetSplit: # a simplified class to be used as DatasetSplit
        def __init__(self, examples):
            self.examples = examples

    database_schema = read_schema(table_schema_path)

    def _evaluation(example, gen_sequence, raw_query):
        # check acc & add to record
        flat_pred = example.flatten_sequence(gen_sequence)
        pred_sql_str = ' '.join(flat_pred)
        assert len(example.identifier.split('/')) == 2
        database_id, interaction_id = example.identifier.split('/')
        postprocessed_sql_str = postprocess_one(pred_sql_str, database_schema[database_id])
        try:
            exact_score, partial_scores, hardness = evaluate_single(
                postprocessed_sql_str, raw_query, db_path, database_id, agent.world_model.kmaps)
        except:
            question = example.interaction.utterances[0].original_input_seq
            print("Exception in evaluate_single:\nidx: {}, db: {}, question: {}\np_str: {}\ng_str: {}\n".format(
                idx, database_id, " ".join(question), postprocessed_sql_str, raw_query))
            exact_score = 0.0
            partial_scores = "Exception"
            hardness = "Unknown"

        return exact_score, partial_scores, hardness

    num_total_examples = len(online_train_data_examples)
    online_train_data = data.get_all_interactions(pseudoDatasetSplit(online_train_data_examples))

    if supervision == "misp_neil":
        weight_mode = "pos,conf"
    else:
        assert supervision == "misp_neil_pos"
        weight_mode = "pos"

    metrics_sums = {}
    for metric in metrics:
        metrics_sums[metric] = 0.

    interaction_records = []
    annotation_buffer = []

    # per 1K iteration
    iter_annotation_buffer = []
    iter_annotated_example_scores = [] # exact scores of annotated examples in the buffer (for analysis)

    if start_idx > 0:
        print("Loading interaction records from %s..." % record_save_path)
        interaction_records = json.load(open(record_save_path, 'r'))
        print("Record item size: %d " % len(interaction_records))

    learning_start_time = datetime.datetime.now()
    print("## Online starting time: {}".format(learning_start_time))
    count_exception, count_exit = 0, 0
    for idx, (raw_example, example) in enumerate(zip(online_train_raw_examples, online_train_data)):
        if idx < len(interaction_records):
            record = interaction_records[idx]

            count_exception += int(record['exception'])
            count_exit += int(record['exit'])

            if record['exception']:
                print("Example skipped!")
                sequence = eval(record['sql'])
            else:
                sequence = eval(record['sql'])
                try:
                    tag_seq = eval(record["tag_seq"])
                except:
                    print("  exception in recovering tag_seq: {}".format(record['tag_seq']))
                    assert "array" in record["tag_seq"]
                    _tag_seq = record["tag_seq"].replace("array", "")
                    _tag_seq = eval(_tag_seq)
                    tag_seq = []
                    for su in _tag_seq:
                        if isinstance(su[-2], list):
                            su = list(su)
                            su[-2] = su[-2][0]
                            su = tuple(su)
                        tag_seq.append(su)

                complete_vocab = []
                try:
                    g_sql = eval(record['true_sql_i'])
                except:
                    # this happens when base_vocab was incorrectly saved in json
                    assert 'base_vocab' in record['true_sql_i']
                    base_vocab_idx = record['true_sql_i'].index("\'base_vocab\'")
                    true_sql_i_str = record['true_sql_i'][:base_vocab_idx-2] + '}'
                    g_sql = eval(true_sql_i_str)

                base_vocab = agent.world_model.vocab
                for id in range(len(base_vocab)):
                    complete_vocab.append(base_vocab.id_to_token(id))
                id2col_name = {v: k for k, v in g_sql["column_names_surface_form_to_id"].items()}
                for id in range(len(g_sql["column_names_surface_form_to_id"])):
                    complete_vocab.append(id2col_name[id])

                annotated_example = extract_weighted_example(example, sequence, utter_revision_fn,
                                                             feedback_records=eval(record['feedback_records']),
                                                             gen_tag_seq=tag_seq,
                                                             dec_seq=eval(record['dec_seq']),
                                                             conf_threshold=agent.error_detector.prob_threshold,
                                                             complete_vocab=complete_vocab,
                                                             weight_mode=weight_mode)
                if annotated_example is not None:
                    iter_annotation_buffer.append(annotated_example)
                    iter_annotated_example_scores.append(record['exact_score'])
        else:
            with torch.no_grad():
                input_item = agent.world_model.semparser.spider_single_turn_encoding(
                    example, max_generation_length)

                question = example.interaction.utterances[0].original_input_seq
                true_sql = example.interaction.utterances[0].original_gold_query
                print("\n" + "#" * 50)
                print("Example {}:".format(idx))
                print("NL input: {}".format(" ".join(question)))
                print("True SQL: {}".format(" ".join(true_sql)))

                g_sql = raw_example['sql']
                g_sql["extracted_clause_asterisk"] = extract_clause_asterisk(true_sql)
                g_sql["column_names_surface_form_to_id"] = input_item[-1].column_names_surface_form_to_id
                g_sql["base_vocab"] = agent.world_model.vocab

                complete_vocab = []
                for id in range(len(g_sql["base_vocab"])):
                    complete_vocab.append(g_sql["base_vocab"].id_to_token(id))
                id2col_name = {v: k for k, v in g_sql["column_names_surface_form_to_id"].items()}
                for id in range(len(g_sql["column_names_surface_form_to_id"])):
                    complete_vocab.append(id2col_name[id])

                try:
                    hyp = agent.world_model.decode(input_item, bool_verbal=False, dec_beam_size=1)[0]
                    print("-" * 50 + "\nBefore interaction: \ninitial SQL: {}".format(" ".join(hyp.sql)))
                except Exception: # tag_seq generation exception - e.g., when its syntax is wrong
                    count_exception += 1
                    # traceback.print_exc()
                    print("Decoding Exception (count = {}) in example {}!".format(count_exception, idx))
                    final_encoder_state, encoder_states, schema_states, max_generation_length, snippets, input_sequence, \
                        previous_queries, previous_query_states, input_schema = input_item
                    prediction = agent.world_model.semparser.decoder(
                        final_encoder_state,
                        encoder_states,
                        schema_states,
                        max_generation_length,
                        snippets=snippets,
                        input_sequence=input_sequence,
                        previous_queries=previous_queries,
                        previous_query_states=previous_query_states,
                        input_schema=input_schema,
                        dropout_amount=0.0)
                    print("-" * 50 + "\nBefore interaction: \ninitial SQL: {}".format(" ".join(prediction.sequence)))
                    sequence = prediction.sequence
                    probability = prediction.probability

                    exact_score, partial_scores, hardness = _evaluation(example, sequence, raw_example['query'])

                    g_sql.pop('base_vocab') # do not save it
                    record = {'nl': " ".join(question), 'true_sql': " ".join(true_sql),
                              'true_sql_i': "{}".format(g_sql),
                              'sql': "{}".format(sequence), 'dec_seq': "None",
                              'tag_seq': "None", 'logprob': "{}".format(np.log(probability)),
                              "questioned_indices": [], 'q_counter': 0,
                              'exit': False, 'exception': True,
                              'exact_score': exact_score, 'partial_scores': "{}".format(partial_scores),
                              'hardness': hardness, 'idx': idx}
                    interaction_records.append(record)
                    print("Example skipped!")
                else:
                    if not bool_interaction:
                        exact_score, partial_scores, hardness = _evaluation(example, hyp.sql, raw_example['query'])

                        g_sql.pop('base_vocab')  # do not save it
                        record = {'nl': " ".join(question), 'true_sql': " ".join(true_sql),
                                  'true_sql_i': "{}".format(g_sql),
                                  'sql': "{}".format(hyp.sql), 'dec_seq': "{}".format(hyp.dec_seq),
                                  'tag_seq': "{}".format(hyp.tag_seq), 'logprob': "{}".format(hyp.logprob),
                                  "questioned_indices": [], 'q_counter': 0, 'exit': False, 'exception': False,
                                  'exact_score': exact_score, 'partial_scores': "{}".format(partial_scores),
                                  'hardness': hardness, 'idx': idx}
                        interaction_records.append(record)
                    else:
                        try:
                            new_hyp, bool_exit = agent.interactive_parsing_session(user, input_item, g_sql, hyp,
                                                                                   bool_verbal=False)
                            if bool_exit:
                                count_exit += 1
                        except Exception:
                            count_exception += 1
                            # traceback.print_exc()
                            print("Interaction Exception (count = {}) in example {}!".format(count_exception, idx))
                            print("-" * 50 + "\nAfter interaction: \nfinal SQL: {}".format(" ".join(hyp.sql)))

                            exact_score, partial_scores, hardness = _evaluation(example, hyp.sql, raw_example['query'])

                            g_sql.pop('base_vocab')  # do not save it
                            record = {'nl': " ".join(question), 'true_sql': " ".join(true_sql),
                                      'true_sql_i': "{}".format(g_sql),
                                      'sql': "{}".format(hyp.sql), 'dec_seq': "{}".format(hyp.dec_seq),
                                      'tag_seq': "{}".format(hyp.tag_seq), 'logprob': "{}".format(hyp.logprob),
                                      'exit': False, 'exception': True,
                                      'q_counter': user.q_counter, # questions are still counted
                                      'questioned_indices': user.questioned_pointers,
                                      'questioned_tags': "{}".format(user.questioned_tags),
                                      'feedback_records': "{}".format(user.feedback_records),
                                      'exact_score': exact_score, 'partial_scores': "{}".format(partial_scores),
                                      'hardness': hardness, 'idx': idx}
                            interaction_records.append(record)
                            print("Example skipped!")
                        else:
                            hyp = new_hyp
                            print("-" * 50 + "\nAfter interaction: \nfinal SQL: {}".format(" ".join(hyp.sql)))

                            exact_score, partial_scores, hardness = _evaluation(example, hyp.sql, raw_example['query'])

                            g_sql.pop('base_vocab')  # do not save it
                            record = {'nl': " ".join(question), 'true_sql': " ".join(true_sql),
                                      'true_sql_i': "{}".format(g_sql),
                                      'sql': "{}".format(hyp.sql), 'dec_seq': "{}".format(hyp.dec_seq),
                                      'tag_seq': "{}".format(hyp.tag_seq), 'logprob': "{}".format(hyp.logprob),
                                      'exit': bool_exit, 'exception': False, 'q_counter': user.q_counter,
                                      'questioned_indices': user.questioned_pointers,
                                      'questioned_tags': "{}".format(user.questioned_tags),
                                      'feedback_records': "{}".format(user.feedback_records),
                                      'exact_score': exact_score, 'partial_scores': "{}".format(partial_scores),
                                      'hardness': hardness, 'idx': idx}
                            interaction_records.append(record)
                            annotated_example = extract_weighted_example(example, hyp.sql, utter_revision_fn,
                                                                         feedback_records=user.feedback_records,
                                                                         gen_tag_seq=hyp.tag_seq,
                                                                         dec_seq=hyp.dec_seq,
                                                                         conf_threshold=agent.error_detector.prob_threshold,
                                                                         complete_vocab=complete_vocab,
                                                                         weight_mode=weight_mode)

                            if annotated_example is not None:
                                iter_annotation_buffer.append(annotated_example)
                                iter_annotated_example_scores.append(record['exact_score'])

                    sequence = hyp.sql
                    probability = np.exp(hyp.logprob)

        original_utt = example.interaction.utterances[0]

        gold_query = original_utt.gold_query_to_use
        original_gold_query = original_utt.original_gold_query

        gold_table = original_utt.gold_sql_results
        gold_queries = [q[0] for q in original_utt.all_gold_queries]
        gold_tables = [q[1] for q in original_utt.all_gold_queries]

        flat_sequence = example.flatten_sequence(sequence)

        update_sums(metrics,
                    metrics_sums,
                    sequence,
                    flat_sequence,
                    gold_query,
                    original_gold_query,
                    database_username=database_username,
                    database_password=database_password,
                    database_timeout=database_timeout,
                    gold_table=gold_table)

        sys.stdout.flush()

        if (idx+1) % update_iter == 0 or (idx+1) == num_total_examples:  # update model
            print("\n~~~\nCurrent interaction performance (iter {}): ".format(idx+1))  # interaction so far
            eval_results = construct_averages(metrics_sums, idx+1)
            for name, value in eval_results.items():
                print(name.name + ":\t" + "%.2f" % value)

            exact_acc = np.average([float(item['exact_score']) for item in interaction_records[:(idx+1)]])
            print("Exact acc: %.3f" % (exact_acc * 100.0))

            # per 1K iteration: the actual acc of collected examples
            iter_annotated_example_acc = np.average(iter_annotated_example_scores)
            print("ANALYSIS: iter_annotated_example_acc: %.3f\n" % (iter_annotated_example_acc * 100.0))

            # stats
            q_count = sum([item['q_counter'] for item in interaction_records[:(idx+1)]])
            print("#questions: {}, #questions per example: {:.3f}.".format(
                q_count, q_count * 1.0 / len(interaction_records[:(idx+1)])))
            print("#exit: {}".format(count_exit))

            if (idx + 1) <= start_idx:
                annotation_buffer.extend(iter_annotation_buffer)
                iter_annotation_buffer = []
                iter_annotated_example_scores = []
                continue

            print("Saving interaction records to %s...\n" % record_save_path)
            json.dump(interaction_records, open(record_save_path, 'w'), indent=4)

            annotation_buffer.extend(iter_annotation_buffer)
            iter_annotation_buffer = []
            iter_annotated_example_scores = []

            # parser update
            print("~~~\nUpdating base semantic parser at iter {}".format(idx+1))
            update_buffer = init_train_data_examples + annotation_buffer
            print("Train data size: %d" % len(update_buffer))

            print("Retraining from scratch...")
            model = agent.world_model.semparser
            # re-initialize
            model = model_renew_fn(model)
            model.build_optim()
            agent.world_model.semparser = model

            # train
            print("## Starting update at iter {}, anno_cost {}...time spent {}".format(
                idx+1, sum([item['q_counter'] for item in interaction_records]),
                datetime.datetime.now() - learning_start_time))
            data.train_data.examples = update_buffer

            model_dir = os.path.join(model_save_dir, '%d/' % (idx + 1))
            if not os.path.isdir(model_dir):
                os.mkdir(model_dir)
            sys.stdout.flush()

            eval_acc, token_acc, string_acc = train(agent.world_model.semparser, data, params, model_dir)

            print("## Ending update at iter {}, anno_cost {}, eval_acc {}, token_acc {}, string_acc {}...time spent {}\n".format(
                idx+1, sum([item['q_counter'] for item in interaction_records]), eval_acc, token_acc, string_acc,
                datetime.datetime.now() - learning_start_time))
            agent.world_model.semparser.eval()
            sys.stdout.flush()

            if end_idx != -1 and idx+1 == end_idx:
                print("## Ending online learning at iter {}\n".format(end_idx))
                break

    print("## End full training at time {}...time spent {}\n".format(
        datetime.datetime.now(), datetime.datetime.now() - learning_start_time))

    # stats
    q_count = sum([item['q_counter'] for item in interaction_records])
    print("#questions: {}, #questions per example: {:.3f}.".format(
        q_count, q_count * 1.0 / len(interaction_records)))
    print("#exit: {}".format(count_exit))
    sys.stdout.flush()

    return interaction_records


def online_learning_full(online_train_data_examples, init_train_data_examples, data, agent,
                         model_save_dir, update_iter, model_renew_fn,
                         online_full_annotation_cost, start_idx=0, end_idx=-1):
    num_total_examples = len(online_train_data_examples)

    learning_start_time = datetime.datetime.now()
    print("## Online starting time: {}".format(learning_start_time))

    for st in np.arange(start_idx, num_total_examples, update_iter):
        annotation_buffer = online_train_data_examples[0: st + update_iter]
        iter_annotation_buffer = online_train_data_examples[st: st + update_iter]

        count_iter = len(annotation_buffer)
        print("~~~\nUpdating base semantic parser at iter {}".format(count_iter))

        # # print information about buffer
        # for item in iter_annotation_buffer:
        #     question = item.utterances[0].original_input_seq
        #     print("NL input: {}".format(question))

        update_buffer = init_train_data_examples + annotation_buffer

        print("Retraining from scratch...")
        model = agent.world_model.semparser
        # re-initialize
        model = model_renew_fn(model)
        model.build_optim()
        agent.world_model.semparser = model

        # train
        print("## Starting update at iter {}, anno_cost {}...time spent {}".format(
            count_iter, sum(online_full_annotation_cost[0:st + update_iter]),
            datetime.datetime.now() - learning_start_time))
        data.train_data.examples = update_buffer

        model_dir = os.path.join(model_save_dir, '%d/' % count_iter)
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        sys.stdout.flush()

        eval_acc, token_acc, string_acc = train(agent.world_model.semparser, data, params, model_dir)

        print("## Ending update at iter {}, anno_cost {}, eval_acc {}, token_acc {}, string_acc {}...time spent {}\n".format(
            count_iter, sum(online_full_annotation_cost[0:st + update_iter]), eval_acc, token_acc, string_acc,
            datetime.datetime.now() - learning_start_time))
        agent.world_model.semparser.eval()

        sys.stdout.flush()

        if end_idx != -1 and count_iter == end_idx:
            print("## Ending online learning at iter {}\n".format(end_idx))
            break

    print("## End full training at time {}...time spent {}\n".format(
        datetime.datetime.now(), datetime.datetime.now() - learning_start_time))


def online_learning_bin_feedback(supervision, online_train_data_examples, init_train_data_examples,
                                 online_train_raw_examples, data, agent, max_generation_length,
                                 model_save_dir, record_save_path,
                                 update_iter, model_renew_fn, utter_revision_fn, online_full_annotation_cost,
                                 start_idx=0, end_idx=-1, metrics=None,
                                 database_username=None, database_password=None, database_timeout=None):
    """ Online learning with binary user feedback (validating results being right or wrong). """

    class pseudoDatasetSplit:  # a simplified class to be used as DatasetSplit
        def __init__(self, examples):
            self.examples = examples

    database_schema = read_schema(table_schema_path)

    num_total_examples = len(online_train_data_examples)
    online_train_data = data.get_all_interactions(pseudoDatasetSplit(online_train_data_examples))

    metrics_sums = {}
    for metric in metrics:
        metrics_sums[metric] = 0.

    interaction_records = []
    annotation_buffer = []
    iter_annotation_buffer = []
    annotation_costs = []

    if start_idx > 0:
        print("Loading interaction records from %s..." % record_save_path)
        interaction_records = json.load(open(record_save_path, 'r'))
        print("Record item size: %d " % len(interaction_records))

    learning_start_time = datetime.datetime.now()
    print("## Online starting time: {}".format(learning_start_time))

    for idx, (raw_example, example) in enumerate(zip(online_train_raw_examples, online_train_data)):
        if idx < len(interaction_records):
            record = interaction_records[idx]
            sequence = eval(record['sql'])
            annotation_costs.append(online_full_annotation_cost[idx])
            if record['exact_score']:
                # iter_annotation_buffer.append(example.interaction)
                annotated_example = extract_weighted_example(example, sequence, utter_revision_fn)
                iter_annotation_buffer.append(annotated_example)
            elif supervision == "bin_feedback_expert":
                iter_annotation_buffer.append(example.interaction)

        else:
            with torch.no_grad():
                input_item = agent.world_model.semparser.spider_single_turn_encoding(
                    example, max_generation_length)

                question = example.interaction.utterances[0].original_input_seq
                true_sql = example.interaction.utterances[0].original_gold_query
                print("\n" + "#" * 50)
                print("Example {}:".format(idx))
                print("NL input: {}".format(" ".join(question)))
                print("True SQL: {}".format(" ".join(true_sql)))

                g_sql = raw_example['sql']

                final_encoder_state, encoder_states, schema_states, max_generation_length, snippets, input_sequence, \
                    previous_queries, previous_query_states, input_schema = input_item
                prediction = agent.world_model.semparser.decoder(
                    final_encoder_state,
                    encoder_states,
                    schema_states,
                    max_generation_length,
                    snippets=snippets,
                    input_sequence=input_sequence,
                    previous_queries=previous_queries,
                    previous_query_states=previous_query_states,
                    input_schema=input_schema,
                    dropout_amount=0.0)
                print("-" * 50 + "\nBefore interaction: \ninitial SQL: {}".format(" ".join(prediction.sequence)))
                sequence = prediction.sequence
                probability = prediction.probability

                # check acc & add to record
                flat_pred = example.flatten_sequence(sequence)
                pred_sql_str = ' '.join(flat_pred)
                assert len(example.identifier.split('/')) == 2
                database_id, interaction_id = example.identifier.split('/')
                postprocessed_sql_str = postprocess_one(pred_sql_str, database_schema[database_id])
                try:
                    exact_score, partial_scores, hardness = evaluate_single(
                        postprocessed_sql_str, raw_example['query'], db_path, database_id, agent.world_model.kmaps)
                except:
                    print("Exception in evaluate_single:\nidx: {}, db: {}, question: {}\np_str: {}\ng_str: {}\n".format(
                        idx, database_id, " ".join(question), postprocessed_sql_str, raw_example['query']))
                    exact_score = 0.0
                    partial_scores = "Exception"
                    hardness = "Unknown"

                if exact_score:
                    # iter_annotation_buffer.append(example.interaction)
                    annotated_example = extract_weighted_example(example, sequence, utter_revision_fn)
                    iter_annotation_buffer.append(annotated_example)
                elif supervision == "bin_feedback_expert":
                    iter_annotation_buffer.append(example.interaction)

                record = {'nl': " ".join(question), 'true_sql': " ".join(true_sql),
                          'true_sql_i': "{}".format(g_sql),
                          'sql': "{}".format(sequence), 'dec_seq': "None",
                          'tag_seq': "None", 'logprob': "{}".format(np.log(probability)),
                          'exact_score': exact_score, 'partial_scores': "{}".format(partial_scores),
                          'hardness': hardness}
                interaction_records.append(record)
                annotation_costs.append(online_full_annotation_cost[idx])

        original_utt = example.interaction.utterances[0]

        gold_query = original_utt.gold_query_to_use
        original_gold_query = original_utt.original_gold_query

        gold_table = original_utt.gold_sql_results
        gold_queries = [q[0] for q in original_utt.all_gold_queries]
        gold_tables = [q[1] for q in original_utt.all_gold_queries]

        flat_sequence = example.flatten_sequence(sequence)

        update_sums(metrics,
                    metrics_sums,
                    sequence,
                    flat_sequence,
                    gold_query,
                    original_gold_query,
                    database_username=database_username,
                    database_password=database_password,
                    database_timeout=database_timeout,
                    gold_table=gold_table)

        sys.stdout.flush()

        if (idx + 1) % update_iter == 0 or (idx + 1) == num_total_examples:  # update model
            print("\n~~~\nCurrent interaction performance (iter {}): ".format(idx + 1))  # interaction so far
            eval_results = construct_averages(metrics_sums, idx + 1)
            for name, value in eval_results.items():
                print(name.name + ":\t" + "%.2f" % value)
            print("")

            if (idx + 1) <= start_idx:
                annotation_buffer.extend(iter_annotation_buffer)
                iter_annotation_buffer = []
                continue

            print("Saving interaction records to %s...\n" % record_save_path)
            json.dump(interaction_records, open(record_save_path, 'w'), indent=4)

            annotation_buffer.extend(iter_annotation_buffer)
            iter_annotation_buffer = []

            # parser update
            print("~~~\nUpdating base semantic parser at iter {}".format(idx + 1))
            update_buffer = init_train_data_examples + annotation_buffer

            print("Retraining from scratch...")
            model = agent.world_model.semparser
            # re-initialize
            model = model_renew_fn(model)
            model.build_optim()
            agent.world_model.semparser = model

            # train
            print("## Starting update at iter {}, anno_cost {}...time spent {}".format(
                idx + 1, sum(annotation_costs),
                datetime.datetime.now() - learning_start_time))
            data.train_data.examples = update_buffer

            model_dir = os.path.join(model_save_dir, '%d/' % (idx + 1))
            if not os.path.isdir(model_dir):
                os.mkdir(model_dir)
            sys.stdout.flush()

            eval_acc, token_acc, string_acc = train(agent.world_model.semparser, data, params, model_dir)

            print("## Ending update at iter {}, anno_cost {}, eval_acc {}, token_acc {}, string_acc {}..."
                  "time spent {}\n".format(
                idx + 1, sum(annotation_costs), eval_acc, token_acc, string_acc,
                datetime.datetime.now() - learning_start_time))
            agent.world_model.semparser.eval()
            sys.stdout.flush()

            if end_idx != -1 and idx + 1 == end_idx:
                print("## Ending online learning at iter {}\n".format(end_idx))
                break

    print("## End full training at time {}...time spent {}\n".format(
        datetime.datetime.now(), datetime.datetime.now() - learning_start_time))

    sys.stdout.flush()

    return interaction_records


def online_learning_self_training(supervision, online_train_data_examples, init_train_data_examples,
                                  online_train_raw_examples, data, agent, max_generation_length,
                                  model_save_dir, record_save_path,
                                  update_iter, model_renew_fn, utter_revision_fn,
                                  start_idx=0, end_idx=-1, metrics=None,
                                  database_username=None, database_password=None, database_timeout=None):
    """ Online learning with binary user feedback (validating results being right or wrong). """

    class pseudoDatasetSplit:  # a simplified class to be used as DatasetSplit
        def __init__(self, examples):
            self.examples = examples

    assert supervision in ('self_train', 'self_train_0.5')
    conf_threshold = None
    if supervision == 'self_train_0.5':
        conf_threshold = 0.5

    database_schema = read_schema(table_schema_path)

    num_total_examples = len(online_train_data_examples)
    online_train_data = data.get_all_interactions(pseudoDatasetSplit(online_train_data_examples))

    metrics_sums = {}
    for metric in metrics:
        metrics_sums[metric] = 0.

    interaction_records = []
    annotation_buffer = []
    iter_annotation_buffer = []

    if start_idx > 0:
        print("Loading interaction records from %s..." % record_save_path)
        interaction_records = json.load(open(record_save_path, 'r'))
        print("Record item size: %d " % len(interaction_records))

    learning_start_time = datetime.datetime.now()
    print("## Online starting time: {}".format(learning_start_time))

    for idx, (raw_example, example) in enumerate(zip(online_train_raw_examples, online_train_data)):
        if idx < len(interaction_records):
            record = interaction_records[idx]
            sequence = eval(record['sql'])

            if supervision == 'self_train' or record['logprob'] > np.log(conf_threshold):
                # iter_annotation_buffer.append(example.interaction)
                annotated_example = extract_weighted_example(example, sequence, utter_revision_fn)
                iter_annotation_buffer.append(annotated_example)

        else:
            with torch.no_grad():
                input_item = agent.world_model.semparser.spider_single_turn_encoding(
                    example, max_generation_length)

                question = example.interaction.utterances[0].original_input_seq
                true_sql = example.interaction.utterances[0].original_gold_query
                print("\n" + "#" * 50)
                print("Example {}:".format(idx))
                print("NL input: {}".format(" ".join(question)))
                print("True SQL: {}".format(" ".join(true_sql)))

                g_sql = raw_example['sql']

                final_encoder_state, encoder_states, schema_states, max_generation_length, snippets, input_sequence, \
                    previous_queries, previous_query_states, input_schema = input_item
                prediction = agent.world_model.semparser.decoder(
                    final_encoder_state,
                    encoder_states,
                    schema_states,
                    max_generation_length,
                    snippets=snippets,
                    input_sequence=input_sequence,
                    previous_queries=previous_queries,
                    previous_query_states=previous_query_states,
                    input_schema=input_schema,
                    dropout_amount=0.0)
                print("-" * 50 + "\nBefore interaction: \ninitial SQL: {}".format(" ".join(prediction.sequence)))
                sequence = prediction.sequence
                probability = prediction.probability

                # check acc & add to record
                flat_pred = example.flatten_sequence(sequence)
                pred_sql_str = ' '.join(flat_pred)
                assert len(example.identifier.split('/')) == 2
                database_id, interaction_id = example.identifier.split('/')
                postprocessed_sql_str = postprocess_one(pred_sql_str, database_schema[database_id])
                try:
                    exact_score, partial_scores, hardness = evaluate_single(
                        postprocessed_sql_str, raw_example['query'], db_path, database_id, agent.world_model.kmaps)
                except:
                    print("Exception in evaluate_single:\nidx: {}, db: {}, question: {}\np_str: {}\ng_str: {}\n".format(
                        idx, database_id, " ".join(question), postprocessed_sql_str, raw_example['query']))
                    exact_score = 0.0
                    partial_scores = "Exception"
                    hardness = "Unknown"

                if supervision == 'self_train' or probability > conf_threshold:
                    # iter_annotation_buffer.append(example.interaction)
                    annotated_example = extract_weighted_example(example, sequence, utter_revision_fn)
                    iter_annotation_buffer.append(annotated_example)

                record = {'nl': " ".join(question), 'true_sql': " ".join(true_sql),
                          'true_sql_i': "{}".format(g_sql),
                          'sql': "{}".format(sequence), 'dec_seq': "None",
                          'tag_seq': "None", 'logprob': "{}".format(np.log(probability)),
                          'exact_score': exact_score, 'partial_scores': "{}".format(partial_scores),
                          'hardness': hardness}
                interaction_records.append(record)

        original_utt = example.interaction.utterances[0]

        gold_query = original_utt.gold_query_to_use
        original_gold_query = original_utt.original_gold_query

        gold_table = original_utt.gold_sql_results
        gold_queries = [q[0] for q in original_utt.all_gold_queries]
        gold_tables = [q[1] for q in original_utt.all_gold_queries]

        flat_sequence = example.flatten_sequence(sequence)

        update_sums(metrics,
                    metrics_sums,
                    sequence,
                    flat_sequence,
                    gold_query,
                    original_gold_query,
                    database_username=database_username,
                    database_password=database_password,
                    database_timeout=database_timeout,
                    gold_table=gold_table)

        sys.stdout.flush()

        if (idx + 1) % update_iter == 0 or (idx + 1) == num_total_examples:  # update model
            print("\n~~~\nCurrent interaction performance (iter {}): ".format(idx + 1))  # interaction so far
            eval_results = construct_averages(metrics_sums, idx + 1)
            for name, value in eval_results.items():
                print(name.name + ":\t" + "%.2f" % value)
            print("")

            if (idx + 1) <= start_idx:
                annotation_buffer.extend(iter_annotation_buffer)
                iter_annotation_buffer = []
                continue

            print("Saving interaction records to %s...\n" % record_save_path)
            json.dump(interaction_records, open(record_save_path, 'w'), indent=4)

            annotation_buffer.extend(iter_annotation_buffer)
            iter_annotation_buffer = []

            # parser update
            print("~~~\nUpdating base semantic parser at iter {}".format(idx + 1))
            update_buffer = init_train_data_examples + annotation_buffer

            print("Retraining from scratch...")
            model = agent.world_model.semparser
            # re-initialize
            model = model_renew_fn(model)
            model.build_optim()
            agent.world_model.semparser = model

            print("Train data size: %d " % len(update_buffer))

            # train
            print("## Starting update at iter {}, anno_cost {}...time spent {}".format(
                idx + 1, 0, datetime.datetime.now() - learning_start_time))
            data.train_data.examples = update_buffer

            model_dir = os.path.join(model_save_dir, '%d/' % (idx + 1))
            if not os.path.isdir(model_dir):
                os.mkdir(model_dir)
            sys.stdout.flush()

            eval_acc, token_acc, string_acc = train(agent.world_model.semparser, data, params, model_dir)

            print("## Ending update at iter {}, anno_cost {}, eval_acc {}, token_acc {}, string_acc {}..."
                  "time spent {}\n".format(
                idx + 1, 0, eval_acc, token_acc, string_acc,
                datetime.datetime.now() - learning_start_time))
            agent.world_model.semparser.eval()
            sys.stdout.flush()

            if end_idx != -1 and idx + 1 == end_idx:
                print("## Ending online learning at iter {}\n".format(end_idx))
                break

    print("## End full training at time {}...time spent {}\n".format(
        datetime.datetime.now(), datetime.datetime.now() - learning_start_time))

    sys.stdout.flush()

    return interaction_records


def online_learning_misp_perfect(online_train_data_examples, online_train_raw_examples, data, user, agent,
                                 max_generation_length, model_save_dir, record_save_path, online_gold_record,
                                 update_iter, start_idx=0, metrics=None,
                                 database_username=None, database_password=None, database_timeout=None):
    assert params.ask_structure and params.user == "gold_sim" and params.err_detector == "perfect"

    class pseudoDatasetSplit: # a simplified class to be used as DatasetSplit
        def __init__(self, examples):
            self.examples = examples

    database_schema = read_schema(table_schema_path)

    def _evaluation(example, gen_sequence, raw_query):
        # check acc & add to record
        flat_pred = example.flatten_sequence(gen_sequence)
        pred_sql_str = ' '.join(flat_pred)
        assert len(example.identifier.split('/')) == 2
        database_id, interaction_id = example.identifier.split('/')
        postprocessed_sql_str = postprocess_one(pred_sql_str, database_schema[database_id])
        try:
            exact_score, partial_scores, hardness = evaluate_single(
                postprocessed_sql_str, raw_query, db_path, database_id, agent.world_model.kmaps)
        except:
            question = example.interaction.utterances[0].original_input_seq
            print("Exception in evaluate_single:\nidx: {}, db: {}, question: {}\np_str: {}\ng_str: {}\n".format(
                idx, database_id, " ".join(question), postprocessed_sql_str, raw_query))
            exact_score = 0.0
            partial_scores = "Exception"
            hardness = "Unknown"

        return exact_score, partial_scores, hardness

    def _multiset_difference(multiset1, multiset2):
        shared_set = multiset1 & multiset2
        one2two = list((multiset1 - shared_set).elements())
        two2one = list((multiset2 - shared_set).elements())

        return max(len(one2two), len(two2one))

    def _extract_element_multiset(tag_seq, extract_start_idx):
        """
        Extracting a dict of {tag: multiset of components}.

        SELECT_AGG_v2, WHERE_COL, WHERE_OP, WHERE_ROOT_TERM, GROUP_COL, HAV_AGG_v2,
        HAV_OP_v2, HAV_ROOT_TERM_v2, ORDER_AGG_v2, ORDER_DESC_ASC, ORDER_LIMIT, IUEN_v2,
        END_NESTED
        """
        tag2components = defaultdict(Counter)
        extract_idx = extract_start_idx

        while extract_idx < len(tag_seq):
            su = tag_seq[extract_idx]
            if su[0] in {SELECT_AGG_v2, ORDER_AGG_v2, HAV_AGG_v2}:  # check duplicates
                component = (su[1][0], su[1][1], su[2][-1], su[3])
                tag2components[su[0]][component] += 1
                extract_idx += 1

            elif su[0] in {WHERE_COL, GROUP_COL}:
                component = (su[1][0], su[1][1])
                tag2components[su[0]][component] += 1
                extract_idx += 1

            elif su[0] == WHERE_OP:
                component = (su[1][0][0], su[1][0][1], su[2][-1])
                tag2components[su[0]][component] += 1
                extract_idx += 1

            elif su[0] == HAV_OP_v2:
                component = (su[1][0][0], su[1][0][1], su[1][1][-1], su[1][2], su[2][-1])
                tag2components[su[0]][component] += 1
                extract_idx += 1

            elif su[0] in {ORDER_DESC_ASC, ORDER_LIMIT}:
                component = (su[1][0][0], su[1][0][1], su[1][1][-1], su[1][2], su[2])
                tag2components[su[0]][component] += 1
                extract_idx += 1

            elif su[0] in {HAV_ROOT_TERM_v2, WHERE_ROOT_TERM}:
                if su[0] == WHERE_ROOT_TERM:
                    component = (su[1][0][0], su[1][0][1], su[2][-1], su[3])
                else:
                    component = (su[1][0][0], su[1][0][1], su[1][1][-1], su[1][2], su[2][-1], su[3])
                tag2components[su[0]][component] += 1
                extract_idx += 1

                if su[3] == 'root':
                    subquery, extract_idx = _extract_element_multiset(tag_seq, extract_idx)
                    for subquery_tag, subquery_components in subquery.items():
                        tag2components[(component, subquery_tag)] = subquery_components

            elif su[0] == IUEN_v2:
                tag2components[su[0]][su[1][-1]] += 1
                extract_idx += 1

                context = (IUEN_v2, su[1][-1])
                subquery, extract_idx = _extract_element_multiset(tag_seq, extract_idx)
                for subquery_tag, subquery_components in subquery.items():
                    tag2components[(context, subquery_tag)] = subquery_components

            else:
                assert su[1] == "##END_NESTED##"
                extract_idx += 1
                return tag2components, extract_idx

        return tag2components, extract_idx

    num_total_examples = len(online_train_data_examples)
    online_train_data = data.get_all_interactions(pseudoDatasetSplit(online_train_data_examples))

    metrics_sums = {}
    for metric in metrics:
        metrics_sums[metric] = 0.

    interaction_records = []

    # if start_idx > 0:
    #     print("Loading interaction records from %s..." % record_save_path)
    #     interaction_records = json.load(open(record_save_path, 'r'))
    #     print("Record item size: %d " % len(interaction_records))
    assert start_idx == 0, "Not implemented for start_idx > 0!"

    learning_start_time = datetime.datetime.now()
    print("## Online starting time: {}".format(learning_start_time))
    count_exception, count_exit = 0, 0
    for idx, (raw_example, example) in enumerate(zip(online_train_raw_examples, online_train_data)):
        if idx < len(interaction_records):
            raise ValueError("Not implemented for start_idx > 0!")
        else:
            with torch.no_grad():
                input_item = agent.world_model.semparser.spider_single_turn_encoding(
                    example, max_generation_length)

                question = example.interaction.utterances[0].original_input_seq
                true_sql = example.interaction.utterances[0].original_gold_query
                print("\n" + "#" * 50)
                print("Example {}:".format(idx))
                print("NL input: {}".format(" ".join(question)))
                print("True SQL: {}".format(" ".join(true_sql)))

                g_sql = raw_example['sql']
                g_sql["extracted_clause_asterisk"] = extract_clause_asterisk(true_sql)
                g_sql["column_names_surface_form_to_id"] = input_item[-1].column_names_surface_form_to_id
                g_sql["base_vocab"] = agent.world_model.vocab

                complete_vocab = []
                for id in range(len(g_sql["base_vocab"])):
                    complete_vocab.append(g_sql["base_vocab"].id_to_token(id))
                id2col_name = {v: k for k, v in g_sql["column_names_surface_form_to_id"].items()}
                for id in range(len(g_sql["column_names_surface_form_to_id"])):
                    complete_vocab.append(id2col_name[id])

                try:
                    hyp = agent.world_model.decode(input_item, bool_verbal=False, dec_beam_size=1)[0]
                    print("-" * 50 + "\nBefore interaction: \ninitial SQL: {}".format(" ".join(hyp.sql)))
                except Exception: # tag_seq generation exception - e.g., when its syntax is wrong
                    count_exception += 1
                    # traceback.print_exc()
                    print("Decoding Exception (count = {}) in example {}!".format(count_exception, idx))
                    final_encoder_state, encoder_states, schema_states, max_generation_length, snippets, input_sequence, \
                        previous_queries, previous_query_states, input_schema = input_item
                    prediction = agent.world_model.semparser.decoder(
                        final_encoder_state,
                        encoder_states,
                        schema_states,
                        max_generation_length,
                        snippets=snippets,
                        input_sequence=input_sequence,
                        previous_queries=previous_queries,
                        previous_query_states=previous_query_states,
                        input_schema=input_schema,
                        dropout_amount=0.0)
                    print("-" * 50 + "\nBefore interaction: \ninitial SQL: {}".format(" ".join(prediction.sequence)))
                    sequence = prediction.sequence
                    probability = prediction.probability

                    g_sql.pop('base_vocab') # do not save it
                    record = {'nl': " ".join(question), 'true_sql': " ".join(true_sql),
                              'true_sql_i': "{}".format(g_sql),
                              'sql': "{}".format(sequence), 'dec_seq': "None",
                              'tag_seq': "None", 'logprob': "{}".format(np.log(probability)),
                              "questioned_indices": [], 'q_counter': 0, 'count_additional_q': 0,
                              'exit': False, 'exception': True,
                              'idx': idx}
                    interaction_records.append(record)
                    print("Example skipped!")
                else:
                    try:
                        new_hyp, bool_exit = agent.interactive_parsing_session(user, input_item, g_sql, hyp,
                                                                               bool_verbal=False)
                        if bool_exit:
                            count_exit += 1
                    except Exception:
                        count_exception += 1
                        # traceback.print_exc()
                        print("Interaction Exception (count = {}) in example {}!".format(count_exception, idx))
                        print("-" * 50 + "\nAfter interaction: \nfinal SQL: {}".format(" ".join(hyp.sql)))

                        g_sql.pop('base_vocab')  # do not save it
                        record = {'nl': " ".join(question), 'true_sql': " ".join(true_sql),
                                  'true_sql_i': "{}".format(g_sql),
                                  'sql': "{}".format(hyp.sql), 'dec_seq': "{}".format(hyp.dec_seq),
                                  'tag_seq': "{}".format(hyp.tag_seq), 'logprob': "{}".format(hyp.logprob),
                                  'exit': False, 'exception': True,
                                  'q_counter': user.q_counter,  # questions are still counted
                                  'questioned_indices': user.questioned_pointers,
                                  'questioned_tags': "{}".format(user.questioned_tags),
                                  'feedback_records': "{}".format(user.feedback_records),
                                  'count_additional_q': 0,
                                  'idx': idx}
                        interaction_records.append(record)
                        print("Example skipped!")
                    else:
                        hyp = new_hyp
                        print("-" * 50 + "\nAfter interaction: \nfinal SQL: {}".format(" ".join(hyp.sql)))

                        exact_score, partial_scores, hardness = _evaluation(example, hyp.sql, raw_example['query'])
                        print("exact_score: {}".format(exact_score))

                        # check missing/redundant part:
                        gen_tag_seq = hyp.tag_seq
                        gen_tag2components, _ = _extract_element_multiset(gen_tag_seq, 0)
                        print("DEBUG gen_tag2components:\n{}".format(gen_tag2components))

                        gold_tag_seq = eval(online_gold_record[idx]['tag_seq'])
                        if isinstance(gold_tag_seq[0], str):
                            print("DEBUG gold_tag2components: EXCEPTION")
                            count_additional_q = 0
                        else:
                            gold_tag2components, _ = _extract_element_multiset(gold_tag_seq, 0)
                            print("DEBUG gold_tag2components:\n{}".format(gold_tag2components))

                            count_additional_q = 0
                            all_tags = set(gen_tag2components.keys()).union(set(gold_tag2components.keys()))
                            for tag in all_tags:
                                gen_components = gen_tag2components[tag]
                                gold_components = gold_tag2components[tag]
                                count_additional_q += _multiset_difference(gen_components, gold_components)

                        if count_additional_q > 0:
                            print("count_additional_q: %d" % count_additional_q)

                        g_sql.pop('base_vocab')  # do not save it
                        record = {'nl': " ".join(question), 'true_sql': " ".join(true_sql),
                                  'true_sql_i': "{}".format(g_sql),
                                  'sql': "{}".format(hyp.sql), 'dec_seq': "{}".format(hyp.dec_seq),
                                  'tag_seq': "{}".format(hyp.tag_seq), 'logprob': "{}".format(hyp.logprob),
                                  'exit': bool_exit, 'exception': False, 'q_counter': user.q_counter,
                                  'questioned_indices': user.questioned_pointers,
                                  'questioned_tags': "{}".format(user.questioned_tags),
                                  'feedback_records': "{}".format(user.feedback_records),
                                  'count_additional_q': count_additional_q,
                                  'exact_score': exact_score, 'hardness': hardness,
                                  'idx': idx}
                        interaction_records.append(record)

                    sequence = hyp.sql
                    probability = np.exp(hyp.logprob)

        original_utt = example.interaction.utterances[0]

        gold_query = original_utt.gold_query_to_use
        original_gold_query = original_utt.original_gold_query

        gold_table = original_utt.gold_sql_results
        gold_queries = [q[0] for q in original_utt.all_gold_queries]
        gold_tables = [q[1] for q in original_utt.all_gold_queries]

        flat_sequence = example.flatten_sequence(sequence)

        update_sums(metrics,
                    metrics_sums,
                    sequence,
                    flat_sequence,
                    gold_query,
                    original_gold_query,
                    database_username=database_username,
                    database_password=database_password,
                    database_timeout=database_timeout,
                    gold_table=gold_table)

        sys.stdout.flush()

        if (idx+1) % update_iter == 0 or (idx+1) == num_total_examples:  # update model
            print("\n~~~\nCurrent interaction performance (iter {}): ".format(idx+1))  # interaction so far
            eval_results = construct_averages(metrics_sums, idx+1)
            for name, value in eval_results.items():
                print(name.name + ":\t" + "%.2f" % value)

            # report q counts
            q_count = sum([item['q_counter'] + item['count_additional_q'] for item in interaction_records])
            print("## End update at iter {}, anno_cost {}\n".format((idx+1), q_count))

            print("Saving interaction records to %s..." % record_save_path)
            json.dump(interaction_records, open(record_save_path, 'w'), indent=4)

            # loading model
            model_dir = os.path.join(model_save_dir, '%d/' % (idx + 1))
            print("Loading model from %s..." % model_dir)
            model_path = os.path.join(model_dir, "model_best.pt")
            agent.world_model.semparser.load(model_path)

    print("Saving interaction records to %s..." % record_save_path)
    json.dump(interaction_records, open(record_save_path, 'w'), indent=4)

    return interaction_records


def extract_clause_asterisk(g_sql_toks):
    """
    This function extracts {clause keyword: tab_col_item with asterisk (*)}.
    Keywords include: SELECT/HAV/ORDER_AGG_v2.
    A tab_col_item lookds like "*" or "tab_name.*".

    The output will be used to simulate user evaluation and selections.
    The motivation is that the structured "g_sql" does not contain tab_name for *, so the simulator cannot decide the
    right decision precisely.
    :param g_sql_toks: the preprocessed gold sql tokens from EditSQL.
    :return: A dict of {clause keyword: tab_col_item with asterisk (*)}.
    """
    kw2item = defaultdict(list)

    keyword = None
    for tok in g_sql_toks:
        if tok in {'select', 'having', 'order_by', 'where', 'group_by'}:
            keyword = tok
        elif keyword in {'select', 'having', 'order_by'} and (tok == "*" or re.findall("\.\*", tok)):
            kw2item[keyword].append(tok)

    kw2item = dict(kw2item)
    for kw, item in kw2item.items():
        try:
            assert len(item) <= 1
        except:
            print("\nException in clause asterisk extraction:\ng_sql_toks: {}\nkw: {}, item: {}\n".format(
                g_sql_toks, kw, item))
        kw2item[kw] = item[0]

    return kw2item


def interaction(raw_proc_example_pairs, user, agent, max_generation_length, output_path,
                metrics=None, database_username=None, database_password=None, database_timeout=None,
                write_results=False, compute_metrics=False, bool_interaction=True):
    """ Evaluates a sample of interactions. """
    database_schema = read_schema(table_schema_path)

    def _evaluation(example, gen_sequence, raw_query):
        # check acc & add to record
        flat_pred = example.flatten_sequence(gen_sequence)
        pred_sql_str = ' '.join(flat_pred)
        assert len(example.identifier.split('/')) == 2
        database_id, interaction_id = example.identifier.split('/')
        postprocessed_sql_str = postprocess_one(pred_sql_str, database_schema[database_id])
        try:
            exact_score, partial_scores, hardness = evaluate_single(
                postprocessed_sql_str, raw_query, db_path, database_id, agent.world_model.kmaps)
        except:
            question = example.interaction.utterances[0].original_input_seq
            print("Exception in evaluate_single:\nidx: {}, db: {}, question: {}\np_str: {}\ng_str: {}\n".format(
                idx, database_id, " ".join(question), postprocessed_sql_str, raw_query))
            exact_score = 0.0
            partial_scores = "Exception"
            hardness = "Unknown"

        return exact_score, partial_scores, hardness

    if write_results:
        predictions_file = open(output_path, "w")
        print("Predicting with file: " + output_path)

    metrics_sums = {}
    for metric in metrics:
        metrics_sums[metric] = 0.

    interaction_records = []

    starting_time = datetime.datetime.now()
    count_exception, count_exit = 0, 0
    for idx, (raw_example, example) in enumerate(raw_proc_example_pairs):
        with torch.no_grad():
            input_item = agent.world_model.semparser.spider_single_turn_encoding(
                example, max_generation_length)

            question = example.interaction.utterances[0].original_input_seq
            true_sql = example.interaction.utterances[0].original_gold_query
            print("\n" + "#" * 50)
            print("Example {}:".format(idx))
            print("NL input: {}".format(" ".join(question)))
            print("True SQL: {}".format(" ".join(true_sql)))

            g_sql = raw_example['sql']
            g_sql["extracted_clause_asterisk"] = extract_clause_asterisk(true_sql)
            g_sql["column_names_surface_form_to_id"] = input_item[-1].column_names_surface_form_to_id
            g_sql["base_vocab"] = agent.world_model.vocab

            try:
                hyp = agent.world_model.decode(input_item, bool_verbal=False, dec_beam_size=1)[0]
                print("-" * 50 + "\nBefore interaction: \ninitial SQL: {}".format(" ".join(hyp.sql)))
            except Exception: # tag_seq generation exception - e.g., when its syntax is wrong
                count_exception += 1
                print("Decoding Exception (count = {}) in example {}!".format(count_exception, idx))
                final_encoder_state, encoder_states, schema_states, max_generation_length, snippets, input_sequence, \
                    previous_queries, previous_query_states, input_schema = input_item
                prediction = agent.world_model.semparser.decoder(
                    final_encoder_state,
                    encoder_states,
                    schema_states,
                    max_generation_length,
                    snippets=snippets,
                    input_sequence=input_sequence,
                    previous_queries=previous_queries,
                    previous_query_states=previous_query_states,
                    input_schema=input_schema,
                    dropout_amount=0.0)
                print("-" * 50 + "\nBefore interaction: \ninitial SQL: {}".format(" ".join(prediction.sequence)))
                sequence = prediction.sequence
                probability = prediction.probability

                exact_score, partial_scores, hardness = _evaluation(example, sequence, raw_example['query'])

                record = {'nl': " ".join(question), 'true_sql': " ".join(true_sql),
                          'true_sql_i': "{}".format(g_sql),
                          'sql': "{}".format(sequence), 'dec_seq': "None",
                          'tag_seq': "None", 'logprob': "{}".format(np.log(probability)),
                          "questioned_indices": [], 'q_counter': 0,
                          'exact_score': exact_score, 'partial_scores': "{}".format(partial_scores),
                          'hardness': hardness, 'exit': False, 'exception': True, 'idx': idx}
                interaction_records.append(record)
            else:
                if not bool_interaction:
                    # print("DEBUG:")
                    # _, eval_outputs, true_sels, true_sus = user.error_evaluator.compare(
                    #     g_sql, 0, hyp.tag_seq, bool_return_true_selections=True,
                    #     bool_return_true_semantic_units=True)
                    # for unit, eval_output, true_sel, true_su in zip(hyp.tag_seq, eval_outputs, true_sels, true_sus):
                    #     print("SU: {}, EVAL: {}, TSEL: {}, TSU: {}".format(unit, eval_output, true_sel, true_su))

                    exact_score, partial_scores, hardness = _evaluation(example, hyp.sql, raw_example['query'])

                    record = {'nl': " ".join(question), 'true_sql': " ".join(true_sql),
                              'true_sql_i': "{}".format(g_sql),
                              'sql': "{}".format(hyp.sql), 'dec_seq': "{}".format(hyp.dec_seq),
                              'tag_seq': "{}".format(hyp.tag_seq), 'logprob': "{}".format(hyp.logprob),
                              "questioned_indices": [], 'q_counter': 0,
                              'exact_score': exact_score, 'partial_scores': "{}".format(partial_scores),
                              'hardness': hardness, 'exit': False, 'exception': False, 'idx': idx}
                    interaction_records.append(record)
                else:
                    try:
                        new_hyp, bool_exit = agent.interactive_parsing_session(user, input_item, g_sql, hyp,
                                                                               bool_verbal=False)
                        if bool_exit:
                            count_exit += 1
                    except Exception:
                        count_exception += 1
                        print("Interaction Exception (count = {}) in example {}!".format(count_exception, idx))
                        bool_exit = False
                    else:
                        hyp = new_hyp

                    print("-" * 50 + "\nAfter interaction: \nfinal SQL: {}".format(" ".join(hyp.sql)))

                    exact_score, partial_scores, hardness = _evaluation(example, hyp.sql, raw_example['query'])

                    record = {'nl': " ".join(question), 'true_sql': " ".join(true_sql),
                              'true_sql_i': "{}".format(g_sql),
                              'sql': hyp.sql, 'dec_seq': "{}".format(hyp.dec_seq),
                              'tag_seq': "{}".format(hyp.tag_seq), 'logprob': "{}".format(hyp.logprob),
                              'q_counter': user.q_counter,
                              'questioned_indices': user.questioned_pointers,
                              'questioned_tags': "{}".format(user.questioned_tags),
                              'feedback_records': "{}".format(user.feedback_records),
                              'exact_score': exact_score, 'partial_scores': "{}".format(partial_scores),
                              'hardness': hardness, 'exit': bool_exit, 'exception': False, 'idx': idx}
                    interaction_records.append(record)

                sequence = hyp.sql
                probability = np.exp(hyp.logprob)

        original_utt = example.interaction.utterances[0]

        gold_query = original_utt.gold_query_to_use
        original_gold_query = original_utt.original_gold_query

        gold_table = original_utt.gold_sql_results
        gold_queries = [q[0] for q in original_utt.all_gold_queries]
        gold_tables = [q[1] for q in original_utt.all_gold_queries]

        flat_sequence = example.flatten_sequence(sequence)

        if write_results:
            write_prediction(
                predictions_file,
                identifier=example.identifier,
                input_seq=question,
                probability=probability,
                prediction=sequence,
                flat_prediction=flat_sequence,
                gold_query=gold_query,
                flat_gold_queries=gold_queries,
                gold_tables=gold_tables,
                index_in_interaction=0,
                database_username=database_username,
                database_password=database_password,
                database_timeout=database_timeout,
                compute_metrics=compute_metrics)

        update_sums(metrics,
                    metrics_sums,
                    sequence,
                    flat_sequence,
                    gold_query,
                    original_gold_query,
                    database_username=database_username,
                    database_password=database_password,
                    database_timeout=database_timeout,
                    gold_table=gold_table)

        sys.stdout.flush()

    if write_results:
        predictions_file.close()
    time_spent = datetime.datetime.now() - starting_time

    # stats
    q_count = sum([item['q_counter'] for item in interaction_records])
    print("#questions: {}, #questions per example: {:.3f}.".format(
        q_count, q_count * 1.0 / len(interaction_records)))
    print("#exit: {}".format(count_exit))
    print("#time spent: {}".format(time_spent))

    eval_results = construct_averages(metrics_sums, len(raw_proc_example_pairs))
    for name, value in eval_results.items():
        print(name.name + ":\t" + "%.2f" % value)

    exact_acc = np.average([item['exact_score'] for item in interaction_records])
    print("Exact_acc: {}".format(exact_acc))

    return eval_results, interaction_records


def real_user_interaction(raw_proc_example_pairs, user, agent, max_generation_length, record_save_path):

    database_schema = read_schema(table_schema_path)

    interaction_records = []
    st = 0
    time_spent = datetime.timedelta()
    count_exception, count_exit = 0, 0

    if os.path.isfile(record_save_path):
        saved_results = json.load(open(record_save_path, 'r'))
        st = saved_results['st']
        interaction_records = saved_results['interaction_records']
        count_exit = saved_results['count_exit']
        count_exception = saved_results['count_exception']
        time_spent = datetime.timedelta(pytimeparse.parse(saved_results['time_spent']))

    for idx, (raw_example, example) in enumerate(raw_proc_example_pairs):
        if idx < st:
            continue

        with torch.no_grad():
            input_item = agent.world_model.semparser.spider_single_turn_encoding(
                example, max_generation_length)

            question = example.interaction.utterances[0].original_input_seq
            true_sql = example.interaction.utterances[0].original_gold_query

            g_sql = raw_example['sql']
            g_sql["extracted_clause_asterisk"] = extract_clause_asterisk(true_sql)
            g_sql["column_names_surface_form_to_id"] = input_item[-1].column_names_surface_form_to_id
            g_sql["base_vocab"] = agent.world_model.vocab

            assert len(example.identifier.split('/')) == 2
            database_id, interaction_id = example.identifier.split('/')

            os.system('clear')  # clear screen
            print_header(len(raw_proc_example_pairs) - idx, bool_table_color=True)  # interface header

            print(bcolors.BOLD + "Suppose you are given some tables with the following " +
                  bcolors.BLUE + "headers" + bcolors.ENDC +
                  bcolors.BOLD + ":" + bcolors.ENDC)
            user.show_table(database_id)  # print table

            print(bcolors.BOLD + "\nAnd you want to answer the following " +
                  bcolors.PINK + "question" + bcolors.ENDC +
                  bcolors.BOLD + " based on this table:" + bcolors.ENDC)
            print(bcolors.PINK + bcolors.BOLD + " ".join(question) + bcolors.ENDC + "\n")
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
            init_hyp = agent.world_model.decode(input_item, bool_verbal=False, dec_beam_size=1)[0]

            try:
                hyp, bool_exit = agent.real_user_interactive_parsing_session(
                    user, input_item, g_sql, init_hyp, bool_verbal=False)
                bool_exception = False
                if bool_exit:
                    count_exit += 1
            except Exception:
                count_exception += 1
                print("Interaction Exception (count = {}) in example {}!".format(count_exception, idx))
                bool_exit = False
                bool_exception = True
                hyp = init_hyp

            print("\nPredicted SQL: {}".format(" ".join(hyp.sql)))
            per_time_spent = datetime.datetime.now() - start_time
            time_spent += per_time_spent
            print("Your time spent: {}".format(per_time_spent))

            # post survey
            print("-" * 50)
            print("Post-study Survey: ")
            bool_unclear = input("Is the " + bcolors.BOLD + bcolors.PINK + "initial question" +
                                     bcolors.ENDC + " clear?\nPlease enter y/n: ")
            while bool_unclear not in {'y', 'n'}:
                bool_unclear = input("Is the " + bcolors.BOLD + bcolors.PINK + "initial question" +
                                         bcolors.ENDC + " clear?\nPlease enter y/n: ")
            print("-" * 50)

            # check acc & add to record
            flat_pred = example.flatten_sequence(hyp.sql)
            pred_sql_str = ' '.join(flat_pred)
            postprocessed_sql_str = postprocess_one(pred_sql_str, database_schema[database_id])
            exact_score, partial_scores, hardness = evaluate_single(
                postprocessed_sql_str, raw_example['query'], db_path, database_id, agent.world_model.kmaps)

            g_sql.pop("base_vocab") # do not save it
            record = {'nl': " ".join(question), 'true_sql': " ".join(true_sql),
                      'true_sql_i': "{}".format(g_sql),
                      'init_sql': init_hyp.sql,
                      'sql': hyp.sql, 'dec_seq': "{}".format(hyp.dec_seq),
                      'tag_seq': "{}".format(hyp.tag_seq), 'logprob': "{}".format(hyp.logprob),
                      'exit': bool_exit, 'exception': bool_exception, 'q_counter': user.q_counter,
                      'questioned_indices': user.questioned_pointers,
                      'questioned_tags': "{}".format(user.questioned_tags),
                      'per_time_spent': str(per_time_spent), 'bool_unclear': bool_unclear,
                      'feedback_records': "{}".format(user.feedback_records),
                      'undo_semantic_units': "{}".format(user.undo_semantic_units),
                      'exact_score': exact_score, 'partial_scores': "{}".format(partial_scores), 'hardness': hardness,
                      'idx': idx}
            interaction_records.append(record)

            print("Saving records...")
            json.dump({'interaction_records': interaction_records,
                       'st': idx+1, 'time_spent': str(time_spent),
                       'count_exit': count_exit, 'count_exception': count_exception},
                      open(record_save_path, "w"), indent=4)

            end_signal = input(bcolors.GREEN + bcolors.BOLD +
                                   "Next? Press 'Enter' to continue, Ctrl+C to quit." + bcolors.ENDC)
            if end_signal != "":
                return

    print(bcolors.RED + bcolors.BOLD + "Congratulations! You have completed all your task!" + bcolors.ENDC)
    print("Your average time spent: {}".format((time_spent / len(raw_proc_example_pairs))))
    print("You exited %d times." % count_exit)
    print("%d exceptions occurred." % count_exception)


if __name__ == "__main__":
    params = interpret_args()

    # Prepare the dataset into the proper form.
    data = atis_data.ATISDataset(params)

    table_schema_path = os.path.join(params.raw_data_directory, "tables.json")
    gold_path = os.path.join(params.raw_data_directory, "dev_gold.sql")
    db_path = os.path.join(os.path.dirname(params.raw_data_directory), "database/")

    db_list = []
    with open(gold_path) as f:
        for line in f:
            line_split = line.strip().split('\t')
            if len(line_split) != 2:
                continue
            db = line.strip().split('\t')[1]
            if db not in db_list:
                db_list.append(db)

    if params.job == "online_learning" and params.supervision == 'full_train':
        model = None # the model will be renewed immediately in online training
    else:
        # model loading
        model = SchemaInteractionATISModel(
            params,
            data.input_vocabulary,
            data.output_vocabulary,
            data.output_vocabulary_schema,
            data.anonymizer if params.anonymize and params.anonymization_scoring else None)

        model.load(os.path.join(params.logdir, "model_best.pt"))
        model = model.to(device)
        model.eval()

    def create_new_model(old_model):
        del old_model
        torch.cuda.empty_cache()
        new_model = SchemaInteractionATISModel(
            params,
            data.input_vocabulary,
            data.output_vocabulary,
            data.output_vocabulary_schema,
            data.anonymizer if params.anonymize and params.anonymization_scoring else None)
        new_model.to(device)
        return new_model

    print("ask_structure: {}".format(params.ask_structure))
    question_generator = QuestionGenerator(bool_structure_question=params.ask_structure)

    if params.err_detector == 'any':
        error_detector = ErrorDetectorProbability(1.1)  # ask any SU
    elif params.err_detector.startswith('prob='):
        prob = float(params.err_detector[5:])
        error_detector = ErrorDetectorProbability(prob)
        print("Error Detector: probability threshold = %.3f" % prob)
        assert params.passes == 1, "Error: For prob-based evaluation, set --passes 1."
    elif params.err_detector.startswith('stddev='):
        stddev = float(params.err_detector[7:])
        error_detector = ErrorDetectorBayesDropout(stddev)
        print("Error Detector: Bayesian Dropout Stddev threshold = %.3f" % stddev)
        print("num passes: %d, dropout rate: %.3f" % (params.passes, params.dropout))
        assert params.passes > 1, "Error: For dropout-based evaluation, set --passes 10."
    elif params.err_detector == "perfect":
        error_detector = ErrorDetectorSim()
        print("Error Detector: using a simulated perfect detector.")
    else:
        raise Exception("Invalid error detector setup %s!" % params.err_detector)

    if params.num_options == 'inf':
        print("WARNING: Unlimited options!")
        num_options = np.inf
    else:
        num_options = int(params.num_options)
        print("num_options: {}".format(num_options))

    kmaps = build_foreign_key_map_from_json(table_schema_path)

    world_model = WorldModel(model, num_options, kmaps, params.passes, params.dropout,
                             bool_structure_question=params.ask_structure)

    print("friendly_agent: {}".format(params.friendly_agent))
    agent = Agent(world_model, error_detector, question_generator,
                  bool_mistake_exit=params.friendly_agent,
                  bool_structure_question=params.ask_structure)

    # environment setup: user simulator
    error_evaluator = ErrorEvaluator()

    if params.user == "real":
        def get_table_dict(table_data_path):
            data = json.load(open(table_data_path))
            table = dict()
            for item in data:
                table[item["db_id"]] = item
            return table

        user = RealUser(error_evaluator, get_table_dict(table_schema_path), db_path)
    elif params.user == "gold_sim":
        user = GoldUserSim(error_evaluator, bool_structure_question=params.ask_structure)
    else:
        user = UserSim(error_evaluator, bool_structure_question=params.ask_structure)

    # load raw data
    raw_train_examples = json.load(open(os.path.join(params.raw_data_directory, "train_reordered.json")))
    raw_valid_examples = json.load(open(os.path.join(params.raw_data_directory, "dev_reordered.json")))

    if params.job == 'online_learning':
        update_iter = params.update_iter #1000
        print("## data_seed: {}".format(params.data_seed))
        print("## supervision: {}".format(params.supervision))
        print("## start: {}, end: {}".format(params.start_iter, params.end_iter))
        print("## update_iter: {}".format(update_iter)) # fixed value

        # load indices for initial/online training instances
        if params.setting == 'online_pretrain_10p':
            train_indices = json.load(open(os.path.join(os.path.dirname(params.raw_train_filename),
                                                        "train_indices_10p.json")))
        else:
            raise Exception("Invalid params.setting=%s!" % params.setting)

        init_raw_train_examples = [raw_train_examples[idx] for idx in train_indices["init"]]
        online_raw_train_examples = [raw_train_examples[idx] for idx in
                                     train_indices["online_seed%d" % params.data_seed]]
        print("\n## setting{}: size {}".format(params.setting, len(init_raw_train_examples)))

        # in case of vocab mismatching, we should re-process the full training data from spider_data_removefrom
        database_schema = None
        if params.database_schema_filename:
            if 'removefrom' not in params.data_directory:
                database_schema, column_names_surface_form, column_names_embedder_input = data.read_database_schema_simple(
                    params.database_schema_filename)
            else:
                database_schema, column_names_surface_form, column_names_embedder_input = data.read_database_schema(
                    params.database_schema_filename)

        int_load_function = load_function(params,
                                          data.entities_dictionary,
                                          data.anonymizer,
                                          database_schema=database_schema)

        # full training examples
        full_train_examples = ds.DatasetSplit(
            os.path.join(params.data_directory, "train_full.pkl"),
            os.path.join(os.path.dirname(params.raw_train_filename), "train.pkl"),
            int_load_function).examples

        init_train_examples = [full_train_examples[idx] for idx in train_indices["init"]]
        online_train_examples = [full_train_examples[idx] for idx in train_indices["online_seed%d" % params.data_seed]]

        # load full annotation cost: truth_annotation_cost
        train_gold_record_path = os.path.join(os.path.dirname(params.logdir),
                                              "logs_spider_editsql/records_train_nointeract_gold.json")
        train_gold_records = json.load(open(train_gold_record_path, "r"))
        full_annotation_costs = [len(list(filter(lambda x: x[0] != OUTSIDE, eval(item["tag_seq"]))))
                                 for item in train_gold_records]
        init_train_full_annotation_costs = [full_annotation_costs[idx] for idx in train_indices['init']]
        online_train_full_annotation_costs = [full_annotation_costs[idx] for idx in train_indices["online_seed%d" % params.data_seed]]

        print("## initial annotation cost: {}".format(sum(init_train_full_annotation_costs)))

        # online learning
        if params.supervision == "full_expert":
            model_save_dir = os.path.join(
                params.logdir, "checkpoint_online_SUP{}_ITER{}_SEED{}".format(
                    params.supervision, update_iter, params.data_seed))
            if not os.path.isdir(model_save_dir):
                os.mkdir(model_save_dir)

            online_learning_full(online_train_examples, init_train_examples, data, agent,
                                 model_save_dir, update_iter, create_new_model,
                                 online_train_full_annotation_costs,
                                 start_idx=params.start_iter, end_idx=params.end_iter)

        elif params.supervision == 'misp_neil_perfect':
            model_save_dir = os.path.join(
                params.logdir, "checkpoint_online_SUPfull_ITER{}_DATASEED{}".format(
                    update_iter, params.data_seed))

            online_train_gold_record = [train_gold_records[idx] for idx in
                                        train_indices["online_seed%d" % params.data_seed]]

            online_learning_misp_perfect(online_train_examples, online_raw_train_examples, data, user, agent,
                                         params.eval_maximum_sql_length, model_save_dir, params.output_path,
                                         online_train_gold_record, update_iter, metrics=FINAL_EVAL_METRICS,
                                         database_username=params.database_username,
                                         database_password=params.database_password,
                                         database_timeout=params.database_timeout)

        elif params.supervision.startswith("misp_neil"): # exclude misp_neil_perfect
            def utterance_revision(utterance, new_output_sequences):
                utterance.process_gold_seq(new_output_sequences, data.entities_dictionary,
                                           utterance.available_snippets,
                                           params.anonymize, data.anonymizer, {})
                return utterance

            model_save_dir = os.path.join(
                params.logdir, "checkpoint_online_SUP{}_OP{}_ED{}{}{}{}ITER{}_DATASEED{}".format(
                    params.supervision, params.num_options, params.err_detector,
                    ("_FRIENDLY" if params.friendly_agent else ""),
                    ("_GoldUser" if params.user == "gold_sim" else ""),
                    ("_ASKSTRUCT" if params.ask_structure else ""),
                    update_iter, params.data_seed))
            if not os.path.isdir(model_save_dir):
                os.mkdir(model_save_dir)

            if params.start_iter > 0:
                print("Loading previous checkpoints at iter {}...".format(params.start_iter))
                model_path = os.path.join(model_save_dir, "%d" % params.start_iter, "model_best.pt")
                agent.world_model.semparser.load(model_path)

            online_learning(online_train_examples, init_train_examples, online_raw_train_examples, data,
                            user, agent, params.eval_maximum_sql_length, model_save_dir, params.output_path,
                            update_iter, create_new_model, utterance_revision, start_idx=params.start_iter,
                            end_idx=params.end_iter, metrics=FINAL_EVAL_METRICS,
                            database_username=params.database_username,
                            database_password=params.database_password,
                            database_timeout=params.database_timeout,
                            bool_interaction=True, supervision=params.supervision)

        elif params.supervision.startswith('self-train'):
            def utterance_revision(utterance, new_output_sequences):
                utterance.process_gold_seq(new_output_sequences, data.entities_dictionary,
                                           utterance.available_snippets,
                                           params.anonymize, data.anonymizer, {})
                return utterance

            model_save_dir = os.path.join(
                params.logdir, "checkpoint_online_SUP{}_ITER{}_DATASEED{}".format(
                    params.supervision, update_iter, params.data_seed))
            if not os.path.isdir(model_save_dir):
                os.mkdir(model_save_dir)

            if params.start_iter > 0:
                print("Loading previous checkpoints at iter {}...".format(params.start_iter))
                model_path = os.path.join(model_save_dir, "%d" % params.start_iter, "model_best.pt")
                agent.world_model.semparser.load(model_path)

            online_learning_self_training(params.supervision, online_train_examples, init_train_examples,
                                          online_raw_train_examples, data,
                                          agent, params.eval_maximum_sql_length, model_save_dir, params.output_path,
                                          update_iter, create_new_model, utterance_revision,
                                          start_idx=params.start_iter,
                                          end_idx=params.end_iter, metrics=FINAL_EVAL_METRICS,
                                          database_username=params.database_username,
                                          database_password=params.database_password,
                                          database_timeout=params.database_timeout)

        else:
            assert params.supervision in {"bin_feedback", "bin_feedback_expert"}

            def utterance_revision(utterance, new_output_sequences):
                utterance.process_gold_seq(new_output_sequences, data.entities_dictionary,
                                           utterance.available_snippets,
                                           params.anonymize, data.anonymizer, {})
                return utterance

            model_save_dir = os.path.join(
                params.logdir, "checkpoint_online_SUP{}_ITER{}_SEED{}".format(
                    params.supervision, update_iter, params.data_seed))
            if not os.path.isdir(model_save_dir):
                os.mkdir(model_save_dir)

            if params.start_iter > 0:
                print("Loading previous checkpoints at iter {}...".format(params.start_iter))
                model_path = os.path.join(model_save_dir, "%d" % params.start_iter, "model_best.pt")
                agent.world_model.semparser.load(model_path)

            online_learning_bin_feedback(params.supervision, online_train_examples, init_train_examples,
                                         online_raw_train_examples, data,
                                         agent, params.eval_maximum_sql_length, model_save_dir, params.output_path,
                                         update_iter, create_new_model, utterance_revision,
                                         online_train_full_annotation_costs,
                                         start_idx=params.start_iter,
                                         end_idx=params.end_iter, metrics=FINAL_EVAL_METRICS,
                                         database_username=params.database_username,
                                         database_password=params.database_password,
                                         database_timeout=params.database_timeout)

    else:
        if params.user == "real":
            user_study_indices = json.load(open(os.path.join(
                params.raw_data_directory, "user_study_indices.json"), "r")) # random.seed(1234), 100
            reorganized_data = list(zip(raw_valid_examples, data.get_all_interactions(data.valid_data)))
            reorganized_data = [reorganized_data[idx] for idx in user_study_indices]

            real_user_interaction(reorganized_data, user, agent, params.eval_maximum_sql_length, params.output_path)

        else:
            reorganized_data = list(zip(raw_valid_examples, data.get_all_interactions(data.valid_data)))

            eval_file = params.output_path[:-5] + "_prediction.json"
            eval_results, interaction_records = interaction(
                reorganized_data, user, agent,
                params.eval_maximum_sql_length,
                eval_file,
                metrics=FINAL_EVAL_METRICS,
                database_username=params.database_username,
                database_password=params.database_password,
                database_timeout=params.database_timeout,
                write_results=True,
                compute_metrics=params.compute_metrics,
                bool_interaction=True)
            json.dump(interaction_records, open(params.output_path, "w"), indent=4)


