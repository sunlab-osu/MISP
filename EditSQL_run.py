"""Contains a main function for training and/or evaluating a model."""

import os
import sys

import numpy as np
import random
import subprocess

from EditSQL.parse_args import interpret_args
from EditSQL.postprocess_eval import read_schema, read_prediction, postprocess, write_and_evaluate
from EditSQL.eval_scripts.evaluation import build_foreign_key_map_from_json, \
    evaluate as eval_script_evaluate
from EditSQL.data_util import atis_data
from EditSQL.model.schema_interaction_model import SchemaInteractionATISModel
from EditSQL.logger import Logger
from EditSQL.model_util import Metrics, evaluate_utterance_sample, evaluate_interaction_sample, \
    train_epoch_with_utterances, train_epoch_with_interactions, evaluate_using_predicted_queries

import torch
import pdb

np.random.seed(0)
random.seed(0)

VALID_EVAL_METRICS = [Metrics.LOSS, Metrics.TOKEN_ACCURACY, Metrics.STRING_ACCURACY]
TRAIN_EVAL_METRICS = [Metrics.LOSS, Metrics.TOKEN_ACCURACY, Metrics.STRING_ACCURACY]
FINAL_EVAL_METRICS = [Metrics.STRING_ACCURACY, Metrics.TOKEN_ACCURACY]



def train_v1(model, data, params):
    """ Trains a model.

    Inputs:
        model (ATISModel): The model to train.
        data (ATISData): The data that is used to train.
        params (namespace): Training parameters.
    """
    # Get the training batches.
    log = Logger(os.path.join(params.logdir, params.logfile), "w")
    num_train_original = atis_data.num_utterances(data.train_data)
    log.put("Original number of training utterances:\t"
            + str(num_train_original))

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

    log.put(
        "Actual number of used training examples:\t" +
        str(num_train_examples))
    log.put("(Shortened by output limit of " +
            str(maximum_output_length) +
            ")")
    log.put("Number of steps per epoch:\t" + str(num_steps_per_epoch))
    log.put("Batch size:\t" + str(batch_size))

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
    patience = params.initial_patience
    learning_rate_coefficient = 1.
    previous_epoch_loss = float('inf')
    maximum_validation_accuracy = 0.
    maximum_string_accuracy = 0.

    countdown = int(patience)

    if params.scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model.trainer, mode='min', )

    keep_training = True
    while keep_training:
        log.put("Epoch:\t" + str(epochs))
        model.set_dropout(params.dropout_amount)

        if not params.scheduler:
            model.set_learning_rate(learning_rate_coefficient * params.initial_learning_rate)

        # Run a training step.
        if params.interaction_level:
            epoch_loss = train_epoch_with_interactions(
                train_batches,
                params,
                model,
                randomize=not params.deterministic)
        else:
            epoch_loss = train_epoch_with_utterances(
                train_batches,
                model,
                randomize=not params.deterministic)

        log.put("train epoch loss:\t" + str(epoch_loss))

        model.set_dropout(0.)

        # Run an evaluation step on a sample of the training data.
        train_eval_results = eval_fn(training_sample,
                                     model,
                                     params.train_maximum_sql_length,
                                     name=os.path.join(params.logdir, "train-eval"),
                                     write_results=True,
                                     gold_forcing=True,
                                     metrics=TRAIN_EVAL_METRICS)[0]

        for name, value in train_eval_results.items():
            log.put(
                "train final gold-passing " +
                name.name +
                ":\t" +
                "%.2f" %
                value)

        # Run an evaluation step on the validation set.
        valid_eval_results = eval_fn(valid_examples,
                                     model,
                                     params.eval_maximum_sql_length,
                                     name=os.path.join(params.logdir, "valid-eval"),
                                     write_results=True,
                                     gold_forcing=True,
                                     metrics=VALID_EVAL_METRICS)[0]
        for name, value in valid_eval_results.items():
            log.put("valid gold-passing " + name.name + ":\t" + "%.2f" % value)

        valid_loss = valid_eval_results[Metrics.LOSS]
        valid_token_accuracy = valid_eval_results[Metrics.TOKEN_ACCURACY]
        string_accuracy = valid_eval_results[Metrics.STRING_ACCURACY]

        if params.scheduler:
            scheduler.step(valid_loss)

        if valid_loss > previous_epoch_loss:
            learning_rate_coefficient *= params.learning_rate_ratio
            log.put(
                "learning rate coefficient:\t" +
                str(learning_rate_coefficient))

        previous_epoch_loss = valid_loss
        saved = False
        
        if not saved and string_accuracy > maximum_string_accuracy:
            maximum_string_accuracy = string_accuracy
            patience = patience * params.patience_ratio
            countdown = int(patience)
            last_save_file = os.path.join(params.logdir, "save_" + str(epochs))
            model.save(last_save_file)

            log.put(
                "maximum string accuracy:\t" +
                str(maximum_string_accuracy))
            log.put("patience:\t" + str(patience))
            log.put("save file:\t" + str(last_save_file))

        if countdown <= 0:
            keep_training = False

        countdown -= 1
        log.put("countdown:\t" + str(countdown))
        log.put("")

        epochs += 1

    log.put("Finished training!")
    log.close()

    return last_save_file


def train(model, data, params):
    """ Trains a model.

    Inputs:
        model (ATISModel): The model to train.
        data (ATISData): The data that is used to train.
        params (namespace): Training parameters.
    """
    if "data_clean" in params.raw_train_filename:
        raw_data_directory = "EditSQL/data_clean/"
    else:
        raw_data_directory = "EditSQL/data/"

    table_schema_path = os.path.join(raw_data_directory, "spider", "tables.json")
    gold_path = os.path.join(raw_data_directory, "spider", "dev_gold.sql")
    db_path = os.path.join(raw_data_directory, "database/")

    db_list = []
    with open(gold_path) as f:
        for line in f:
            line_split = line.strip().split('\t')
            if len(line_split) != 2:
                continue
            db = line.strip().split('\t')[1]
            if db not in db_list:
                db_list.append(db)

    kmaps = build_foreign_key_map_from_json(table_schema_path)

    # Get the training batches.
    log = Logger(os.path.join(params.logdir, params.logfile), "w")
    num_train_original = atis_data.num_utterances(data.train_data)
    log.put("Original number of training utterances:\t"
            + str(num_train_original))

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

    log.put(
        "Actual number of used training examples:\t" +
        str(num_train_examples))
    log.put("(Shortened by output limit of " +
            str(maximum_output_length) +
            ")")
    log.put("Number of steps per epoch:\t" + str(num_steps_per_epoch))
    log.put("Batch size:\t" + str(batch_size))

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
    patience = params.initial_patience
    learning_rate_coefficient = 1.
    previous_epoch_loss = float('inf')
    maximum_validation_accuracy = 0.

    countdown = int(patience)

    if params.scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model.trainer, mode='min', )

    keep_training = True
    while keep_training:
        log.put("Epoch:\t" + str(epochs))
        model.set_dropout(params.dropout_amount)

        if epochs > 0:
            # TODO: you need to reload valid_examples, since eval_fn may have revised them
            valid_examples = validsample_fn(data.valid_data,
                                            max_output_length=maximum_output_length)

        if not params.scheduler:
            model.set_learning_rate(learning_rate_coefficient * params.initial_learning_rate)

        # Run a training step.
        if params.interaction_level:
            epoch_loss = train_epoch_with_interactions(
                train_batches,
                params,
                model,
                randomize=not params.deterministic)
        else:
            epoch_loss = train_epoch_with_utterances(
                train_batches,
                model,
                randomize=not params.deterministic)

        log.put("train epoch loss:\t" + str(epoch_loss))

        model.set_dropout(0.)

        # # Run an evaluation step on a sample of the training data.
        train_eval_results = eval_fn(training_sample,
                                     model,
                                     params.train_maximum_sql_length,
                                     name=os.path.join(params.logdir, "train-eval"),
                                     write_results=True,
                                     gold_forcing=True,
                                     metrics=TRAIN_EVAL_METRICS)[0]

        for name, value in train_eval_results.items():
            log.put(
                "train final gold-passing " +
                name.name +
                ":\t" +
                "%.2f" %
                value)

        # Run an evaluation step on the validation set. - WITH GOLD FEED
        valid_eval_results = eval_fn(valid_examples,
                                     model,
                                     params.eval_maximum_sql_length,
                                     name=os.path.join(params.logdir, "valid-eval"),
                                     write_results=True,
                                     gold_forcing=True,
                                     metrics=VALID_EVAL_METRICS)[0]
        for name, value in valid_eval_results.items():
            log.put("valid gold-passing " + name.name + ":\t" + "%.2f" % value)

        valid_loss = valid_eval_results[Metrics.LOSS]
        # valid_token_accuracy = valid_eval_results[Metrics.TOKEN_ACCURACY]
        # string_accuracy = valid_eval_results[Metrics.STRING_ACCURACY]

        if params.scheduler:
            scheduler.step(valid_loss)

        if valid_loss > previous_epoch_loss:
            learning_rate_coefficient *= params.learning_rate_ratio
            log.put(
                "learning rate coefficient:\t" +
                str(learning_rate_coefficient))

        previous_epoch_loss = valid_loss
        saved = False

        # Run evaluation WITHOUT GOLD FEED
        actual_valid_eval_results = eval_fn(valid_examples,
                                            model,
                                            name=os.path.join(params.logdir, "valid_use_predicted_queries"),
                                            metrics=FINAL_EVAL_METRICS,
                                            total_num=atis_data.num_utterances(data.valid_data),
                                            database_username=params.database_username,
                                            database_password=params.database_password,
                                            database_timeout=params.database_timeout,
                                            use_predicted_queries=True,
                                            max_generation_length=params.eval_maximum_sql_length,
                                            write_results=True,
                                            use_gpu=True,
                                            compute_metrics=params.compute_metrics)[0]
        actual_token_accuracy = actual_valid_eval_results[Metrics.TOKEN_ACCURACY]
        actual_string_accuracy = actual_valid_eval_results[Metrics.STRING_ACCURACY]

        print("postprocess_eval...")

        database_schema = read_schema(table_schema_path)
        predictions = read_prediction(os.path.join(params.logdir, "valid_use_predicted_queries") + "_predictions.json")
        postprocess_db_sqls = postprocess(predictions, database_schema, True)

        postprocess_sqls = []
        for db in db_list:
            for postprocess_sql, interaction_id, turn_id in postprocess_db_sqls[db]:
                postprocess_sqls.append([postprocess_sql])

        actual_validation_accuracy = eval_script_evaluate(
            gold_path, postprocess_sqls, db_path, "match", kmaps, bool_verbal=False, bool_predict_file=False)['all']['exact']
        actual_validation_accuracy = float(actual_validation_accuracy) * 100  # percentage

        if not saved and actual_validation_accuracy > maximum_validation_accuracy:
            maximum_validation_accuracy = actual_validation_accuracy
            countdown = int(patience)
            last_save_file = os.path.join(params.logdir, "save_" + str(epochs))
            model.save(last_save_file)

            log.put("maximum validation accuracy:\t" + str(actual_validation_accuracy))
            log.put("actual token accuracy:\t" + str(actual_token_accuracy))
            log.put("actual string accuracy:\t" + str(actual_string_accuracy))
            log.put("patience:\t" + str(patience))
            log.put("save file:\t" + str(last_save_file))

        if countdown <= 0:
            keep_training = False

        countdown -= 1
        log.put("countdown:\t" + str(countdown))
        log.put("")

        epochs += 1

    log.put("Finished training!")
    log.close()

    return last_save_file


def evaluate(model, data, params, last_save_file, split, full_name=None):
    """Evaluates a pretrained model on a dataset.

    Inputs:
        model (ATISModel): Model class.
        data (ATISData): All of the data.
        params (namespace): Parameters for the model.
        last_save_file (str): Location where the model save file is.
    """

    if "data_clean" in params.raw_train_filename:
        raw_data_directory = "EditSQL/data_clean/"
    else:
        raw_data_directory = "EditSQL/data/"

    table_schema_path = os.path.join(raw_data_directory, "spider", "tables.json")
    gold_path = os.path.join(raw_data_directory, "spider", "dev_gold.sql")
    db_path = os.path.join(raw_data_directory, "database/")

    db_list = []
    with open(gold_path) as f:
        for line in f:
            line_split = line.strip().split('\t')
            if len(line_split) != 2:
                continue
            db = line.strip().split('\t')[1]
            if db not in db_list:
                db_list.append(db)

    kmaps = build_foreign_key_map_from_json(table_schema_path)

    if last_save_file:
        model.load(last_save_file)
    else:
        if not params.save_file:
            raise ValueError(
                "Must provide a save file name if not training first.")
        model.load(params.save_file)

    filename = split

    if filename == 'dev':
        split = data.dev_data
    elif filename == 'train':
        split = data.train_data
    elif filename == 'test':
        split = data.test_data
    elif filename == 'valid':
        split = data.valid_data
    else:
        raise ValueError("Split not recognized: " + str(params.evaluate_split))

    if params.use_predicted_queries:
        filename += "_use_predicted_queries"
    else:
        filename += "_use_gold_queries"

    if full_name is None:
        full_name = os.path.join(params.logdir, filename) + params.results_note

    if params.interaction_level or params.use_predicted_queries:
        examples = data.get_all_interactions(split)
        if params.interaction_level:
            evaluate_interaction_sample(
                examples,
                model,
                name=full_name,
                metrics=FINAL_EVAL_METRICS,
                total_num=atis_data.num_utterances(split),
                database_username=params.database_username,
                database_password=params.database_password,
                database_timeout=params.database_timeout,
                use_predicted_queries=params.use_predicted_queries,
                max_generation_length=params.eval_maximum_sql_length,
                write_results=True,
                use_gpu=True,
                compute_metrics=params.compute_metrics)
        else:
            evaluate_using_predicted_queries(
                examples,
                model,
                name=full_name,
                metrics=FINAL_EVAL_METRICS,
                total_num=atis_data.num_utterances(split),
                database_username=params.database_username,
                database_password=params.database_password,
                database_timeout=params.database_timeout)
    else:
        examples = data.get_all_utterances(split)
        evaluate_utterance_sample(
            examples,
            model,
            name=full_name,
            gold_forcing=False,
            metrics=FINAL_EVAL_METRICS,
            total_num=atis_data.num_utterances(split),
            max_generation_length=params.eval_maximum_sql_length,
            database_username=params.database_username,
            database_password=params.database_password,
            database_timeout=params.database_timeout,
            write_results=True)

    database_schema = read_schema(table_schema_path)
    predictions = read_prediction(full_name + "_predictions.json")
    postprocess_db_sqls = postprocess(predictions, database_schema, True) # TODO: add token/string acc?

    postprocess_sqls = []
    for db in db_list:
        for postprocess_sql, interaction_id, turn_id in postprocess_db_sqls[db]:
            postprocess_sqls.append([postprocess_sql])

    eval_scores = eval_script_evaluate(gold_path, postprocess_sqls, db_path, "match",
                                       kmaps, bool_verbal=False, bool_predict_file=False)

    print("\nall #={} acc={:3f}, easy #={} acc={:3f}, medium #={} acc={:3f}, "
          "hard #={} acc={:3f}, extra #={} acc={:3f}".format(
        eval_scores['all']['count'], eval_scores['all']['exact'],
        eval_scores['easy']['count'], eval_scores['easy']['exact'],
        eval_scores['medium']['count'], eval_scores['medium']['exact'],
        eval_scores['hard']['count'], eval_scores['hard']['exact'],
        eval_scores['extra']['count'], eval_scores['extra']['exact']
    ))


def main():
    """Main function that trains and/or evaluates a model."""
    params = interpret_args()

    # Prepare the dataset into the proper form.
    data = atis_data.ATISDataset(params)

    # Construct the model object.
    if params.interaction_level:
        model_type = SchemaInteractionATISModel
    else:
        print('not implemented')
        exit()

    model = model_type(
        params,
        data.input_vocabulary,
        data.output_vocabulary,
        data.output_vocabulary_schema,
        data.anonymizer if params.anonymize and params.anonymization_scoring else None)

    model = model.cuda()
    print('=====================Model Parameters=====================')
    for name, param in model.named_parameters():
        print(name, param.requires_grad, param.is_cuda, param.size())
        assert param.is_cuda

    model.build_optim()

    print('=====================Parameters in Optimizer==============')
    for param_group in model.trainer.param_groups:
        print(param_group.keys())
        for param in param_group['params']:
            print(param.size())

    if params.fine_tune_bert:
        print('=====================Parameters in BERT Optimizer==============')
        for param_group in model.bert_trainer.param_groups:
            print(param_group.keys())
            for param in param_group['params']:
                print(param.size())

    sys.stdout.flush()

    last_save_file = ""

    if params.train:
        last_save_file = train(model, data, params)
    if params.evaluate and 'valid' in params.evaluate_split:
        evaluate(model, data, params, last_save_file, split='valid')
    if params.evaluate and 'dev' in params.evaluate_split:
        evaluate(model, data, params, last_save_file, split='dev')
    if params.evaluate and 'test' in params.evaluate_split:
        evaluate(model, data, params, last_save_file, split='test')

if __name__ == "__main__":
    main()
