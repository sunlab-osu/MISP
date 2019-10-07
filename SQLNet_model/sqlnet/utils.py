"""
Adapted by Ziyu Yao
"""

import json
from lib.dbengine import DBEngine
import re
import numpy as np
import pickle
import os
#from nltk.tokenize import StanfordTokenizer


def generate_sql_q1(sql_i1, raw_q1, raw_col1):
    """
        Adapted from SQLova_model script:
        https://github.com/naver/sqlova/blob/master/sqlova/utils/utils_wikisql.py#L1814

        sql = {'sel': 5, 'agg': 4, 'conds': [[3, 0, '59']]}
        agg_ops = ['', 'max', 'min', 'count', 'sum', 'avg']
        cond_ops = ['=', '>', '<', 'OP']

        Temporal as it can show only one-time conditioned case.
        sql_query: real sql_query
        sql_plus_query: More redable sql_query

        "PLUS" indicates, it deals with the some of db specific facts like PCODE <-> NAME
    """
    agg_ops = ['', 'max', 'min', 'count', 'sum', 'avg']
    cond_ops = ['=', '>', '<', 'OP']

    # headers = tb1["header"]
    headers = raw_col1
    # select_header = headers[sql['sel']].lower()
    # try:
    #     select_table = tb1["name"]
    # except:
    #     print(f"No table name while headers are {headers}")
    # select_table = tb1["id"]
    select_table = "GIVEN_TABLE" #TODO: where to get table id?

    select_agg = agg_ops[sql_i1['agg']]
    select_header = headers[sql_i1['sel']].encode('utf-8')
    sql_query_part1 = 'SELECT {}({}) '.format(select_agg, select_header)

    where_num = len(sql_i1['conds'])
    if where_num > 0:
        sql_query_part2 = 'WHERE'
        for i in range(where_num):
            # check 'OR'
            # number_of_sub_conds = len(sql['conds'][i])
            where_header_idx, where_op_idx, where_str = sql_i1['conds'][i]
            where_header = headers[where_header_idx].encode('utf-8')
            where_op = cond_ops[where_op_idx]
            if i > 0:
                sql_query_part2 += ' AND'
                # sql_plus_query_part2 += ' AND'

            sql_query_part2 += " {} {} {}".format(where_header, where_op, where_str.encode('utf-8'))
    else:
        sql_query_part2 = ''

    sql_query = sql_query_part1 + sql_query_part2
    # sql_plus_query = sql_plus_query_part1 + sql_plus_query_part2

    return sql_query

def load_data(sql_paths, table_paths, use_small=False):
    if not isinstance(sql_paths, list):
        sql_paths = (sql_paths, )
    if not isinstance(table_paths, list):
        table_paths = (table_paths, )
    sql_data = []
    table_data = {}

    max_col_num = 0
    for SQL_PATH in sql_paths:
        print "Loading data from %s"%SQL_PATH
        with open(SQL_PATH) as inf:
            for idx, line in enumerate(inf):
                if use_small and idx >= 1000:
                    break
                sql = json.loads(line.strip())
                sql_data.append(sql)

    for TABLE_PATH in table_paths:
        print "Loading data from %s"%TABLE_PATH
        with open(TABLE_PATH) as inf:
            for line in inf:
                tab = json.loads(line.strip())
                table_data[tab[u'id']] = tab

    for sql in sql_data:
        assert sql[u'table_id'] in table_data

    return sql_data, table_data

def load_dataset(dataset_id, use_small=False, data_dir=''):
    if dataset_id == 0:
        print "Loading from original dataset"
        sql_data, table_data = load_data(data_dir + 'data/train_tok.jsonl',
                data_dir + 'data/train_tok.tables.jsonl', use_small=use_small)
        val_sql_data, val_table_data = load_data(data_dir + 'data/dev_tok.jsonl',
                data_dir + 'data/dev_tok.tables.jsonl', use_small=use_small)
        test_sql_data, test_table_data = load_data(data_dir + 'data/test_tok.jsonl',
                data_dir + 'data/test_tok.tables.jsonl', use_small=use_small)
        TRAIN_DB = data_dir + 'data/train.db'
        DEV_DB = data_dir + 'data/dev.db'
        TEST_DB = data_dir + 'data/test.db'
    else:
        print "Loading from re-split dataset"
        sql_data, table_data = load_data(data_dir + 'data_resplit/train.jsonl',
                data_dir + 'data_resplit/tables.jsonl', use_small=use_small)
        val_sql_data, val_table_data = load_data(data_dir + 'data_resplit/dev.jsonl',
                data_dir + 'data_resplit/tables.jsonl', use_small=use_small)
        test_sql_data, test_table_data = load_data(data_dir + 'data_resplit/test.jsonl',
                data_dir + 'data_resplit/tables.jsonl', use_small=use_small)
        TRAIN_DB = data_dir + 'data_resplit/table.db'
        DEV_DB = data_dir + 'data_resplit/table.db'
        TEST_DB = data_dir + 'data_resplit/table.db'

    return sql_data, table_data, val_sql_data, val_table_data,\
            test_sql_data, test_table_data, TRAIN_DB, DEV_DB, TEST_DB

def best_model_name(args, for_load=False, data_dir='', prefix=''):
    new_data = 'new' if args.dataset > 0 else 'old'
    mode = 'seq2sql' if args.baseline else 'sqlnet'
    if for_load:
        use_emb = use_rl = ''
    else:
        use_emb = '_train_emb' if args.train_emb else ''
        use_rl = 'rl_' if args.rl else ''
    use_ca = '_ca' if args.ca else ''

    agg_model_name = data_dir + 'saved_model/%s%s_%s%s%s.agg_model'%(prefix, new_data,
            mode, use_emb, use_ca)
    sel_model_name = data_dir + 'saved_model/%s%s_%s%s%s.sel_model'%(prefix, new_data,
            mode, use_emb, use_ca)
    cond_model_name = data_dir + 'saved_model/%s%s_%s%s%s.cond_%smodel'%(prefix, new_data,
            mode, use_emb, use_ca, use_rl)

    if not for_load and args.train_emb:
        agg_embed_name = data_dir + 'saved_model/%s_%s%s%s.agg_embed'%(new_data,
                mode, use_emb, use_ca)
        sel_embed_name = data_dir + 'saved_model/%s_%s%s%s.sel_embed'%(new_data,
                mode, use_emb, use_ca)
        cond_embed_name = data_dir + 'saved_model/%s_%s%s%s.cond_embed'%(new_data,
                mode, use_emb, use_ca)

        return agg_model_name, sel_model_name, cond_model_name,\
                agg_embed_name, sel_embed_name, cond_embed_name
    else:
        return agg_model_name, sel_model_name, cond_model_name


def to_batch_seq(sql_data, table_data, idxes, st, ed, ret_vis_data=False):
    q_seq = []
    col_seq = []
    col_num = []
    ans_seq = []
    query_seq = []
    gt_cond_seq = []
    vis_seq = []
    for i in range(st, ed):
        sql = sql_data[idxes[i]]
        q_seq.append(sql['question_tok'])
        col_seq.append(table_data[sql['table_id']]['header_tok'])
        col_num.append(len(table_data[sql['table_id']]['header']))
        ans_seq.append((sql['sql']['agg'],
            sql['sql']['sel'], 
            len(sql['sql']['conds']),
            tuple(x[0] for x in sql['sql']['conds']),
            tuple(x[1] for x in sql['sql']['conds'])))
        query_seq.append(sql['query_tok'])
        gt_cond_seq.append(sql['sql']['conds'])
        vis_seq.append((sql['question'],
            table_data[sql['table_id']]['header'], sql['query']))
    if ret_vis_data:
        return q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, vis_seq
    else:
        return q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq

def to_batch_query(sql_data, idxes, st, ed):
    query_gt = []
    table_ids = []
    for i in range(st, ed):
        query_gt.append(sql_data[idxes[i]]['sql'])
        table_ids.append(sql_data[idxes[i]]['table_id'])
    return query_gt, table_ids

def epoch_train(model, optimizer, batch_size, sql_data, table_data, pred_entry):
    model.train()
    perm=np.random.permutation(len(sql_data))
    cum_loss = 0.0

    cum_loss_sel, cum_loss_agg, cum_loss_cond = 0., 0., 0.
    st = 0
    while st < len(sql_data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)

        q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq = \
                to_batch_seq(sql_data, table_data, perm, st, ed)
        gt_where_seq = model.generate_gt_where_seq(q_seq, col_seq, query_seq)
        gt_sel_seq = [x[1] for x in ans_seq]
        score = model.forward(q_seq, col_seq, col_num, pred_entry,
                gt_where=gt_where_seq, gt_cond=gt_cond_seq, gt_sel=gt_sel_seq)
        loss_output = model.loss(score, ans_seq, pred_entry, gt_where_seq)

        if model.temperature:
            [loss, loss_sel, loss_agg, loss_cond] = loss_output
            cum_loss_sel += loss_sel.data.cpu().numpy()[0] * (ed - st)
            cum_loss_agg += loss_agg.data.cpu().numpy()[0] * (ed - st)
            cum_loss_cond += loss_cond.data.cpu().numpy()[0] * (ed - st)
        else:
            [loss] = loss_output

        cum_loss += loss.data.cpu().numpy()[0]*(ed - st)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        st = ed

    if model.temperature:
        return [cum_loss / len(sql_data), cum_loss_sel/len(sql_data), cum_loss_agg/len(sql_data), cum_loss_cond/len(sql_data)]

    return [cum_loss / len(sql_data)]

def epoch_exec_acc(model, batch_size, sql_data, table_data, db_path):
    engine = DBEngine(db_path)

    model.eval()
    perm = list(range(len(sql_data)))
    tot_acc_num = 0.0
    acc_of_log = 0.0
    st = 0
    while st < len(sql_data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)
        q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, raw_data = \
            to_batch_seq(sql_data, table_data, perm, st, ed, ret_vis_data=True)
        raw_q_seq = [x[0] for x in raw_data]
        raw_col_seq = [x[1] for x in raw_data]
        gt_where_seq = model.generate_gt_where_seq(q_seq, col_seq, query_seq)
        query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)
        gt_sel_seq = [x[1] for x in ans_seq]
        score = model.forward(q_seq, col_seq, col_num,
                (True, True, True), gt_sel=gt_sel_seq)
        pred_queries = model.gen_query(score, q_seq, col_seq,
                raw_q_seq, raw_col_seq, (True, True, True))

        for idx, (sql_gt, sql_pred, tid) in enumerate(
                zip(query_gt, pred_queries, table_ids)):
            ret_gt = engine.execute(tid,
                    sql_gt['sel'], sql_gt['agg'], sql_gt['conds'])
            try:
                ret_pred = engine.execute(tid,
                        sql_pred['sel'], sql_pred['agg'], sql_pred['conds'])
            except:
                ret_pred = None
            tot_acc_num += (ret_gt == ret_pred)
        
        st = ed

    return tot_acc_num / len(sql_data)

def epoch_acc(model, batch_size, sql_data, table_data, pred_entry):
    model.eval()
    perm = list(range(len(sql_data)))
    st = 0
    one_acc_num = 0.0
    tot_acc_num = 0.0
    while st < len(sql_data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)

        q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, raw_data = to_batch_seq(sql_data, table_data, perm, st, ed, ret_vis_data=True)
        raw_q_seq = [x[0] for x in raw_data]
        raw_col_seq = [x[1] for x in raw_data]
        query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)
        gt_sel_seq = [x[1] for x in ans_seq]
        score = model.forward(q_seq, col_seq, col_num,
                pred_entry, gt_sel = gt_sel_seq)
        pred_queries = model.gen_query(score, q_seq, col_seq,
                raw_q_seq, raw_col_seq, pred_entry)
        one_err, tot_err = model.check_acc(raw_data,
                pred_queries, query_gt, pred_entry)

        one_acc_num += (ed-st-one_err)
        tot_acc_num += (ed-st-tot_err)

        st = ed
    return tot_acc_num / len(sql_data), one_acc_num / len(sql_data)

def epoch_reinforce_train(model, optimizer, batch_size, sql_data, table_data, db_path):
    engine = DBEngine(db_path)

    model.train()
    perm = np.random.permutation(len(sql_data))
    cum_reward = 0.0
    st = 0
    while st < len(sql_data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)

        q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, raw_data =\
            to_batch_seq(sql_data, table_data, perm, st, ed, ret_vis_data=True)
        gt_where_seq = model.generate_gt_where_seq(q_seq, col_seq, query_seq)
        raw_q_seq = [x[0] for x in raw_data]
        raw_col_seq = [x[1] for x in raw_data]
        query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)
        gt_sel_seq = [x[1] for x in ans_seq]
        score = model.forward(q_seq, col_seq, col_num, (True, True, True),
                reinforce=True, gt_sel=gt_sel_seq)
        pred_queries = model.gen_query(score, q_seq, col_seq, raw_q_seq,
                raw_col_seq, (True, True, True), reinforce=True)

        query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)
        rewards = []
        for idx, (sql_gt, sql_pred, tid) in enumerate(
                zip(query_gt, pred_queries, table_ids)):
            ret_gt = engine.execute(tid,
                    sql_gt['sel'], sql_gt['agg'], sql_gt['conds'])
            try:
                ret_pred = engine.execute(tid,
                        sql_pred['sel'], sql_pred['agg'], sql_pred['conds'])
            except:
                ret_pred = None

            if ret_pred is None:
                rewards.append(-2)
            elif ret_pred != ret_gt:
                rewards.append(-1)
            else:
                rewards.append(1)

        cum_reward += (sum(rewards))
        optimizer.zero_grad()
        model.reinforce_backward(score, rewards)
        optimizer.step()

        st = ed

    return cum_reward / len(sql_data)


def load_word_emb(file_name, load_used=False, use_small=False, data_dir=''):
    if not load_used:
        pkl_filename = file_name[:-3] + ("small.pkl" if use_small else "pkl")
        if os.path.exists(data_dir + pkl_filename):
            print ('Loading word embedding from %s' % (data_dir + pkl_filename))
            ret = pickle.load(open(data_dir + pkl_filename))
        else:
            print ('Loading word embedding from %s' % (data_dir + file_name))
            ret = {}
            with open(data_dir + file_name) as inf:
                for idx, line in enumerate(inf):
                    if (use_small and idx >= 5000):
                        break
                    info = line.strip().split(' ')
                    if info[0].lower() not in ret:
                        ret[info[0]] = np.array(map(lambda x: float(x), info[1:]))
            print "Save word_emb to %s" % pkl_filename
            pickle.dump(ret, open(data_dir + pkl_filename, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

        return ret
    else:
        print ('Load used word embedding')
        with open(data_dir + 'glove/word2idx.json') as inf:
            w2i = json.load(inf)
        with open(data_dir + 'glove/usedwordemb.npy') as inf:
            word_emb_val = np.load(inf)
        return w2i, word_emb_val
