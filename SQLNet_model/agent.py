# Interactive SQLNet agent
from MISP_SQL.agent import Agent as BaseAgent


class Agent(BaseAgent):
    def __init__(self, world_model, error_detector, question_generator):
        BaseAgent.__init__(self, world_model, error_detector, question_generator)

    def evaluation(self, raw_data, pred_queries, query_gt, table_ids, engine):
        exe_tot_acc_num = 0.
        qm_one_acc_num = 0.
        qm_tot_acc_num = 0.

        one_err, tot_err = self.world_model.semparser.check_acc(raw_data, pred_queries, query_gt, (True, True, True))

        for idx, (sql_gt, sql_pred, tid) in enumerate(
                zip(query_gt, pred_queries, table_ids)):
            ret_gt = engine.execute(tid,
                                    sql_gt['sel'], sql_gt['agg'], sql_gt['conds'])
            try:
                ret_pred = engine.execute(tid,
                                          sql_pred['sel'], sql_pred['agg'], sql_pred['conds'])
            except:
                ret_pred = None
            exe_tot_acc_num += (ret_gt == ret_pred)

        qm_one_acc_num += (len(raw_data) - one_err) #ed - st
        qm_tot_acc_num += (len(raw_data) - tot_err)

        return qm_one_acc_num, qm_tot_acc_num, exe_tot_acc_num
