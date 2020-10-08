from MISP_SQL.agent import Agent as BaseAgent
from .sqlova.utils.utils_wikisql import *


class Agent(BaseAgent):
    def __init__(self, world_model, error_detector, question_generator, bool_mistake_exit,
                 bool_structure_question=False):
        BaseAgent.__init__(self, world_model, error_detector, question_generator,
                           bool_mistake_exit=bool_mistake_exit,
                           bool_structure_question=bool_structure_question)

    def evaluation(self, p_list, g_list, engine, tb, bool_verbal=False):
        pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi, pr_sql_i = p_list
        g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi, sql_i = g_list

        cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, \
        cnt_wc1_list, cnt_wo1_list, \
        cnt_wvi1_list, cnt_wv1_list = get_cnt_sw_list(g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi,
                                                      pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi,
                                                      sql_i, pr_sql_i,
                                                      mode='test')

        cnt_lx1_list = get_cnt_lx_list(cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list,
                                       cnt_wo1_list, cnt_wv1_list)
        lx_correct = sum(cnt_lx1_list) # lx stands for logical form accuracy

        cnt_x1_list, g_ans, pr_ans = get_cnt_x_list(engine, tb, g_sc, g_sa, sql_i, pr_sc, pr_sa, pr_sql_i)
        x_correct = sum(cnt_x1_list)

        cnt_list1 = [cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list, cnt_wo1_list, cnt_wv1_list, cnt_lx1_list,
                     cnt_x1_list]

        if bool_verbal:
            print("lf correct: {}, x correct: {}, cnt_list: {}".format(lx_correct, x_correct, cnt_list1))

        return cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list, cnt_wo1_list, cnt_wv1_list, cnt_wvi1_list, \
               cnt_lx1_list, cnt_x1_list, cnt_list1, g_ans, pr_ans