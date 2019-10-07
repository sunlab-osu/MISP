# user simulator

# from prettytable import PrettyTable
from interaction_framework.user_simulator import UserSim, RealUser
from user_study_utils import bcolors


class RealUserSQLNet(RealUser):
    def __init__(self, err_evaluator, tables):
        RealUser.__init__(self, err_evaluator)

        self.tables = tables

    def show_table(self, table_id):
        table = self.tables[table_id]

        # # basic information
        # for key, key_alias in zip(["page_title", "section_title", "caption"],
        #                           ["Page Title", "Section Title", "Table Caption"]):
        #     if key in table:
        #         print("{}: {}".format(key_alias, table[key]))

        # print("\n")
        # x = PrettyTable()
        # x.field_names = table['header']
        # for row in table['rows']:
        #     x.add_row(row)
        # print(x)

        print(bcolors.BLUE + bcolors.BOLD + "{}".format(table['header']) + bcolors.ENDC)


