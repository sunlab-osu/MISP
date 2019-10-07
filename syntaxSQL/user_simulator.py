# user simulator

from collections import defaultdict

# from prettytable import PrettyTable
from interaction_framework.user_simulator import UserSim, RealUser
from user_study_utils import bcolors


class RealUsersyntaxSQL(RealUser):
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

        table2columns = defaultdict(list)
        for tab_id, column in table['column_names_original']:
            if tab_id >= 0:
                table2columns[tab_id].append(column)

        for tab_id in range(len(table2columns)):
            print(bcolors.BOLD + "Table %d " % (tab_id + 1) + bcolors.YELLOW + table['table_names_original'][tab_id] + bcolors.ENDC)
            print(bcolors.BLUE + bcolors.BOLD + "{}\n".format(table2columns[tab_id]) + bcolors.ENDC)

