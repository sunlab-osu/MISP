# utils for user study
import random
import json

class bcolors:
    """
    Usage: print bcolors.WARNING + "Warning: No active frommets remain. Continue?" + bcolors.ENDC
    """
    PINK = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_main(sent):
    print(bcolors.PINK + bcolors.BOLD + sent + bcolors.ENDC)


def print_header(remaining_size, bool_table_color=False):
    task_notification = " Interactive Database Query "
    remain_notification = " Remaining: %d " % (remaining_size)
    print("=" * 50)
    print(bcolors.BOLD + task_notification + bcolors.ENDC)
    print(bcolors.BOLD + remain_notification + bcolors.ENDC)
    print(bcolors.BOLD + "\n Tip: Words referring to table headers/attributes are marked in " +
          bcolors.BLUE + "this color" + bcolors.ENDC + ".")
    if bool_table_color:
        print(bcolors.BOLD + " Tip: Words referring to table names are marked in " + bcolors.YELLOW + "this color" +
              bcolors.ENDC + ".")
    print("=" * 50)
    print("")
