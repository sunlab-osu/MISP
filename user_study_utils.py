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


def case_sampling_SQLNet(K=100):
    from SQLNet_model.sqlnet.utils import load_data
    data_dir = "SQLNet_model/data/"
    sql_data, table_data = load_data(data_dir + "test_tok.jsonl", data_dir + "test_tok.tables.jsonl")
    size = len(sql_data)
    print(size)
    sampled_ids = []
    while len(sampled_ids) < K:
        id = random.choice(range(size))
        if id in sampled_ids:
            continue

        question = sql_data[id]['question']
        table_id = sql_data[id]['table_id']
        headers = table_data[table_id]['header']

        try:
            print("question: {}\nheaders: {}".format(question, headers))
            action = raw_input("Take or not?")
            if action == 'y':
                sampled_ids.append(id)
                json.dump(sampled_ids, open(data_dir + "user_study_ids.json", "w"))
        except:
            pass
