import os
print("pid: ", os.getpid())
from mcts_utils import load_rules
from pipeline import generate as generate_and_save
from config import args
import json
from load_data.training_data import load_data

def print_args_to_json(args):
    args_dict = vars(args)
    json_str = json.dumps(args_dict, indent=4)
    print(json_str)

if __name__ == "__main__":    
    print_args_to_json(args)

    # load the dataset
    dataset = load_data(
        args.dataset_path,
        to_parse=True
    )
    
    dataset = [sample for sample in dataset if (sample["line_index"] - args.rank) % args.rank_sum == 0]

    rules = load_rules(
        file_name=args.rule_name,
        rules_path="rules"
    )

    generate_and_save(
        dataset=dataset,
        rules=rules,
        args=args,
    )

    