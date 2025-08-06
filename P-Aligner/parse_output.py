import argparse

import sys
sys.path.append("../")
from utils.utils import load_raw_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dataset_path", type=str, default="")
    parser.add_argument("--tgt_dataset_path", type=str, default="")
    parser.add_argument("--medium_path", type=str, default="")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    medium = load_raw_dataset(args.medium_path)
    record = {}
    for medium_sample in medium:
        record[medium_sample["meta"]] = {
            "best_instruction": medium_sample["better_instruction"],
        }
    
    if "arena_hard" in args.src_dataset_path:
        from dump_data.arena_hard import dump_data
    elif "bpo_test" in args.src_dataset_path:
        from dump_data.bpo_test import dump_data
    elif "dolly_eval" in args.src_dataset_path:
        from dump_data.dolly_eval import dump_data
    elif "self_instruct_eval" in args.src_dataset_path:
        from dump_data.self_instruct_eval import dump_data
    elif "vicuna_eval" in args.src_dataset_path:
        from dump_data.vicuna_eval import dump_data
    else:
        raise NotImplementedError
    
    dump_data(
        record,
        args.src_dataset_path,
        args.tgt_dataset_path
    )

    print("Done.")