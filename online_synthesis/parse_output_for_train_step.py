import argparse
import json
from mcts_utils import load_tree

import sys
sys.path.append("../")
from utils.utils import load_raw_dataset
from dump_data.training_step_data import dump_data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dataset_path", type=str, default="")
    parser.add_argument("--tgt_dataset_path", type=str, default="")
    parser.add_argument("--tree_path", type=str, default="")
    parser.add_argument("--max_iterations", type=int, default=20)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    trees = load_raw_dataset(args.tree_path)
    record = {}
    for tree_sample in trees:
        tree_json = tree_sample["tree"]
        root, next_iter, max_iter, is_success = load_tree(tree_json)
        if not is_success:
            continue

        paths = []
        tree_record = json.loads(tree_json)
        for node_id, node_data in tree_record["nodes"].items():
            action = node_data["action"]
            tgt_instruction = node_data["instruction"]
            tgt_reward = node_data["local_avg_reward"]

            if node_data["parent"] is None:
                continue

            parent_node = tree_record["nodes"][node_data["parent"]]
            parent_instruction = parent_node["instruction"]
            parent_reward = parent_node["local_avg_reward"]
            
            if parent_reward < tgt_reward:
                paths.append(
                    (action, parent_instruction, parent_reward, tgt_instruction, tgt_reward)
                )

        
        record[tree_sample["meta"]] = {
            "tree": tree_json,
            "paths": paths,
            "tasks": tree_sample["tasks"],
            "is_success": is_success,
        }
    
    for meta in record:
        assert record[meta]["is_success"]
    
    dump_data(
        record,
        args.src_dataset_path,
        args.tgt_dataset_path,
    )

    print("Done.")