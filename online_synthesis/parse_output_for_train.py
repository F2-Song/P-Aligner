import argparse
import json
from mcts_utils import load_tree

import sys
sys.path.append("../")
from utils.utils import load_raw_dataset
from dump_data.training_data import dump_data

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
            path = [
                ("self", node_data["instruction"], node_data["local_avg_reward"]) # (action, instruction, reward)
            ]
            current_node = node_data
            while True:
                path.append((current_node["action"], current_node["instruction"], current_node["local_avg_reward"]))
                if current_node["parent"] is None:
                    break
                current_node = tree_record["nodes"][current_node["parent"]]
            path = path[::-1]
            paths.append(path)
        
        # sort the paths by the reward of the last node by descend, and ensure the last node action is "self"
        paths = sorted(paths, key=lambda x: x[-1][2], reverse=True)
        for path in paths:
            assert path[-1][0] == "self"
        
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