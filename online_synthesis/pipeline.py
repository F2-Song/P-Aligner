import multiprocessing
from tqdm import tqdm
import os
from support import create_lm_optimizer
from functools import partial
import time
from mcts_utils import (
    MCTSNode,
    save_tree, 
    load_tree, 
    mcts, 
    check_completed,
)

import sys
sys.path.append("../")
from utils.utils import load_raw_dataset, save_dataset

def check(
    tree_sample,
    args
):
    # Check if the tree is completed
    tree_json = tree_sample["tree"]
    root, next_iter, max_iter, is_success = load_tree(tree_json)
    state = check_completed(
        root, 
        start_iter_index=next_iter,
        iterations=args.iterations,
    )

    return state

def search(
    tree_sample,
    rules,
    args
):
    if os.path.exists(
        args.flag_path
    ):
        lm_optimize_instruction = create_lm_optimizer(
            optimizer_name=args.lm_optimizer_type, # local | openai
            model_path=args.model_path,
        )

        tree_json = tree_sample["tree"]
        task_labels = tree_sample["tasks"]
        local_rules = [rules[label] for label in task_labels]
        local_rules = sum(local_rules, [])
        temp = {}
        for rule in local_rules:
            temp[rule["name"]] = rule["content"]
        local_rules = temp

        while True:
            root, next_iter, max_iter, is_success = load_tree(tree_json)
            state, tree_json = mcts(
                root, 
                local_rules,
                lm_optimize_instruction,
                args=args,
                start_iter_index=next_iter,
                iterations=args.iterations,
            )
            if state == "success":
                break
            else:
                time.sleep(1)
        tree_sample["tree"] = tree_json
        save_dataset([tree_sample], args.output_path, flag="a")

        return state
    else:
        print("Interrupted")
        exit()

def generate(
    dataset,
    rules,
    args,
):
    # load the last output
    record = {}
    try:
        trees = load_raw_dataset(args.last_output_path, is_lines=True)
    except:
        trees = []
    
    for tree_sample in trees:
        try:
            record[tree_sample["meta"]] = tree_sample
        except:
            pass
    
    trees = []
    for sample in dataset:
        if sample["meta"] in record:
            potential_tree_sample = record[sample["meta"]]
            if check(potential_tree_sample, args):
                continue
            else:
                trees.append(potential_tree_sample)
        else:
            root = MCTSNode(
                sample["instruction"], 
                sample["instruction"],
                index=0,
            )
            tree_json = save_tree(
                root, 
                0, 
                args.iterations, 
                is_success=False
            )
            trees.append({
                "tree": tree_json,
                "meta": sample["meta"],
                "tasks": sample["tasks"],
            })

    search_func = partial(
        search,
        rules=rules, 
        args=args
    )
    
    with multiprocessing.Pool(processes=args.num_threads) as pool:
        pbar = tqdm(total=len(trees), ncols=50)
        for re in pool.imap_unordered(search_func, trees):
            if re == "success":
                pbar.update(1)
    
    print("All tasks are completed.")

