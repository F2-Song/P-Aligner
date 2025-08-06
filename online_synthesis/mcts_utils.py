import os
import json
import uuid
import numpy as np
import random
import time
import sys
sys.path.append("../")
from online_mcts.support import generate_responses, score_responses
from utils.utils import load_raw_dataset

class MCTSNode:
    def __init__(
            self, 
            instruction, 
            raw_instruction, 
            parent=None,
            index=0,
        ):
        self.id = str(uuid.uuid4())
        self.index = index
        self.instruction = instruction
        self.raw_instruction = raw_instruction
        self.parent = parent
        self.children = []
        self.visits = 0
        self.reward = 0
        self.local_avg_reward = None
        self.local_responses = None
        self.local_rewards = None
        self.action = "root"

    def is_fully_expanded(self, rules):
        return len(self.children) == len(rules)

    def chosen_child(self, args):
        exploration_weight = args.exploration_weight
        def compute_Q_fn(node): 
            reward_list = [node.local_avg_reward]
            if len(node.children) > 0:
                for child in node.children:
                    reward_list.append(
                        compute_Q_fn(child)
                    )
            return np.mean(reward_list)

        weights = [
            compute_Q_fn(child) + exploration_weight * np.sqrt(np.log(self.visits) / child.visits)
            for child in self.children
        ]
        
        return self.children[np.argmax(weights)]
    
    def best_child(self):
        return max(self.children, key=lambda child: child.reward)

    def expand(self, rules, lm_optimize_instruction, next_node_index):
        unused_rules = [rule for rule in rules if rule not in [child.action for child in self.children]]
        action = random.choice(unused_rules)
        new_instruction = lm_optimize_instruction(self.instruction, action, rules[action])
        if new_instruction:
            new_node = MCTSNode(new_instruction, self.raw_instruction, parent=self, index=next_node_index)
            new_node.action = action
            self.children.append(new_node)
            return new_node
        else:
            return None

    def update(self, reward):
        self.visits += 1
        if self.children:
            if self.local_avg_reward is None:
                self.reward = max([child.reward for child in self.children])
            else:
                self.reward = max([child.reward for child in self.children] + [self.local_avg_reward])
        else:
            self.reward = reward

def save_tree(
    root,
    next_iter,
    max_iter,
    is_success=True
):
    # save the tree to a json str
    record = {
        "root_id": root.id,
        "nodes": {},
        "next_iter": next_iter,
        "max_iter": max_iter,
        "is_success": is_success
    }

    def traverse(node):
        record["nodes"][node.id] = {
            "instruction": node.instruction,
            "raw_instruction": node.raw_instruction,
            "visits": node.visits,
            "local_avg_reward": node.local_avg_reward,
            "local_responses": node.local_responses,
            "local_rewards": node.local_rewards,
            "reward": node.reward,
            "action": node.action,
            "children": [child.id for child in node.children],
            "parent": node.parent.id if node.parent else None,
            "index": node.index,
        }

        for child in node.children:
            traverse(child)
    
    traverse(root)
    return json.dumps(record, ensure_ascii=False)

def load_tree(record_str):
    # load the tree from a json str
    record = json.loads(record_str)
    root_id, next_iter, max_iter, is_success = record["root_id"], record["next_iter"], record["max_iter"], record["is_success"]

    nodes = {}
    for node_id, node_data in record["nodes"].items():
        node = MCTSNode(
            node_data["instruction"], 
            node_data["raw_instruction"],
            index=node_data["index"],
        )
        node.visits = node_data["visits"]
        node.reward = node_data["reward"]
        node.local_avg_reward = node_data["local_avg_reward"]
        node.local_responses = node_data["local_responses"]
        node.local_rewards = node_data["local_rewards"]
        node.action = node_data["action"]
        nodes[node_id] = node
    
    for node_id, node_data in record["nodes"].items():
        node = nodes[node_id]
        if node_data["parent"]:
            node.parent = nodes[node_data["parent"]]
        node.children = [nodes[child_id] for child_id in node_data["children"]]
    
    return nodes[root_id], next_iter, max_iter, is_success

def load_rules(
    file_name,
    rules_path="rules",
):
    return load_raw_dataset(os.path.join(rules_path, f"{file_name}.json"), is_lines=False)

def simulate(node, args):
    context = [
        {"role": "user", "content": node.instruction},
    ]
    raw_context = [
        {"role": "user", "content": node.raw_instruction},
    ]

    generate_failed_attempts = 0
    while generate_failed_attempts < 10:
        responses = generate_responses(
            context=context,
            port=args.port,
        )
        if responses is not None:
            break
        generate_failed_attempts += 1
        time.sleep(1)
    if responses is None:
        return None
    
    score_failed_attempts = 0
    while score_failed_attempts < 10:
        score_result = score_responses(
            raw_context=raw_context,
            responses=responses,
            port=args.port,
        )
        if score_result is not None:
            break
        score_failed_attempts += 1
        time.sleep(1)
    if score_result is None:
        return None
    
    assert score_result is not None
    avg_score, scores = score_result

    return (float(avg_score), responses, scores)

def get_node_num(root_node):
    def traverse(node):
        num = 1
        for child in node.children:
            num += traverse(child)
        return num
    return traverse(root_node)

def check_completed(
    root,
    start_iter_index=0, 
    iterations=10,
):
    is_completed = False
    if start_iter_index >= iterations:
         if root.local_rewards is not None:
            is_completed = True
    
    return is_completed

def mcts(
    root, 
    rules,
    lm_optimize_instruction,
    args,
    start_iter_index=0, 
    iterations=10,
):
    for iter_index in range(iterations):
        if iter_index < start_iter_index:
            continue

        current_node = root

        # Selection
        while current_node.is_fully_expanded(rules):
            current_node = current_node.chosen_child(args)

        # Expansion
        assert not current_node.is_fully_expanded(rules)
        if not current_node.is_fully_expanded(rules):
            next_node_index = get_node_num(
                root_node=root
            )
            current_node = current_node.expand(
                rules, 
                lm_optimize_instruction,
                next_node_index=next_node_index,
            )
            if current_node is None:
                # delete the expanded node
                tree_json = save_tree(
                    root = root,
                    next_iter = iter_index,
                    max_iter = iterations,
                    is_success = False,
                )
                return "failed", tree_json

        # Simulation
        assert len(current_node.children) == 0 # current_node is a leaf node
        result = simulate(current_node, args)
        if result is None:
            # delete the expanded node
            current_node.parent.children.remove(current_node)
            del current_node
            
            # save the tree
            tree_json = save_tree(
                root = root,
                next_iter = iter_index,
                max_iter = iterations,
                is_success = False,
            )
            return "failed", tree_json
        else:
            current_node.local_avg_reward, current_node.local_responses, current_node.local_rewards = result
            reward = current_node.local_avg_reward

            # Backpropagation
            while current_node is not None:
                current_node.update(reward)
                current_node = current_node.parent
    
    if root.local_rewards is None:
        assert root.local_responses is None
        result = simulate(root, args)
        if result is None:
            tree_json = save_tree(
                root = root,
                next_iter = iterations if start_iter_index < iterations else start_iter_index,
                max_iter = iterations,
                is_success = False,
            )
            return "failed", tree_json
        else:
            root.local_avg_reward, root.local_responses, root.local_rewards = result

    tree_json = save_tree(
        root = root,
        next_iter = iterations if start_iter_index < iterations else start_iter_index, # if start_iter_index < iterations, it will be updated to iterations; otherwise, it will do nothing
        max_iter = iterations,
        is_success = True,
    )
    return "success", tree_json