from utils import load_raw_dataset, save_dataset
import os
import json
import argparse

def get_prompt(
    src_instruction,
    rules,
    action,
):
    prompt = "You are an expert prompt engineer." + " "
    prompt += "Please help me optimize this prompt to get better response:\n\n[The Start of Raw Prompt]\n{}\n[The End of Raw Prompt]".format(src_instruction)
    prompt += "\n\nYou should optimize this prompt by {}".format(rules[action])

    return prompt

def get_response(
    tgt_instruction,
):    
    response = "The Optimized Prompt:\n\n[The Start of Optimized Prompt]\n{}\n[The End of Optimized Prompt]".format(tgt_instruction)
    
    return response
    
def prepare_dpo_training_data(
    samples,
    rules,
):  
    dataset_info = {
        "file_name": "[placeholder]",
        "formatting": "sharegpt",
        "ranking": True,
        "columns": {
            "messages": "messages",
            "chosen": "chosen",
            "rejected": "rejected",
        },
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system",
        }
    }
    new_samples = []
    
    for sample in samples:
        src_instruction = sample["src_instruction"]
        src_reward = sample["src_reward"]
        chosen_instruction = sample["chosen_instruction"]
        chosen_reward = sample["chosen_reward"]
        action = sample["action"]
        category = sample["category"]

        prompt = get_prompt(src_instruction, rules, action)
        chosen = get_response(chosen_instruction)
        
        new_samples.append({
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "chosen": {
                "role": "assistant",
                "content": chosen,
            },
            "rejected": {
                "role": "assistant",
                "content": None,
            },
            "chosen_reward": chosen_reward,
            "src_reward": src_reward,
            "category": category,
            "meta": sample["meta"] if "meta" in sample else sample["source"],
        })
    
    return new_samples, dataset_info

def prepare_sft_training_data_from_dpo(
    samples,
):
    new_samples = []
    dataset_info = {
        "file_name": "[placeholder]",
        "formatting": "sharegpt",
        "columns": {
            "messages": "messages",
        },
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system",
        }
    }
    
    for sample in samples:
        context = sample["messages"]
        chosen = sample["chosen"]["content"]
        context = context + [
            {
                "role": "assistant",
                "content": chosen,
            }
        ]

        new_samples.append({
            "messages": context,
            "meta": sample["meta"] if "meta" in sample else sample["source"],
            "src_reward": sample["src_reward"],
            "tgt_reward": sample["chosen_reward"],
            "category": sample["category"],
        })
    
    return new_samples, dataset_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_path", type=str)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--sft_tgt_path", type=str)
    parser.add_argument("--dataset_info_tgt_path", type=str)
    parser.add_argument("--rule_path", type=str, default="online_synthesis/rules/rule_P-Aligner.json")
    args = parser.parse_args()
    
    rules = load_raw_dataset(args.rule_path, is_lines=False)
    new_rules = {}
    for category in rules:
        for rule in rules[category]:
            new_rules[rule["name"]] = rule["content"]
    rules = new_rules

    samples = load_raw_dataset(args.src_path)
    dpo_training_samples, dpo_dataset_info = prepare_dpo_training_data(samples, rules)
    sft_training_samples, sft_dataset_info = prepare_sft_training_data_from_dpo(dpo_training_samples)

    with open(args.sft_tgt_path, 'w', encoding='utf-8') as outfile:
        json.dump(sft_training_samples, outfile, ensure_ascii=False, indent=4)

    os.makedirs(os.path.dirname(args.dataset_info_tgt_path), exist_ok=True)
    if os.path.exists(args.dataset_info_tgt_path):
        with open(args.dataset_info_tgt_path, 'r', encoding='utf-8') as f:
            original_dataset_info = json.load(f)
    else:
        original_dataset_info = {}

    sft_dataset_info["file_name"] = os.path.basename(args.sft_tgt_path)
    sft_name = args.dataset_name + "_sft"
    original_dataset_info[sft_name] = sft_dataset_info
    
    with open(args.dataset_info_tgt_path, 'w', encoding='utf-8') as outfile:
        json.dump(original_dataset_info, outfile, ensure_ascii=False, indent=4)
