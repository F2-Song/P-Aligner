from utils import load_raw_dataset
import os
import json
import argparse

def get_prompt(
    src_instruction,
):    
    prompt = "You are an expert prompt engineer." + " "
    prompt += "Please help me optimize this prompt to get better response:\n\n[The Start of Raw Prompt]\n{}\n[The End of Raw Prompt]".format(src_instruction)
    
    return prompt

def get_actions(
    action_seq
):
    filtered_action_seq = []
    for action in action_seq:
        if action in filtered_action_seq:
            continue
        filtered_action_seq.append(action)
    action_seq = filtered_action_seq

    return action_seq

def get_response(
    tgt_instruction,
):  
    response = "The Optimized Prompt:\n\n[The Start of Optimized Prompt]\n{}\n[The End of Optimized Prompt]".format(tgt_instruction)
    
    return response
    
def prepare_dpo_training_data(
    samples,
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
        chosen_instruction = sample["chosen_instruction"]
        rejected_instruction = sample["rejected_instruction"]
        chosen_reward = sample["chosen_reward"]
        rejected_reward = sample["rejected_reward"]

        prompt = get_prompt(src_instruction)
        chosen = get_response(chosen_instruction)
        rejected = get_response(rejected_instruction)
        
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
                "content": rejected,
            },
            "chosen_reward": chosen_reward,
            "rejected_reward": rejected_reward,
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
        })
    
    return new_samples, dataset_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_path", type=str)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--sft_tgt_path", type=str)
    parser.add_argument("--dpo_tgt_path", type=str)
    parser.add_argument("--dataset_info_tgt_path", type=str)
    args = parser.parse_args()
    
    samples = load_raw_dataset(args.src_path)
    dpo_training_samples, dpo_dataset_info = prepare_dpo_training_data(samples)
    sft_training_samples, sft_dataset_info = prepare_sft_training_data_from_dpo(dpo_training_samples)

    with open(args.sft_tgt_path, 'w', encoding='utf-8') as outfile:
        json.dump(sft_training_samples, outfile, ensure_ascii=False, indent=4)

    with open(args.dpo_tgt_path, 'w', encoding='utf-8') as outfile:
        json.dump(dpo_training_samples, outfile, ensure_ascii=False, indent=4)

    os.makedirs(os.path.dirname(args.dataset_info_tgt_path), exist_ok=True)
    if os.path.exists(args.dataset_info_tgt_path):
        with open(args.dataset_info_tgt_path, 'r', encoding='utf-8') as f:
            original_dataset_info = json.load(f)
    else:
        original_dataset_info = {}

    sft_dataset_info["file_name"] = os.path.basename(args.sft_tgt_path)
    dpo_dataset_info["file_name"] = os.path.basename(args.dpo_tgt_path)
    sft_name = args.dataset_name + "_sft"
    dpo_name = args.dataset_name + "_dpo"
    original_dataset_info[sft_name] = sft_dataset_info
    original_dataset_info[dpo_name] = dpo_dataset_info
    
    with open(args.dataset_info_tgt_path, 'w', encoding='utf-8') as outfile:
        json.dump(original_dataset_info, outfile, ensure_ascii=False, indent=4)
