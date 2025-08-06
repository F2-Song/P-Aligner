import sys
sys.path.append("../")
from load_data.training_data import load_data
from utils.utils import save_dataset

def process_one_path(
    one_path,
):
    # one_path = [(action, instruction, reward), (action, instruction, reward), ...]
    # the first action is the initial action root, and the last action is "self"
    assert one_path[0][0] == "root"
    assert one_path[-1][0] == "self"
    src_instruction = one_path[0][1]
    tgt_instruction = one_path[-1][1]
    tgt_reward = one_path[-1][2]
    action_seq = [t[0] for t in one_path[1:-1]]

    return src_instruction, tgt_instruction, tgt_reward, action_seq

def dump_data(
    record,
    src_dataset_path, 
    tgt_dataset_path,
):
    src_samples = load_data(
        src_dataset_path,
        to_parse=False,
    )
    
    src_samples_2 = load_data(
        src_dataset_path,
        to_parse=True,
    )

    metas = [sample["meta"] for sample in src_samples_2]
    
    training_samples = []
    for src_sample, meta in zip(src_samples, metas):
        assert meta in record
        paths = record[meta]["paths"]
        category = src_sample["category"]
        best_path = paths[0]
        worst_path = paths[-1]

        src_best_instruction, tgt_best_instruction, tgt_best_reward, best_actions = process_one_path(best_path)
        src_worst_instruction, tgt_worst_instruction, tgt_worst_reward, worst_actions = process_one_path(worst_path)
        assert src_best_instruction == src_worst_instruction
        new_sample = {
            "src_instruction": src_best_instruction,
            "chosen_instruction": tgt_best_instruction,
            "chosen_actions": best_actions,
            "rejected_instruction": tgt_worst_instruction,
            "rejected_actions": worst_actions,
            "chosen_reward": tgt_best_reward,
            "rejected_reward": tgt_worst_reward,
            "meta": meta,
            "category": category,
        }

        training_samples.append(new_sample)
    
    save_dataset(training_samples, tgt_dataset_path)