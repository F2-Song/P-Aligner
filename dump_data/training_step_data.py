import sys
sys.path.append("../")
from load_data.training_data import load_data
from utils.utils import save_dataset

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

        for path_index, path in enumerate(paths):
            action = path[0]
            src_instruction = path[1] 
            src_reward = path[2]
            tgt_instruction = path[3]
            tgt_reward = path[4]

            new_sample = {
                "src_instruction": src_instruction,
                "src_reward": src_reward,
                "chosen_instruction": tgt_instruction,
                "chosen_reward": tgt_reward,
                "action": action,
                "meta": meta + "_{}".format(path_index),
                "category": category,
            }

            training_samples.append(new_sample)
    
    save_dataset(training_samples, tgt_dataset_path)