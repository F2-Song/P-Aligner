import sys
sys.path.append("../")
from utils.utils import load_raw_dataset

def load_data(
    path,
    pre_context = [],
    to_parse = True,
):
    dataset = load_raw_dataset(path, is_lines=True, has_line_index=True)
    if to_parse:
        new_dataset = []
        for sample in dataset:
            task_label = sample["task_category"]
            if task_label not in ["Math", "Coding & Debugging", "Honesty", "Harmlessness"]:
                task_labels = ["Helpfulness"]
            else:
                task_labels = [task_label, "Helpfulness"]
            
            new_sample = {
                "instruction": sample["turns"][0]["content"],
                "line_index": sample["line_index"],
                "meta": "arena_hard:{}".format(sample["line_index"]),
                "tasks": task_labels,
            }
            new_dataset.append(new_sample)

        return new_dataset
    else:
        return dataset