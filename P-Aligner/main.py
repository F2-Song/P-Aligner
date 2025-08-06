from pipeline import generate as generate_and_save
from config import args
import json

def print_args_to_json(args):
    args_dict = vars(args)
    json_str = json.dumps(args_dict, indent=4)
    print(json_str)

if __name__ == "__main__":    
    # load the dataset
    if "arena_hard" in args.dataset_path:
        from load_data.arena_hard import load_data
        dataset = load_data(
            args.dataset_path,
            to_parse=True
        )
    elif "bpo_test" in args.dataset_path:
        from load_data.bpo_test import load_data
        dataset = load_data(
            args.dataset_path,
            to_parse=True
        )
    elif "dolly_eval" in args.dataset_path:
        from load_data.dolly_eval import load_data
        dataset = load_data(
            args.dataset_path,
            to_parse=True
        )
    elif "self_instruct_eval" in args.dataset_path:
        from load_data.self_instruct_eval import load_data
        dataset = load_data(
            args.dataset_path,
            to_parse=True
        )
    elif "vicuna_eval" in args.dataset_path:
        from load_data.vicuna_eval import load_data
        dataset = load_data(
            args.dataset_path,
            to_parse=True
        )
    else:
        raise NotImplementedError
    
    print_args_to_json(args)

    generate_and_save(
        dataset=dataset,
        args=args,
    )