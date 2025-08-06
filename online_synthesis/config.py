import random
import numpy as np
import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--id",
        type=str,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--rule_name",
        type=str,
        default="rule_P-Aligner",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--exploration_weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--last_output_path",
        type=str,
        default="",  
        help="Path to the output file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="",  
        help="Path to the output file",
    )
    parser.add_argument(
        "--flag_path",
        type=str,
        default="thread_flag",
        help="Path to the flag file",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="gpt-4",
    )
    parser.add_argument(
        "--lm_optimizer_type",
        type=str,
        default="openai", # openai | local
    )
    parser.add_argument(
        "--port",
        type=int,
        default=12000,
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--rank_sum",
        type=int,
        default=1,
    )
    return parser.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

args = parse_args()
setup_seed(args.seed)
args_message = '\n'+'\n'.join([f'{k:<40}: {v}' for k, v in vars(args).items()])
print(args_message)