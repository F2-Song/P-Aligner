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
        "--dataset_path",
        type=str,
        default="",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="",  
        help="Path to the output file",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
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