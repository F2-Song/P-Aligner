from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

import sys
sys.path.append("../")
from utils.utils import save_dataset

def generate(
    dataset,
    args,
):
    # load the tokenizer and pre-process the instructions
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    raw_instructions = [sample["instruction"] for sample in dataset]
    instructions = [tokenizer.get_context(raw_instruction) for raw_instruction in raw_instructions]

    # load P-Aligner
    model = LLM(
        model=args.model_path,
        gpu_memory_utilization=0.9,
        enable_prefix_caching=True,
        dtype="bfloat16",
    )

    outputs = model.generate(
        instructions,
        sampling_params=SamplingParams(
            n=1,
            temperature=0.0,
            max_tokens=4096,
        ),
    )

    better_instructions = []
    for raw_instruction, output in zip(raw_instructions, outputs):
        better_instructions.append(
            tokenizer.parse_output(
                output.outputs[0].text,
                raw_instruction,
            )
        )
    
    res = []
    for i, sample in enumerate(dataset):
        res.append(
            {
                "instruction": sample["instruction"],
                "better_instruction": better_instructions[i],
            }
        )

    save_dataset(res, args.output_path, flag="w")