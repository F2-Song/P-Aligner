<h1 align="center">
    <br>P-Aligner: Enabling Pre-Alignment of Language Models via Principled Instruction Synthesis
</h1>
<p align="center">
    <a href="https://arxiv.org/abs/2508.04626">
        <img alt="Static Badge" src="https://img.shields.io/badge/arXiv-2508.04626-red">
    </a>
    <a href="https://huggingface.co/datasets/songff/UltraPrompt">
        <img alt="Static Badge" src="https://img.shields.io/badge/HFDataset-UltraPrompt-green">
    </a>
    <a href="https://huggingface.co/songff/P-Aligner">
        <img alt="Static Badge" src="https://img.shields.io/badge/HFModel-P--Aligner-yellow">
    </a>
    <a href="https://huggingface.co/songff/SinglePO">
        <img alt="Static Badge" src="https://img.shields.io/badge/HFModel-SinglePO-blue">
    </a>
    <a href="https://github.com/F2-Song/P-Aligner">
        <img alt="Static Badge" src="https://img.shields.io/badge/Github-P--Aligner-black">
    </a>
</p>

<p align="center">
    Authors: Feifan Song, Bofei Gao, Yifan Song, Yi Liu, Weimin Xiong, Yuyang Song, Tianyu Liu, Guoyin Wang and Houfeng Wang
</p>

## Overview

Large Language Models (LLMs) are expected to produce safe, helpful, and honest content during interaction with human users, but they frequently fail to align with such values when given flawed instructions, e.g., missing context, ambiguous directives, or inappropriate tone, leaving substantial room for improvement along multiple dimensions. A cost-effective yet high-impact way is to pre-align instructions before the model begins decoding. Existing approaches either rely on prohibitive test-time search costs or end-to-end model rewrite, which is powered by a customized training corpus with unclear objectives. In this work, we demonstrate that the goal of efficient and effective preference alignment can be achieved by P-Aligner, a lightweight module generating instructions that preserve the original intents while being expressed in a more human-preferred form. P-Aligner is trained on UltraPrompt, a new dataset synthesized via a proposed principle-guided pipeline using Monte-Carlo Tree Search, which systematically explores the space of candidate instructions that are closely tied to human preference. Experiments across different methods show that P-Aligner generally outperforms strong baselines across various models and benchmarks, including average win-rate gains of 28.35% and 8.69% on GPT-4-turbo and Gemma-2-SimPO, respectively. Further analyses validate its effectiveness and efficiency through multiple perspectives, including data quality, search strategies, iterative deployment, and time overhead.

## Easy Start
We provide scripts of instruction synthesis and P-Aligner inference for easy start.

```bash
# execute the pipeline with API calls; please first set the related environment variables in this script
bash exec/run_online.sh <task_id> <src_dataset_path> <tgt_dataset_path>

# execute the pipeline with SinglePO
bash exec/run_offline.sh <task_id> <src_dataset_path> <tgt_dataset_path>

# use the trained P-Aligner to align instructions
bash exec/run.sh <task_id> <src_dataset_path> <tgt_dataset_path>
```

## Training
To help acquire P-Aligner and SinglePO, as well as further exploration, we release [UltraPrompt](https://huggingface.co/datasets/songff/UltraPrompt) for direct use. You can also synthesize your own data with the above tools. 

After the synthesis processes end, just run `prepare_data.py` and `prepare_step_data.py` to get the data in ShareGPT-like formats for [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), a powerful and easy-to-use framework which we leverage to train P-Aligner and SinglePO.

## ⚠️Caution

We deeply thank you for your recognition on our work. 

Before scanning all material or running any code, please bear in mind that this project inevitably contains sensitive content and you should be well-prepared to have access to it. For example, the data used in evaluation and UltraPrompt may include unsafe information, such as misleading content or offensive instructions. Such content does not represent our attitudes, and should be handled carefully to avoid potential harm. We request you and any other potential users treat it responsibly without any use or distribution outside of research contexts.

## Citation
If you find this work useful, please consider citing:
```
@misc{song2025paligner,
  title={P-Aligner: Enabling Pre-Alignment of Language Models via Principled Instruction Synthesis},
  author={Song, Feifan and Gao, Bofei and Song, Yifan and Liu, Yi and Xiong, Weimin and Song, Yuyang and Liu, Tianyu and Wang, Guoyin and Wang, Houfeng},
  year={2025},
  eprint={2508.04626},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```