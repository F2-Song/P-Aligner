<h1 align="center">
    <br>P-Aligner: Pre-Aligning LLMs via Principled Instruction Synthesis
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

Large Language Models (LLMs) can fail to align with human preference on flawed instructions, leaving large room for improvement along multiple dimensions via pre-aligning instructions before normal decoding. Current methods rely on high test-time search costs, or end-to-end but unclear rewrite. In this work, we show that efficient and effective preference alignment can be achieved by P-Aligner, a lightweight module generating instructions which hold the original intents while being expressed in a human-preferred way. It is trained on UltraPrompt, a dataset synthesized via our proposed principle-guided pipeline using Monte-Carlo Tree Search, exploring candidate instructions that closely tied to human preference. Experiments show that it beats baselines across various models and benchmarks, including win-rate gains of 28.35% and 8.69% on GPT-4-turbo and Gemma-2-SimPO, respectively. More analyses validate its effectiveness and efficiency from multiple perspectives, like data quality, search strategies and time overhead.

## More information

Due to space limitations, we include additional information related to P-Aligner in the repository. These materials includes iterative-optimization experiments, dataset details, principle definitions, more experimental configurations, prompt templates, and representative cases, which are supplementary and do not affect the self-contained nature of the main paper. Please refer to [**our technical appendix**](https://github.com/F2-Song/P-Aligner/appendix.pdf) for these details.

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
