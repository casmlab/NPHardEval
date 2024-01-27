# NPHardEval: Benchmarking Reasoning Ability of Large Language Models via Complexity Classes

<a href='https://arxiv.org/abs/2312.14890'><img src='https://img.shields.io/badge/Paper-PDF-red'></a> 
<a href="https://github.com/casmlab/NPHardEval/tags"><img alt="GitHub release" src="https://img.shields.io/github/tag/casmlab/NPHardEval"></a>
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/casmlab/NPHardEval/blob/main/LICENSE)


<div align="center">
    <img src="NP-hard.jpg" alt="Questions of different complexity classes" style="width:80%">
</div>

**NPHardEval** serves as a comprehensive benchmark for assessing the reasoning abilities of large language models (LLMs) through the lens of computational complexity classes. This repository contains datasets, data generation scripts, and experimental procedures designed to evaluate LLMs in various reasoning tasks.

Our benchmark offers several advantages compared with current benchmarks:
* Data construction grounded in the established computational complexity hierarchy
* Automatic checking mechanisms 
* Automatic generation of datapoints
* Complete focus on reasoning while exclude numerical computation

--------------------

## Quick Start
### Environment setup
```bash
conda create --name llm_reason python=3.10
conda activate llm_reason
git clone https://github.com/casmlab/NPHardEval.git
pip install -r requirements.txt
```

### Set-up API keys
Please set up your API keys in `secrets.txt`. **Please don't directly upload your keys to any public repository.**

### Example Commands
Let's use the GPT 4 Turbo model (GPT-4-1106-preview) and the EDP for example. 

For its zeroshot experiment, you can use:
```
cd Close/run
cd run_close_zeroshot
python run_hard_GCP.py gpt-4-1106-preview
```

For its fewshot experiment, 
```
cd Close/run
python run_p_BSP_few.py gpt-4-1106-preview self
```
We currrently support fewshot examples from the same question (self), and may support examples from other questions (other) in the future.

### Result Visualization
**Directory:** `summary`

This directory primarily focuses on the visualization and analysis of data related to the performance of Large Language Models (LLMs). It includes two key components:

**Jupyter Notebooks**: `result_complexities.ipynb` analyzes the accuracy rates of LLMs across various question complexity classes, while `result_problems.ipynb` examines specific problem instances and their solutions as processed by LLMs.

**Source Code** (`src` sub-directory): This sub-directory contains Python scripts dedicated to the visualization and analysis of ablation studies and research questions (RQs). The scripts, named as `visualize_ablation*.py` and `visualize_rq*.py`, provide detailed insights into different aspects of LLM performance and characteristics.

Overall, the summary directory serves as a comprehensive hub for assessing and understanding the efficacy of Large Language Models in handling questions of varying complexities and types.

--------------------

## Leaderboard

| model  | P | NP-complete | NP-hard
| ------------- | ------------- | ------------- |------------- |
| GPT-4  | 0.7218 | 0.3533 | 0.05705 |
| GPT-3.5  | 0.4933 | 0.1823 | 0.0351 |
| Claude-2 | 0.3127 | 0.4654 | 0.0242 |
| Claude-instant-1.2  | 0.2927 | 0.1914 | 0.0073 |
| Palm2 | 0.2223 | 0.2151 | 0.05633 |
| Yi-34b | 0.2618 | 0.2982 | 0.0079 |
| Mistral-7b | 0.07452 | 0.4024 | 0.0697 |
| MPT-30b | 0.0006 | 0.0 | 0.0 |
| Vicuna-13b | 0.08663 | 0.1242 | 0.0158 |
| Phi-1.5 | 0.0 | 0.0 | 0.0067 |

Metric: average weighted accuracy

Upcoming: Gemini, Mixtral (Mistral-7b MoE), Phi-2

The updated leaderboard is available on 🤗[Huggingface](https://huggingface.co/spaces/hyfrankl/NPHardEval-leaderboard).

--------------------


## Full Experiments
To successfully replicate the experiments detailed in this repository, the following prerequisites must be met:

1. **Access to Large Language Model (LLM) APIs**: Essential for interfacing with the LLMs under evaluation.
2. **Datasets**: Located in the `Data` directory, these datasets are vital for conducting both zero-shot and few-shot experiments.
3. **Script Utilization**: Scripts located in the `run` directory are designed to facilitate the experimental process. This repository is meticulously organized to support distinct experimental approaches for zero-shot and few-shot scenarios.

### Execution Script
**Directory:** `Close/run` and `Open/run` \
**File**: `run_how_to.sh`

This script is your primary tool for experiment execution. It is meticulously crafted to provide a seamless and efficient experimental workflow. Detailed instructions within `run.sh` guide users through each step of the process, ensuring a smooth and error-free execution.


### Key Takeaways on Experiments
We compare different foundation models' reasoning ability across task complexity and experimented with different prompt styles to understand their in-context learnability. Our study reveals a notable disparity in performance between closed-source and open-source models not only on general reasoning ability but also the disparity between "learning" and "mimicking". In particular, we found:

* All models exhibit decreased accuracy and increased failure rates with rising task complexity, especially at NP-Hard levels.
* The transition from P to NP-Complete complexity impacts the performance models differently.
* Closed-source models like GPT 4 Turbo and Claude 2 maintain consistent performance across difficulty levels of in-context examples, indicating robust learning from few-shot examples, while open-source models vary in adaptability.

--------------------

## Benchmark Construction
**Directory:** `Data`

The `Data` directory houses the datasets utilized in our study, categorized into `Fewshot` and `Zeroshot` datasets, corresponding to their respective experimental setups.

**Structure:**
```bash
$ tree -d Data 
Data
├── BSP
├── EDP
├── GCP
├── GCP_Decision
├── KSP
├── MSP
├── SPP
├── TSP
└── TSP_Decision
```

### Datapoints
The data used is under `data` directory. You can find the zeroshot/fewshot under the corresponding directory. They are the data used in our report.

### Answer Verification
**Directory**: `check`

Contained within this directory are utility functions crucial for verifying answers provided by the LLMs. These functions are automatically invoked during experiments executed via `run.sh`. As the experiment progresses, these utilities rigorously evaluate the responses from LLMs and compile the outcomes in the `Results` directory. This automated process ensures a comprehensive and objective assessment of the LLM's performance.

--------------------

## News
-[2024.1.26] We release the first version (V1) of NPHardEval with (1) We have **replaced** all data and (2) Several minor bugs fixed regarding data directory and code dependecy issues. 

-[2023.12.24] We release the default version (V0) of NPHardEval with data, answer-checking code, and example run files. The generation code will not be released, to avoid potential benchmark hacking.

## License
The source code of NPHardEval is licensed under [Apache 2.0](https://github.com/casmlab/NPHardEval/blob/main/LICENSE). The intended purpose is solely for research use.

--------------------

## Reference
```
@article{fan2023nphardeval,
  title={NPHardEval: Dynamic Benchmark on Reasoning Ability of Large Language Models via Complexity Classes},
  author={Fan, Lizhou and Hua, Wenyue and Li, Lingyao and Ling, Haoyang and Zhang, Yongfeng and Hemphill, Libby},
  journal={arXiv preprint arXiv:2312.14890},
  year={2023}
}
```


