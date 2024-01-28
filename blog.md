---
title: "NPHardEval: Unveiling the Reasoning Abilities of Large Language Models through Complexity Classes and Dynamic Updates"
---


https://github.com/casmlab/NPHardEval


# NPHardEval: Unveiling the Reasoning Abilities of Large Language Models through Complexity Classes and Dynamic Updates

Welcome to an exploration of [NPHardEval](https://arxiv.org/abs/2312.14890), a cutting-edge benchmark revolutionizing the evaluation of Large Language Models' (LLMs) reasoning abilities. Developed by researchers from the University of Michigan and Rutgers University, [NPHardEval](https://arxiv.org/abs/2312.14890) introduces a dynamic, complexity-based framework for assessing LLMs. It poses 900 algorithmic questions spanning the NP-Hard complexity class and lower, designed to rigorously test LLMs' reasoning abilities.

## A Unique Approach to LLM Evaluation

[NPHardEval](https://arxiv.org/abs/2312.14890) stands apart by employing computational complexity classes, offering a quantifiable and robust measure of LLM reasoning skills. The benchmark's tasks mirror real-world decision-making challenges, enhancing its relevance and applicability. Regular monthly updates of the benchmark data points mitigate the risk of model overfitting, ensuring a reliable evaluation. 

In particular, our major contributions are two-fold:
- LLM Benchmarking Strategies:
    - Dynamic Benchmark: The method allows for the automatic generation of questions so that we can update the benchmark on a monthly basis. This monthly-refreshed benchmark helps prevent model's overfitting as we can always generate novel questions with varying difficulty levels for evaluation. 
    - Automatic Checking Mechanisms: The questions in the benchmark are based on algorithmically computable problems. Human intervention is not required to determine the correctness of the LLM's responses.
- LLM Reasoning:
    - Defining Reasoning via Complexity Classes: The questions in the benchmark utilized are grounded in the established computational complexity hierarchy, a concept extensively studied in theoretical computer science. This foundation enables us to leverage existing research to rigorously and quantitatively measure an LLM's logical reasoning extent.
    - Core Reasoning Tasks focusing on Logics: The benchmark excludes numerical computation from the questions, which is notably difficult for LLM. This focus allows for a more accurate evaluation of an LLM's pure logical reasoning ability, as numerical computation can obscure this assessment.


## Experimentation and Insights

The benchmark includes comprehensive experiments to analyze LLMs across various complexity classes and difficulty levels. It delves into the nuances of LLM performance, providing valuable insights into their reasoning strengths and limitations. In general:
- Close-source models generally perform better than open-source models.
- Models generally perform better on less-complex questions, i.e. easier complexity classes.

## Setting up NPHardEval Benchmark

To set up the NPHardEval Benchmark, we had to follow a few steps:

1. Environment setup: after cloning the repository to the local machine, we installed the required python library with `conda`. 
   ```bash
   conda create --name llm_reason python==3.10
   conda activate llm_reason
   git clone https://github.com/casmlab/NPHardEval.git
   pip install -r requirements.txt
   ```
2. Set-up API keys: we fetched API keys and changed the corresponding entries in `secrets.txt`.
3. Example Commands: we evaluated the model with the NPHardEval benchmark. For example, if we want to use the GPT 4 Turbo model (GPT-4-1106-preview) and the edit distance problem (EDP) for evaluation: 
    - For its zeroshot experiment, we can use:
      ```
      cd Close/run
      python run_p_EDP.py gpt-4-1106-preview
      ```
    - For its fewshot experiment, 
      ```
      cd Close/run
      python run_p_EDP_few.py gpt-4-1106-preview self
      ```

We currrently support fewshot examples from the same question (self), and may support examples from other questions (other) in the future.

We have published the leaderboard on [Huggingface](https://huggingface.co/spaces/hyfrankl/NPHardEval-leaderboard)  and may support model submission in the future.

## Join the Conversation
[The NPHardEval benchmark](https://huggingface.co/spaces/hyfrankl/NPHardEval-leaderboard), along with its [dataset](https://github.com/casmlab/NPHardEval/releases) and [code](https://github.com/casmlab/NPHardEval), is available on Github for community access and contributions.

Engage with this pioneering work and explore the frontiers of LLM reasoning abilities at [NPHardEval GitHub Repository](https://github.com/casmlab/NPHardEval).

Stay tuned for more updates and deep dives into the world of LLM evaluation with NPHardEval!

