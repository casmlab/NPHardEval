import urllib.request
import json
import os
import ssl
import requests
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from tqdm import tqdm
import torch

'''
Open-source models
'''

def run_mistral(text_prompt, model_name=None):
    sampling_params = SamplingParams(temperature=0, max_tokens=1000)
    if model_name is None:
        model_name = 'mistralai/Mistral-7B-Instruct-v0.1'
    else:
        download_dir = model_name
    llm = LLM(
        model=model_name,
        tensor_parallel_size=4, 
        max_num_seqs=4,
        max_num_batched_tokens=4 * 8192,
    )
    text_prompt = ['<s>[INST] '+ t + '[INST]\n' for t in text_prompt]
    predictions = llm.generate(text_prompt, sampling_params)
    all_predictions = []
    for RequestOutput in predictions:
        output = RequestOutput.outputs[0].text
        all_predictions.append(output)
    return all_predictions


def run_mixtral(text_prompt):
    sampling_params = SamplingParams(temperature=0, max_tokens=1000)
    llm = LLM(
        model="mistralai/Mistral-7B-Instruct-v0.1",
        tensor_parallel_size=4, 
        max_num_seqs=4,
        max_num_batched_tokens=4 * 8192,
    )
    text_prompt = ['<s>[INST] '+ t + '[INST]\n' for t in text_prompt]
    predictions = llm.generate(text_prompt, sampling_params)
    all_predictions = []
    for RequestOutput in predictions:
        output = RequestOutput.outputs[0].text
        all_predictions.append(output)
    return all_predictions


def run_yi(text_prompt):
    sampling_params = SamplingParams(temperature=0, max_tokens=1000)
    llm = LLM(
        model='01-ai/Yi-34B-Chat',
        tensor_parallel_size=4, 
        max_num_seqs=4,
        max_num_batched_tokens=4 * 4096,
    )
    text_prompt = ['<|im_start|> '+ t + '<|im_end|>\n<|im_start|>assistant\n' for t in text_prompt]
    predictions = llm.generate(text_prompt, sampling_params)
    all_predictions = []
    for RequestOutput in predictions:
        output = RequestOutput.outputs[0].text
        all_predictions.append(output)
    return all_predictions

def run_vicuna(text_prompt, model_name=None):
    sampling_params = SamplingParams(temperature=0, max_tokens=1000)
    if model_name is None:
        model_name = 'lmsys/vicuna-13b-v1.3'
    else:
        download_dir = model_name
    llm = LLM(
        model=model_name,
        download_dir=download_dir,
        tensor_parallel_size=4, 
        max_num_seqs=4,
        max_num_batched_tokens=4 * 2048,
    )
    predictions = llm.generate(text_prompt, sampling_params)
    all_predictions = []
    for RequestOutput in predictions:
        output = RequestOutput.outputs[0].text
        all_predictions.append(output)
    return all_predictions

def run_phi(text_prompt):
    sampling_params = SamplingParams(temperature=0, max_tokens=1000)
    llm = LLM(
        model='microsoft/phi-1_5',
        trust_remote_code=True,
        tensor_parallel_size=4, 
        max_num_seqs=4,
        max_num_batched_tokens=4 * 2048,
    )
    text_prompt = [t + 'Answer: \n' for t in text_prompt]
    predictions = llm.generate(text_prompt, sampling_params)
    all_predictions = []
    for RequestOutput in predictions:
        output = RequestOutput.outputs[0].text
        all_predictions.append(output)
    return all_predictions

def run_phi_2(text_prompt, model_name=None):
    torch.set_default_device("cuda")
    if model_name is None:
        model_name = 'microsoft/phi-2'
    else:
        download_dir = model_name
    all_predictions = []
    # sampling_params = SamplingParams(temperature=0, max_tokens=1000)
    # llm = LLM(
    #     model = model_name,
    #     download_dir=download_dir,
    #     trust_remote_code=True,
    #     tensor_parallel_size=4, 
    #     max_num_seqs=4,
    #     max_num_batched_tokens=4 * 2048,
    # )
    # predictions = llm.generate(text_prompt, sampling_params)
    # all_predictions = []
    # for RequestOutput in predictions:
    #     output = RequestOutput.outputs[0].text
    #     all_predictions.append(output)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side='left')
    model.config.pad_token_id = tokenizer.pad_token_id = 0
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config_eos_token_id = tokenizer.eos_token_id
    text_prompt = ['Instruct: '+ t + 'Output: \n' for t in text_prompt]
    batch_size = 10 
    i = 0
    for _ in tqdm(range(0,int(len(text_prompt)/batch_size))):
        inputs = tokenizer(text_prompt[i:i+batch_size], return_tensors="pt", return_attention_mask=False, padding=True)
        outputs = model.generate(**inputs, max_length=2048)
        texts = tokenizer.batch_decode(outputs)
        output = [text.replace(one_text_prompt, '') for text, one_text_prompt in zip(texts, text_prompt[i:i+batch_size])]
        all_predictions += output
        i = i+batch_size
    return all_predictions


def run_mpt(text_prompt):
    sampling_params = SamplingParams(temperature=0, max_tokens=1000)
    llm = LLM(
        model='mosaicml/mpt-30b-instruct',
        trust_remote_code=True,
        tensor_parallel_size=4, 
        max_num_seqs=4,
        max_num_batched_tokens=4 * 4096,
    )
    predictions = llm.generate(text_prompt, sampling_params)
    all_predictions = []
    for RequestOutput in predictions:
        output = RequestOutput.outputs[0].text
        all_predictions.append(output)
    return all_predictions

def run_qwen(text_prompt, model_name=None):
    sampling_params = SamplingParams(temperature=0, max_tokens=1000)
    if model_name is None:
        model_name = 'Qwen/Qwen-14B-Chat'
    else:
        download_dir = model_name
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        tensor_parallel_size=4, 
        max_num_seqs=4,
        max_num_batched_tokens=4 * 8192,
    )
    predictions = llm.generate(text_prompt, sampling_params)
    all_predictions = []
    for RequestOutput in predictions:
        output = RequestOutput.outputs[0].text
        all_predictions.append(output)
    return all_predictions

if __name__ == '__main__':
    text_prompt = ['generate a random sentence', 'give me a dollar']
    output = run_mpt(text_prompt)
    print(output)