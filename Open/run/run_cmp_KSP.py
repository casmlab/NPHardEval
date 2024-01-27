import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import *
from prompts import kspPrompts
from check.check_cmp_KSP import *

import pandas as pd
import numpy as np
import json
import argparse
from tqdm import tqdm
from utils import run_opensource_models

def load_data():
    data_path = DATA_PATH
    with open(data_path + "ksp_instances.json", 'r') as f:
        all_data = json.load(f)
    return all_data

def runKSP(MODEL,q, p=kspPrompts):
    knapsack_capacity = q['knapsack_capacity']
    items = q['items']
    prompt_text = p['Intro'] + '\n' + \
                  p['Initial_question'].format(knapsack_capacity=knapsack_capacity) + '\n' + \
                  p['Output_content'] + '\n' + \
                  p['Output_format'] + \
                  '\n The items details are as below: \n'
    for item in items:
        this_line = f"Item {item['id']} has weight {item['weight']} and value {item['value']}."
        prompt_text += this_line + '\n'

    if 'gpt' in MODEL:
        output = run_gpt(prompt_text, model=MODEL)
    elif 'claude' in MODEL:
        output = run_claude(prompt_text, model=MODEL)
    else:
        print('Model not found')
        return None

    return output

def run_opensource_KSP(qs, p=kspPrompts):
    all_prompts = []
    for q in tqdm(qs):
        knapsack_capacity = q['knapsack_capacity']
        items = q['items']
        prompt_text = p['Intro'] + '\n' + \
                    p['Initial_question'].format(knapsack_capacity=knapsack_capacity) + '\n' + \
                    p['Output_content'] + '\n' + \
                    p['Output_format'] + \
                    '\n The items details are as below: \n'
        for item in items:
            this_line = f"Item {item['id']} has weight {item['weight']} and value {item['value']}."
            prompt_text += this_line + '\n'
        all_prompts.append(prompt_text)

    output = run_opensource_models(args, MODEL, all_prompts)
    return output


if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description='Run KSP model script')

    # Add an argument for the model name
    parser.add_argument('model', type=str, help='The name of the model to run')
    parser.add_argument('--tuned_model_dir', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='../Data/KSP/', help='../Data/finetune_data/test_1/KSP/')
    

    # Parse the argument
    args = parser.parse_args()

    # Script logic using args.model as the model name
    MODEL = str(args.model)

    DATA_PATH = args.data_dir
    if args.tuned_model_dir:
        RESULT_PATH = '../Results/finetuned/'
        if 'test_1' in args.data_dir:
            RESULT_PATH += 'test_1/'
        elif 'test_2' in args.data_dir:
            RESULT_PATH += 'test_2/'
        else:
            RESULT_PATH += 'original/'
    else:
        RESULT_PATH = '../Results/'
        if 'test_1' in args.data_dir:
            RESULT_PATH += 'test_1/'
        elif 'test_2' in args.data_dir:
            RESULT_PATH += 'test_2/'

    # load data
    kspData = load_data()
    #kspData = kspData[:20]
    kspdResults = []
    print('number of datapoints: ', len(kspData))

    print("Using model: {}".format(MODEL))

    outputs = run_opensource_KSP(kspData)
    for q, output in zip(kspData, outputs):
        output_dict = {}
        output, reasoning = parse_xml_to_dict(output)
        output_dict['output'] = output
        output_dict['correctness'] = kspCheck(q, output)
        output_dict['reasoning'] = reasoning
        kspdResults.append(output_dict)
    # save the results
    if args.tuned_model_dir:
        number_of_benchmarks = args.tuned_model_dir.split('_')[-1]
        with open(RESULT_PATH+MODEL+'_'+'kspResults_benchmarks{}.json'.format(number_of_benchmarks), 'w') as f:
            f.write(json.dumps(kspdResults) + '\n')
    else:
        with open(RESULT_PATH+MODEL+'_'+'kspResults.json', 'w') as f:
            f.write(json.dumps(kspdResults) + '\n')


# if __name__ == '__main__':
#     kspData = load_data()
#     kspResults = []

#     print("Using model: {}".format(MODEL))

#     MAX_TRY = 1
#     for q in kspData:
#         output_dict = {}
#         num_try = 0
#         while num_try < MAX_TRY:
#             try:
#                 llm_string = runKSP(q)
#                 output, reasoning = parse_xml_to_dict(llm_string)
#                 output_dict['output'] = output
#                 output_dict['correctness'] = kspCheck(q, output)
#                 output_dict['reasoning'] = reasoning
#                 break
#             except Exception as e:
#                 print(f"Attempt {num_try + 1} failed: {e}")
#                 num_try += 1
#         if output_dict:
#             kspResults.append(output_dict)
#         else:
#             print(f"Failed to run {q}")
#             kspResults.append({'output': '', 'correctness': False})

#     # Save the results
#     with open(RESULT_PATH + MODEL + '_' + 'kspResults.json', 'a') as f:
#         f.write(json.dumps(kspResults) + '\n')