import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import *
from prompts import tspPrompts
from check.check_hard_TSP import *

import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from utils import run_opensource_models

import argparse

def load_data():
    data_path = DATA_PATH
    all_data = []
    if 'test_1' in data_path:
        n = 21
    elif 'test_2' in data_path:
        n = 31 
    else:
        n = 10
    start = n - 10
    for level in range(start, n):
        for file_num in range(10):
            # read np arrary
            df = pd.read_csv(data_path+"synthesized_data_TSP_level_{}_instance_{}.csv".format(level,file_num+1),
                                header=None, 
                                index_col=False)
            # transform df to 
            all_data.append(df)
    return all_data

def runTSP(MODEL,q, p=tspPrompts): # q is the data for the HP-hard question, p is the prompt
    total_cities = q.shape[0]
    prompt_text = p['Intro'] + '\n' \
        + p['Initial_question'].format(total_cities=total_cities) + '\n' \
        + p['Output_content'] + '\n' \
        + p['Output_format'] + \
        '\n The distances between cities are below: \n'
    for i in range(q.shape[0]):
        for j in range(q.shape[1]):
            if i < j: # only use the upper triangle
                this_line = "The path between City {} and City {} is with distance {}.".format(i,j,q.iloc[i,j])
                prompt_text += this_line + '\n'
    # output = run_gpt(prompt_text,model = MODEL)
    # remove \n in the output

    # get output
    if 'gpt' in MODEL:
        output = run_gpt(prompt_text,model = MODEL)
    elif 'claude' in MODEL:
        output = run_claude(text_prompt=prompt_text,model = MODEL)
    else:
        # raise error
        print('Model not found')
    return output

def run_opensource_TSP(qs, p=tspPrompts): # q is the data for the HP-hard question, p is the prompt
    all_prompts = []
    for q in tqdm(qs):
        total_cities = q.shape[0]
        prompt_text = p['Intro'] + '\n' \
            + p['Initial_question'].format(total_cities=total_cities) + '\n' \
            + p['Output_content'] + '\n' \
            + p['Output_format'] + \
            '\n The distances between cities are below: \n'
        for i in range(q.shape[0]):
            for j in range(q.shape[1]):
                if i < j: # only use the upper triangle
                    this_line = "The path between City {} and City {} is with distance {}.".format(i,j,q.iloc[i,j])
                    prompt_text += this_line + '\n'
        all_prompts.append(prompt_text)

    outputs = run_opensource_models(args, MODEL, all_prompts)
    return outputs


if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description='Run model script')

    # Add an argument for the model name
    parser.add_argument('model', type=str, help='The name of the model to run')
    parser.add_argument('--tuned_model_dir', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='../Data/TSP/', help='../Data/finetune_data/test_1/TSP/')

    # Parse the argument
    args = parser.parse_args()

    # Your script's logic here, using args.model as the model name
    MODEL = str(args.model)

    # MODEL = 'gpt-4-1106-preview'
    # # models: gpt-4-1106-preview, gpt-3.5-turbo-1106, claude-2, claude-instant, palm-2

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
    tspData = load_data()
    tspResults = []
    print('number of datapoints: ', len(tspData))

    print("Using model: {}".format(MODEL))

    outputs = run_opensource_TSP(tspData)
    for q, output in zip(tspData, outputs):
        output_dict = {}
        output_dict['output'] = output
        correctness = tspCheck(q,output)
        output_dict['correctness'] = correctness
        tspResults.append(output_dict)
    # save the results
    if args.tuned_model_dir:
        number_of_benchmarks = args.tuned_model_dir.split('_')[-1]
        with open(RESULT_PATH+MODEL+'_'+'tspResults_benchmarks{}.json'.format(number_of_benchmarks), 'w') as f:
            f.write(json.dumps(tspResults) + '\n')
    else:
        with open(RESULT_PATH+MODEL+'_'+'tspResults.json', 'w') as f:
            f.write(json.dumps(tspResults) + '\n')

# if __name__ == '__main__':
#     tspData = load_data()
#     print(len(tspData))
#     tspResults = []

#     print("Using model: {}".format(MODEL))

#     MAX_TRY = 10 # updated MAX_TRY
#     for q in tspData:
#         output_dict = {}
#         num_try = 0
#         # print(q)
#         # print("_________________________________________________________")
#         while num_try < MAX_TRY:
#             try:
#                 output = runTSP(q)
#                 print(q)
#                 print(output)
#                 output_dict['output'] = output
#                 output_dict['correctness'] = tspCheck(q, output)
#                 break
#             except Exception as e:
#                 print(f"Attempt {num_try+1} failed: {e}")
#                 num_try += 1
#         if output_dict:
#             tspResults.append(output_dict)
#         else:
#             print(f"Failed to run {q}")
#             tspResults.append({'output': '', 'correctness': False})

#     # save the results
#     with open(RESULT_PATH+MODEL+'_'+'tspResults.json', 'a') as f:
#         f.write(json.dumps(tspResults) + '\n')
