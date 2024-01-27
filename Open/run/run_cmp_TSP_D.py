import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import *
from prompts import tsp_dPrompts
from check.check_cmp_TSP_D import *
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import argparse
from utils import run_opensource_models


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
            df = pd.read_csv(data_path + "decision_data_TSP_level_{}_instance_{}.csv".format(level, file_num + 1),
                             header=None, 
                             index_col=False)
            all_data.append(df)
    return all_data

def runTSP_D(adj_matrix, distance_limit, p=tsp_dPrompts):
    total_cities = adj_matrix.shape[0] # exclude the last row
    prompt_text = p['Intro'] + '\n' + \
                  p['Initial_question'].format(total_cities=total_cities, distance_limit=distance_limit) + '\n' + \
                  p['Output_content'] + '\n' + \
                  p['Output_format'] + '\n' + \
                  'The distances between cities are below: \n'
    
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if i < j:  # only use the upper triangle
                this_line = "The distance between City {} and City {} is {}.".format(i, j, adj_matrix[i, j])
                prompt_text += this_line + '\n'

    if 'gpt' in MODEL:
        output = run_gpt(prompt_text, model=MODEL)
    elif 'claude' in MODEL:
        output = run_claude(prompt_text, model=MODEL)
    else:
        print('Model not found')
        return None

    return output


def run_opensource_TSP_D(qs, p=tsp_dPrompts):
    all_prompts = []
    for q in tqdm(qs):
        threshold = q.iloc[-1, 0] # therashold is the last row
        adj_matrix = q.iloc[:-1].values # distance matrix is the rest of the rows
        total_cities = adj_matrix.shape[0] # exclude the last row
        prompt_text = p['Intro'] + '\n' + \
                    p['Initial_question'].format(total_cities=total_cities, distance_limit=threshold) + '\n' + \
                    p['Output_content'] + '\n' + \
                    p['Output_format'] + '\n' + \
                    'The distances between cities are below: \n'
        
        for i in range(adj_matrix.shape[0]):
            for j in range(adj_matrix.shape[1]):
                if i < j:  # only use the upper triangle
                    this_line = "The distance between City {} and City {} is {}.".format(i, j, adj_matrix[i, j])
                    prompt_text += this_line + '\n'
        all_prompts.append(prompt_text)

    output = run_opensource_models(args, MODEL, all_prompts)
    return output


if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description='Run TSP-D model script')

    # Add an argument for the model name
    parser.add_argument('model', type=str, help='The name of the model to run')
    parser.add_argument('--tuned_model_dir', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='../Data/TSP_Decision/', help='../Data/finetune_data/test_1/TSP_Decision/')

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

    tsp_d_Data = load_data()
    #tsp_d_Data = tsp_d_Data[:2]
    print(len(tsp_d_Data))
    tsp_d_Results = []

    outputs = run_opensource_TSP_D(tsp_d_Data)
    for result, instance in zip(outputs, tsp_d_Data):
        output_dict = {}
        threshold = instance.iloc[-1, 0] # therashold is the last row
        distance_matrix = instance.iloc[:-1].values # distance matrix is the rest of the rows
        output, reasoning = parse_xml_to_dict(result)
        output_dict['output'] = output
        output_dict['correctness'] = tsp_decision_check(distance_matrix, threshold, output)
        output_dict['reasoning'] = reasoning
        tsp_d_Results.append(output_dict)

    # Save the results
    if args.tuned_model_dir:
        number_of_benchmarks = args.tuned_model_dir.split('_')[-1]
        with open(RESULT_PATH+MODEL+'_'+'tsp_d_Results_benchmarks{}.json'.format(number_of_benchmarks), 'w') as f:
            f.write(json.dumps(tsp_d_Results) + '\n')
    else:
        with open(RESULT_PATH+MODEL+'_'+'tsp_d_Results.json', 'w') as f:
            f.write(json.dumps(tsp_d_Results) + '\n')


# if __name__ == '__main__':
#     # Create the parser
#     parser = argparse.ArgumentParser(description='Run TSP-D model script')

#     # Add an argument for the model name
#     parser.add_argument('model', type=str, help='The name of the model to run')

#     # Parse the argument
#     args = parser.parse_args()

#     # Script logic using args.model as the model name
#     MODEL = str(args.model)

#     DATA_PATH = '../Data/TSP_Decision/'
#     RESULT_PATH = '../Results/'


#     tsp_d_Data = load_data()
#     print(len(tsp_d_Data))
#     tsp_d_Results = []

#     MAX_TRY = 10
#     for q in tsp_d_Data:
#         threshold = q.iloc[-1, 0] # therashold is the last row
#         distance_matrix = q.iloc[:-1].values # distance matrix is the rest of the rows
#         output_dict = {}
#         num_try = 0
#         while num_try < MAX_TRY:
#             try:
#                 llm_string = runTSP_D(distance_matrix, threshold)
#                 output, reasoning = parse_xml_to_dict(llm_string)
#                 output_dict['output'] = output
#                 output_dict['correctness'] = tsp_decision_check(distance_matrix, threshold, output)
#                 output_dict['reasoning'] = reasoning
#                 break
#             except Exception as e:
#                 print(f"Attempt {num_try + 1} failed: {e}")
#                 num_try += 1
#         if output_dict:
#             tsp_d_Results.append(output_dict)
#         else:
#             print(f"Failed to run {q}")
#             tsp_d_Results.append({'output': '', 'correctness': False})

#     # Save the results
#     with open(RESULT_PATH + MODEL + '_' + 'tsp_d_Results.json', 'a') as f:
#         f.write(json.dumps(tsp_d_Results) + '\n')