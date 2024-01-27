import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import *
from prompts import tsp_dPrompts
from check.check_cmp_TSP_D import *
from utils import parse_xml_to_dict

import pandas as pd
import numpy as np
import json
import argparse

# Create the parser
parser = argparse.ArgumentParser(description='Run TSP-D model script')

# Add an argument for the model name
parser.add_argument('model', type=str, help='The name of the model to run')

# Parse the argument
args = parser.parse_args()

# Script logic using args.model as the model name
MODEL = str(args.model)

DATA_PATH = '../Data/TSP_Decision/'
RESULT_PATH = '../Results/'

def load_data():
    data_path = DATA_PATH
    all_data = []
    for level in range(10):
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

if __name__ == '__main__':
    tsp_d_Data = load_data()
    print(len(tsp_d_Data))
    tsp_d_Results = []

    MAX_TRY = 10
    for q in tsp_d_Data:
        threshold = q.iloc[-1, 0] # therashold is the last row
        distance_matrix = q.iloc[:-1].values # distance matrix is the rest of the rows
        output_dict = {}
        num_try = 0
        while num_try < MAX_TRY:
            try:
                llm_string = runTSP_D(distance_matrix, threshold)
                output, reasoning = parse_xml_to_dict(llm_string)
                output_dict['output'] = output
                output_dict['correctness'] = tsp_decision_check(distance_matrix, threshold, output)
                output_dict['reasoning'] = reasoning
                break
            except Exception as e:
                print(f"Attempt {num_try + 1} failed: {e}")
                num_try += 1
        if output_dict:
            tsp_d_Results.append(output_dict)
        else:
            print(f"Failed to run {q}")
            tsp_d_Results.append({'output': '', 'correctness': False})

    # Save the results
    with open(RESULT_PATH + MODEL + '_' + 'tsp_d_Results.json', 'a') as f:
        f.write(json.dumps(tsp_d_Results) + '\n')
