import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import *
from prompts import tspPrompts
from check.check_hard_TSP import *

import pandas as pd
import numpy as np
import json

import argparse

# Create the parser
parser = argparse.ArgumentParser(description='Run model script')

# Add an argument for the model name
parser.add_argument('model', type=str, help='The name of the model to run')

# Parse the argument
args = parser.parse_args()

# Your script's logic here, using args.model as the model name
MODEL = str(args.model)

# MODEL = 'gpt-4-1106-preview'
# # models: gpt-4-1106-preview, gpt-3.5-turbo-1106, claude-2, claude-instant, palm-2

DATA_PATH = '../Data/TSP/'
RESULT_PATH = '../Results/'

def load_data():
    data_path = DATA_PATH
    all_data = []
    for level in range(10):
        for file_num in range(10):
            #df = pd.read_csv(data_path+"synthesized_data_TSP_level_{}_instance_{}.csv".format(file_num,file_num+1))
            # read np arrary
            df = pd.read_csv(data_path+"synthesized_data_TSP_level_{}_instance_{}.csv".format(level,file_num+1),
                                header=None, 
                                index_col=False)
            # transform df to 
            all_data.append(df)
    return all_data

def runTSP(q, p=tspPrompts): # q is the data for the HP-hard question, p is the prompt
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

if __name__ == '__main__':
    tspData = load_data()
    print(len(tspData))
    tspResults = []

    print("Using model: {}".format(MODEL))

    MAX_TRY = 10 # updated MAX_TRY
    for q in tspData:
        output_dict = {}
        num_try = 0
        # print(q)
        # print("_________________________________________________________")
        while num_try < MAX_TRY:
            try:
                output = runTSP(q)
                print(q)
                print(output)
                output_dict['output'] = output
                output_dict['correctness'] = tspCheck(q, output)
                break
            except Exception as e:
                print(f"Attempt {num_try+1} failed: {e}")
                num_try += 1
        if output_dict:
            tspResults.append(output_dict)
        else:
            print(f"Failed to run {q}")
            tspResults.append({'output': '', 'correctness': False})

    # save the results
    with open(RESULT_PATH+MODEL+'_'+'tspResults.json', 'a') as f:
        f.write(json.dumps(tspResults) + '\n')
