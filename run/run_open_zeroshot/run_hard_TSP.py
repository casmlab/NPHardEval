import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append('../..')

from models import *
from prompts import tspPrompts
from check.check_hard_TSP import *

import pandas as pd
import numpy as np
import json
from tqdm import tqdm

import argparse

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

    if MODEL.startswith('mistral'):
        output = run_mistral(all_prompts)
    elif MODEL.startswith('yi'):
        output = run_yi(all_prompts)
    elif MODEL.startswith('mpt'):
        output = run_mpt(all_prompts)
    elif MODEL.startswith('phi'):
        output = run_phi(all_prompts)
    elif MODEL.startswith('vicuna'):
        output = run_vicuna(all_prompts)
    else:
        raise NotImplementedError
    return output


if __name__ == '__main__':
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

    DATA_PATH = '../../Data/Zeroshot/TSP/'
    RESULT_PATH = '../../Results/'


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
    with open(RESULT_PATH+MODEL+'_'+'tspResults.json', 'a') as f:
        f.write(json.dumps(tspResults) + '\n')
