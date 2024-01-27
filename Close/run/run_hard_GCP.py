import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import *
from prompts import gcpPrompts
from check.check_hard_GCP import *

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

DATA_PATH = '../Data/GCP/'
RESULT_PATH = '../Results/'


def load_data():
    data_path = DATA_PATH
    all_data = []
    for file_num in range(10):
        with open(data_path+"synthesized_data_GCP_{}.txt".format(file_num)) as f:
            data = f.read()
        all_data += data.split('\n\n')[:-1]
    return all_data

def runGCP(q, p=gcpPrompts): # q is the data for the HP-hard question, p is the prompt
    # print(q)
    chromatic_number = q.split('\n')[0][-1] # last character of the first line
    number_of_vertices = q.split('\n')[1].split(' ')[2] # third word of the second line
    prompt_text = p['Intro'] + '\n' \
        + p['Initial_question'].format(max_vertices=number_of_vertices,max_colors=chromatic_number) + '\n' \
        + p['Output_content'] + '\n' \
        + p['Output_format'] + \
        '\n The graph is below: \n'
    for line in q.split('\n')[2:]:
        vertex_list = line.split(' ')
        this_line = "Vertex {} is connected to vertex {}.".format(vertex_list[1],vertex_list[2])
        prompt_text += this_line + '\n'
    
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
    # load data
    gcpData = load_data()
    # print(len(gcpData))
    gcpResults = []

    print("Using model: {}".format(MODEL))

    MAX_TRY = 10 # updated MAX_TRY
    for q in gcpData:
        output_dict = {}
        num_try = 0
        while num_try < MAX_TRY:
            try:
                output = runGCP(q)
                output_dict['output'] = output
                output_dict['correctness'] = gcpCheck(q, output)
                break
            except Exception as e:
                print(f"Attempt {num_try+1} failed: {e}")
                num_try += 1
        if output_dict:
            gcpResults.append(output_dict)
        else:
            print(f"Failed to run {q}")
            gcpResults.append({'output': '', 'correctness': False})
    # save the results
    with open(RESULT_PATH+MODEL+'_'+'gcpResults.json', 'a') as f:
        f.write(json.dumps(gcpResults) + '\n')
