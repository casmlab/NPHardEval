import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import *
from prompts import gcp_dPrompts
from check.check_cmp_GCP_D import *
from utils import parse_xml_to_dict

import pandas as pd
import numpy as np
import json
import argparse

# Create the parser
parser = argparse.ArgumentParser(description='Run GCP-D model script')

# Add an argument for the model name
parser.add_argument('model', type=str, help='The name of the model to run')

# Parse the argument
args = parser.parse_args()

# Script logic using args.model as the model name
MODEL = str(args.model)

DATA_PATH = '../Data/GCP_Decision/'
RESULT_PATH = '../Results/'

def load_data():
    data_path = DATA_PATH
    all_data = []
    for file_num in range(10):
        with open(data_path + "decision_data_GCP_{}.txt".format(file_num)) as f:
            data = f.read()
        all_data += data.split('\n\n')[:-1]
    return all_data

def runGCP_D(q, p=gcp_dPrompts):
    number_of_colors = q.split('\n')[0].split()[-2] # last character of the first line
    number_of_vertices = q.split('\n')[1].split(' ')[2] # third word of the second line
    prompt_text = p['Intro'] + '\n' + \
                  p['Initial_question'].format(total_vertices=number_of_vertices, number_of_colors=number_of_colors) + '\n' + \
                  p['Output_content'] + '\n' + \
                  p['Output_format'] + '\n' + \
                    '\n The graph is below: \n'
    for line in q.split('\n')[2:]:
        vertex_list = line.split(' ')
        this_line = "Vertex {} is connected to vertex {}.".format(vertex_list[1],vertex_list[2])
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
    gcp_d_Data = load_data()
    print(len(gcp_d_Data))
    gcp_d_Results = []

    print("Using model: {}".format(MODEL))

    MAX_TRY = 10
    for q in gcp_d_Data:
        output_dict = {}
        num_try = 0
        while num_try < MAX_TRY:
            try:
                llm_string = runGCP_D(q)
                number_of_colors = int(q.split('\n')[0].split()[-2])
                output, reasoning = parse_xml_to_dict(llm_string)
                output_dict['output'] = output
                output_dict['correctness'] = gcp_decision_check(q, output, number_of_colors)
                output_dict['reasoning'] = reasoning
                break
            except Exception as e:
                print(f"Attempt {num_try + 1} failed: {e}")
                num_try += 1
        if output_dict:
            gcp_d_Results.append(output_dict)
        else:
            print(f"Failed to run {q}")
            gcp_d_Results.append({'output': '', 'correctness': False})

    # Save the results
    with open(RESULT_PATH + MODEL + '_' + 'gcp_d_Results.json', 'a') as f:
        f.write(json.dumps(gcp_d_Results) + '\n')
