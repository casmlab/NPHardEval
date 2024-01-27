import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import *
from prompts import mfpPrompts
from check.check_p_MFP import *
from utils import parse_xml_to_dict

import pandas as pd
import numpy as np
import json
import argparse

# Create the parser
parser = argparse.ArgumentParser(description='Run MFP model script')

# Add an argument for the model name
parser.add_argument('model', type=str, help='The name of the model to run')

# Parse the argument
args = parser.parse_args()

# Script logic using args.model as the model name
MODEL = str(args.model)

DATA_PATH = '../Data/MFP/'
RESULT_PATH = '../Results/'

def load_data():
    data_path = DATA_PATH
    with open(data_path + "mfp_instances.json", 'r') as f:
        all_data = json.load(f)
    return all_data

def runMFP(q, p=mfpPrompts):
    source_node = q['source']
    sink_node = q['sink']

    edges = q['edges']
    prompt_text = p['Intro'] + '\n' + \
                  p['Initial_question'].format(source_node=source_node, sink_node=sink_node) + '\n' + \
                  p['Output_content'] + '\n' + \
                  p['Output_format'] + \
                  "\n The capacities of the network's edges are as follows: \n"
    for edge in edges:
        this_line = f"Edge from {edge['from']} to {edge['to']} has a capacity of {edge['capacity']}."
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
    mfpData = load_data()
    mfpResults = []

    print("Using model: {}".format(MODEL))

    MAX_TRY = 10
    for q in mfpData:
        output_dict = {}
        num_try = 0
        while num_try < MAX_TRY:
            try:
                llm_string = runMFP(q)
                output, reasoning = parse_xml_to_dict(llm_string)
                output_dict['output'] = output
                output_dict['correctness'] = mfp_check(q, output)
                output_dict['reasoning'] = reasoning
                break
            except Exception as e:
                print(f"Attempt {num_try + 1} failed: {e}")
                num_try += 1
        if output_dict:
            mfpResults.append(output_dict)
        else:
            print(f"Failed to run {q}")
            mfpResults.append({'output': '', 'correctness': False})

    # Save the results
    with open(RESULT_PATH + MODEL + '_' + 'mfpResults.json', 'a') as f:
        f.write(json.dumps(mfpResults) + '\n')
