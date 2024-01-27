import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import *
from prompts import bspPrompts
from check.check_p_BSP import *
from utils import parse_xml_to_dict

import pandas as pd
import numpy as np
import json
import argparse

# Create the parser
parser = argparse.ArgumentParser(description='Run BSP model script')

# Add an argument for the model name
parser.add_argument('model', type=str, help='The name of the model to run')

# Parse the argument
args = parser.parse_args()

# Script logic using args.model as the model name
MODEL = str(args.model)

DATA_PATH = '../Data/BSP/'
RESULT_PATH = '../Results/'

def load_data():
    data_path = DATA_PATH
    with open(data_path + "bsp_instances.json", 'r') as f:
        all_data = json.load(f)
    return all_data

def runBSP(q, p=bspPrompts):
    target_value = q['target']
    # TO-DO: fix data not being sorted
    array = sorted(q['array'])
    prompt_text = p['Intro'] + '\n' + \
                  p['Initial_question'].format(target_value=target_value) + '\n' + \
                  p['Output_content'] + '\n' + \
                  p['Output_format'] + \
                  '\n The sorted array elements are: ' + ', '.join(map(str, array)) + '\n'

    if 'gpt' in MODEL:
        output = run_gpt(prompt_text, model=MODEL)
    elif 'claude' in MODEL:
        output = run_claude(prompt_text, model=MODEL)
    else:
        print('Model not found')
        return None

    return output

if __name__ == '__main__':
    bspData = load_data()
    bspResults = []

    print("Using model: {}".format(MODEL))

    MAX_TRY = 10
    for q in bspData:
        output_dict = {}
        num_try = 0
        while num_try < MAX_TRY:
            try:
                llm_string = runBSP(q)
                output, reasoning = parse_xml_to_dict(llm_string)
                output_dict['output'] = output
                output_dict['correctness'] = bsp_check(q, output)
                output_dict['reasoning'] = reasoning
                break
            except Exception as e:
                print(f"Attempt {num_try + 1} failed: {e}")
                num_try += 1
        if output_dict:
            bspResults.append(output_dict)
        else:
            print(f"Failed to run {q}")
            bspResults.append({'output': '', 'correctness': False})

    # Save the results
    with open(RESULT_PATH + MODEL + '_' + 'bspResults.json', 'a') as f:
        f.write(json.dumps(bspResults) + '\n')
