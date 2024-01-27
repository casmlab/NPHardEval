import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import *
from prompts import kspPrompts
from check.check_cmp_KSP import *
from utils import parse_xml_to_dict

import pandas as pd
import numpy as np
import json
import argparse

# Create the parser
parser = argparse.ArgumentParser(description='Run KSP model script')

# Add an argument for the model name
parser.add_argument('model', type=str, help='The name of the model to run')

# Parse the argument
args = parser.parse_args()

# Script logic using args.model as the model name
MODEL = str(args.model)

DATA_PATH = '../Data/KSP/'
RESULT_PATH = '../Results/'

def load_data():
    data_path = DATA_PATH
    with open(data_path + "ksp_instances.json", 'r') as f:
        all_data = json.load(f)
    return all_data

def runKSP(q, p=kspPrompts):
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

if __name__ == '__main__':
    kspData = load_data()
    kspResults = []

    print("Using model: {}".format(MODEL))

    MAX_TRY = 1
    for q in kspData:
        output_dict = {}
        num_try = 0
        while num_try < MAX_TRY:
            try:
                llm_string = runKSP(q)
                output, reasoning = parse_xml_to_dict(llm_string)
                output_dict['output'] = output
                output_dict['correctness'] = kspCheck(q, output)
                output_dict['reasoning'] = reasoning
                break
            except Exception as e:
                print(f"Attempt {num_try + 1} failed: {e}")
                num_try += 1
        if output_dict:
            kspResults.append(output_dict)
        else:
            print(f"Failed to run {q}")
            kspResults.append({'output': '', 'correctness': False})

    # Save the results
    with open(RESULT_PATH + MODEL + '_' + 'kspResults.json', 'a') as f:
        f.write(json.dumps(kspResults) + '\n')
