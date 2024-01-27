import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import *
from prompts import edpPrompts
from check.check_p_EDP import *
from utils import parse_xml_to_dict

import json
import argparse

# Create the parser
parser = argparse.ArgumentParser(description='Run EDP model script')

# Add an argument for the model name
parser.add_argument('model', type=str, help='The name of the model to run')

# Parse the argument
args = parser.parse_args()

# Script logic using args.model as the model name
MODEL = str(args.model)

DATA_PATH = '../Data/EDP/'
RESULT_PATH = '../Results/'

def load_data():
    data_path = DATA_PATH
    with open(data_path + "edp_instances.json", 'r') as f:
        all_data = json.load(f)
    return all_data

def runEDP(q, p=edpPrompts):
    string_a = q['string_a']
    string_b = q['string_b']
    prompt_text = p['Intro'] + '\n' + \
                  p['Initial_question'].format(string_a=string_a, string_b=string_b) + '\n' + \
                  p['Output_content'] + '\n' + \
                  p['Output_format']

    if 'gpt' in MODEL:
        output = run_gpt(prompt_text, model=MODEL)
    elif 'claude' in MODEL:
        output = run_claude(prompt_text, model=MODEL)
    else:
        print('Model not found')
        return None

    return output

if __name__ == '__main__':
    edpData = load_data()
    edpResults = []

    print("Using model: {}".format(MODEL))

    MAX_TRY = 10
    for q in edpData:
        output_dict = {}
        num_try = 0
        while num_try < MAX_TRY:
            try:
                llm_string = runEDP(q)
                output, reasoning = parse_xml_to_dict(llm_string)
                output_dict['output'] = output
                output_dict['correctness'] = edp_check(q, output)
                output_dict['reasoning'] = reasoning
                break
            except Exception as e:
                print(f"Attempt {num_try + 1} failed: {e}")
                num_try += 1
        if output_dict:
            edpResults.append(output_dict)
        else:
            print(f"Failed to run {q}")
            edpResults.append({'output': '', 'correctness': False})

    # Save the results
    with open(RESULT_PATH + MODEL + '_' + 'edpResults.json', 'a') as f:
        f.write(json.dumps(edpResults) + '\n')
