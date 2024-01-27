import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import *
from prompts import edpPrompts, bspPrompts
from check.check_p_EDP import *
from utils import parse_xml_to_dict

import pandas as pd
import numpy as np
import json
import argparse
import ast
import random

# Create the parser
parser = argparse.ArgumentParser(description='Run EDP model script')

# Add an argument for the model name
parser.add_argument('model', type=str, help='The name of the model to run')
parser.add_argument('prompt_style', type=str, help='The name of the prompt style to run')
# parser.add_argument('difficulty_level', type=int, help='-5, -4, -3, ...')

# Parse the argument
args = parser.parse_args()

# Script logic using args.model as the model name
MODEL = str(args.model)
PROMPT_STYLE = str(args.prompt_style)
# DIFFICULTY_LEVEL = int(args.difficulty_level)

DATA_PATH = '../Data/EDP/'
RESULT_PATH = '../Results_fewshot/'
EXAMPLE_PATH = DATA_PATH

def load_data():
    data_path = DATA_PATH
    with open(data_path + "edp_instances.json", 'r') as f:
        all_data = json.load(f)
    return all_data

def construct_few_shot_examples(examples, PROMPT_STYLE, p=edpPrompts):
    few_shot_examples = '\n\nBelow are 5 examples:\n'
    for i, example in enumerate(examples):
        if PROMPT_STYLE == 'self':
            string_a = example['question']['string_a']
            string_b = example['question']['string_b']
            question = p['Initial_question'].format(string_a=string_a, string_b=string_b)
            # print(question)
            # few_shot_examples += '<example{}>\nQuestion:\n'.format(i+1)+question + '\nAnswer:\n' + str(example['output']) + '\n</example{}>\n\n'.format(str(i+1))
        # elif PROMPT_STYLE == 'other':
        #     input_dict = ast.literal_eval(str(example['question']))
        #     source_node = input_dict['source']
        #     sink_node = input_dict['sink']

        #     edges = input_dict['edges']
        #     question = example['Initial_question'].format(source_node=source_node, sink_node=sink_node) + "\n The capacities of the network's edges are as follows: \n"
        #     for edge in edges:
        #         this_line = f"Edge from {edge['from']} to {edge['to']} has a capacity of {edge['capacity']}."
        #         question += this_line + '\n'
        few_shot_examples += '<example{}>\nQuestion:\n'.format(i+1)+question + '\nAnswer:\n' + str(example['output']) + '\n</example{}>\n\n'.format(str(i+1))

    return few_shot_examples

def runEDP(q, eg, p=edpPrompts):
    string_a = q['string_a']
    string_b = q['string_b']

    # prompt_text = p['Intro'] + '\n' + \
    #               p['Initial_question'].format(target_value=target_value) + '\n' + \
    #               eg + '\n' + \
    #               'Again, '+ p['Initial_question'].format(target_value=target_value) + '\n' + \
    #               p['Output_format'] + '\n' + \
    #               '\n For the question you need to solve, the sorted array elements are: ' + ', '.join(map(str, array)) + '\n'

    prompt_text = p['Intro'] + '\n' + \
            p['Initial_question'].format(string_a=string_a, string_b=string_b) + '\n' + \
            p['Output_content'] + '\n' + \
            p['Output_format'] + \
            eg + \
            'Here is the question you need to solve:\n' + p['Initial_question'].format(string_a=string_a, string_b=string_b) + '\n' + \
            'Follow the format in the above examples to write your answer.\n' +\
            '\nAnswer:\n'

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

    print(edpData[0])

    print("Using model: {}".format(MODEL))

    MAX_TRY = 10

    if PROMPT_STYLE == 'self':
        with open(EXAMPLE_PATH+'EDP_few_shots.json', 'r') as f:
            fewshot_data = json.load(f)
    elif PROMPT_STYLE == 'other':
        with open(EXAMPLE_PATH+'BSP_few_shots_other.json', 'r') as f:
            fewshot_data = json.load(f)
    else:
        print('Prompt style not found')
        exit(1)

    for DIFFICULTY_LEVEL in range(-5, 6):
        edpResults = []
        print("**********")
        print(f"Difficulty level: {DIFFICULTY_LEVEL}")
        print("**********")
        for i, q in enumerate((edpData)):
            # get examples
            dif_level = (i//10) + DIFFICULTY_LEVEL
            if dif_level < 0:
                continue
            examples = [d for d in fewshot_data if d['complexity_level'] == dif_level+1][:5]
            few_shot_examples = construct_few_shot_examples(examples, PROMPT_STYLE)

            # save result
            output_dict = {}
            num_try = 0
            while num_try < MAX_TRY:
                try:
                    llm_string = runEDP(q, few_shot_examples)
                    output, reasoning = parse_xml_to_dict(llm_string)
                    output_dict['output'] = output
                    output_dict['correctness'] = edp_check(q, output)
                    output_dict['reasoning'] = reasoning
                    break
                except Exception as e:
                    print(f"Attempt {num_try + 1} failed: {e}")
                    num_try += 1
            if output_dict:
                llm_string = runEDP(q, few_shot_examples)
                edpResults.append(output_dict)
            else:
                print(f"Failed to run {q}")
                edpResults.append({'output': '', 'correctness': False})

        # Save the results
        def set_default(obj):
            if isinstance(obj, set):
                return list(obj)
            raise TypeError

        with open(RESULT_PATH + MODEL + '_' + 'edpResults_{}_{}.json'.format(PROMPT_STYLE,DIFFICULTY_LEVEL), 'a') as f:
            f.write(json.dumps(edpResults, default=set_default) + '\n')
