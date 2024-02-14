import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import *
from prompts import mfpPrompts, bspPrompts
from check.check_p_MFP import *
from utils import parse_xml_to_dict
from utils import find_data_path


import pandas as pd
import numpy as np
import json
import argparse
import ast
import random

# Create the parser
parser = argparse.ArgumentParser(description='Run MFP model script')

# Add an argument for the model name
parser.add_argument('model', type=str, help='The name of the model to run')
parser.add_argument('prompt_style', type=str, help='The name of the prompt style to run')
# parser.add_argument('difficulty_level', type=int, help='-5, -4, -3, ...')

# Parse the argument
args = parser.parse_args()

# Script logic using args.model as the model name
MODEL = str(args.model)
PROMPT_STYLE = str(args.prompt_style)

DATA_PATH = '../Data/MFP/'
RESULT_PATH = '../Results_fewshot/'
EXAMPLE_PATH = DATA_PATH

if not os.path.exists(DATA_PATH) or not os.path.exists(RESULT_PATH):
    DATA_PATH,RESULT_PATH = find_data_path(os.path.abspath(__file__))
    EXAMPLE_PATH = DATA_PATH

# DATA_PATH = '../Data/New_replace/'
# RESULT_PATH = '../Results_fewshot/MFP_new/'
# EXAMPLE_PATH = '../fewshot_eg/'


def load_data():
    data_path = DATA_PATH
    with open(data_path + "mfp_instances.json", 'r') as f:
        all_data = json.load(f)
    return all_data

def construct_few_shot_examples(examples, PROMPT_STYLE, p=mfpPrompts):
    few_shot_examples = '\n\nBelow are 5 examples:\n'
    for i, example in enumerate(examples):
        if PROMPT_STYLE == 'self':
            input_dict = ast.literal_eval(str(example['question']))
            source_node = input_dict['source']
            sink_node = input_dict['sink']

            edges = input_dict['edges']
            question = p['Initial_question'].format(source_node=source_node, sink_node=sink_node) + "\n The capacities of the network's edges are as follows: \n"
            for edge in edges:
                this_line = f"Edge from {edge['from']} to {edge['to']} has a capacity of {edge['capacity']}."
                question += this_line + '\n'
        elif PROMPT_STYLE == 'other':
            question = 'The sorted array elements are: ' + ', '.join(str(a) for a in example['question']['array'])
        few_shot_examples += '<example{}>\nQuestion:\n'.format(i+1)+question + '\nAnswer:\n' + str(example['output']) + '\n</example{}>\n\n'.format(str(i+1))

    return few_shot_examples


def runMFP(q, eg, p=mfpPrompts):
    source_node = q['source']
    sink_node = q['sink']
    edges = q['edges']

    # prompt_text = p['Intro'] + '\n' + \
    #               p['Initial_question'].format(source_node=source_node, sink_node=sink_node) + '\n' + \
    #               p['Output_content'] + '\n' + \
    #               p['Output_format'] + '\n' + \
    #               eg + '\n' + \
    #               'Again, ' + p['Initial_question'].format(source_node=source_node, sink_node=sink_node) + '\n' + \
    #               'Follow the format in the above examples to write your answer.\n' +\
    #               '\nThis is the new question you need to solve:\n\nQuestion: the capacities of the network\'s edges are as follows: \n'
    # for edge in edges:
    #     this_line = f"Edge from {edge['from']} to {edge['to']} has a capacity of {edge['capacity']}."
    #     prompt_text += this_line + '\n'

    # prompt_text = p['Intro'] + '\n' + \
    #               p['Initial_question'].format(source_node=source_node, sink_node=sink_node) + '\n' + \
    #               p['Output_content'] + '\n' + \
    #               p['Output_format'] + '\n' + \
    #               'Please learn from the example after the dash line (-----) at the end\n' + \
    #               'For the problem you need to solve, here is the data -- The capacities of the network\'s edges are as follows: \n'
    # for edge in edges:
    #     this_line = f"Edge from {edge['from']} to {edge['to']} has a capacity of {edge['capacity']}."
    #     prompt_text += this_line + '\n'
    # prompt_text += '----- Below this line are the examples for your reference\n'
    # prompt_text += eg + '\n'
    # prompt_text += 'Again, please follow the output format:\n' + p['Output_format']

    # prompt_text = p['Intro'] + '\n' + \
    #             p['Output_content'] + '\n' + \
    #             p['Output_format'] + \
    #             few_shot_examples + \
    #             'Again, ' + p['Initial_question'].format(source_node=source_node, sink_node=sink_node) +\
    #             p['Output_content'] + '\n' + \
    #             'Follow the format in the above examples to write your answer.\n' +\
    #             "\nThis is the new question you need to solve:\n\nQuestion:\n" +\
    #             "The capacities of the network's edges are as follows: \n"
    # for edge in edges:
    #     this_line = f"Edge from {edge['from']} to {edge['to']} has a capacity of {edge['capacity']}."
    #     prompt_text += this_line + '\n'
    # prompt_text += '\nAnswer:\n'

    prompt_text = p['Intro'] + '\n' + \
                p['Output_content'] + '\n' + \
                p['Output_format'] + \
                few_shot_examples + \
                'Again, ' + p['Initial_question'].format(source_node=source_node, sink_node=sink_node) +\
                p['Output_content'] + '\n' + \
                'Follow the format in the above examples to write your answer.\n' +\
                "\nThis is the new question you need to solve:\n\nQuestion:\n" +\
                "The capacities of the network's edges are as follows: \n"
    for edge in edges:
        this_line = f"Edge from {edge['from']} to {edge['to']} has a capacity of {edge['capacity']}."
        prompt_text += this_line + '\n'
    prompt_text += '\nAnswer:\n'

    # print(prompt_text)

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

    print("Using model: {}".format(MODEL))

    MAX_TRY = 10

    if PROMPT_STYLE == 'self':
        with open(EXAMPLE_PATH+'MFP_few_shots.json', 'r') as f:
            fewshot_data = json.load(f)
    elif PROMPT_STYLE == 'other':
        with open(EXAMPLE_PATH+'BSP_few_shots_other.json', 'r') as f:
            fewshot_data = json.load(f)
    else:
        print('Prompt style not found')
        exit(1)

    for DIFFICULTY_LEVEL in range(-4, 6):
        mfpResults = []
        print("**********")
        print(f"Difficulty level: {DIFFICULTY_LEVEL}")
        print("**********")
        for i, q in enumerate(mfpData):
            # get examples
            dif_level = (i//10) + DIFFICULTY_LEVEL
            if dif_level < 0:
                continue
            with_reasoning = [d for d in fewshot_data if d['complexity_level'] == dif_level+1 and '<reasoning>' in d['output']]
            without_reasoning = [d for d in fewshot_data if d['complexity_level'] == dif_level+1 and '<reasoning>' not in d['output']]
            examples = with_reasoning[:5]
            if len(examples) < 5:
                examples += without_reasoning[:5-len(examples)]
            random.shuffle(examples)
            few_shot_examples = construct_few_shot_examples(examples, PROMPT_STYLE, )

            # save result
            output_dict = {}
            num_try = 0
            while num_try < MAX_TRY:
                try:
                    llm_string = runMFP(q, few_shot_examples)
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

        def set_default(obj):
            if isinstance(obj, set):
                return list(obj)
            raise TypeError
        
        # Save the results
        with open(RESULT_PATH + MODEL + '_' + 'mfpResults_few_{}_{}.json'.format(PROMPT_STYLE,DIFFICULTY_LEVEL), 'a') as f:
            f.write(json.dumps(mfpResults, default=set_default) + '\n')
        

        # result_file = RESULT_PATH + MODEL + '_' + 'mfpResults_few_{}_{}.json'.format(PROMPT_STYLE,DIFFICULTY_LEVEL)
        # with open(result_file, 'a') as f:
        #     json.dump(mfpResults, f)
        #     f.write('\n')
