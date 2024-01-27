import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import *
from prompts import mfpPrompts
from check.check_p_MFP import *
#from utils import parse_xml_to_dict
import ast
import pandas as pd
import numpy as np
import json
import argparse
from tqdm import tqdm
import time
import random

def load_data():
    data_path = DATA_PATH
    with open(data_path + "mfp_instances.json", 'r') as f:
        all_data = json.load(f)
    return all_data

def runMFP(MODEL,q, p=mfpPrompts):
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

def construct_few_shot_examples(examples, p):
    few_shot_examples = '\n\nBelow are 2 examples:\n\n'
    for i, example in enumerate(examples):
        input_dict = ast.literal_eval(str(example['question']))
        source_node = input_dict['source']
        sink_node = input_dict['sink']

        edges = input_dict['edges']
        question = p['Initial_question'].format(source_node=source_node, sink_node=sink_node) + "\n The capacities of the network's edges are as follows: \n"
        for edge in edges:
            this_line = f"Edge from {edge['from']} to {edge['to']} has a capacity of {edge['capacity']}."
            question += this_line + '\n'
        few_shot_examples += '<example{}>\nQuestion:\n'.format(i+1)+question + '\nAnswer:\n' + str(example['output']) + '\n</example{}>\n\n'.format(str(i+1))

    return few_shot_examples


def run_opensource_few_shot_MFP(args, qs, p=mfpPrompts):
    FEWSHOT_DATA_PATH = '../Data/{}/few_shot/{}_few_shots.json'.format(args.prompt_question_type, args.prompt_question_type)
    with open(FEWSHOT_DATA_PATH, 'r') as f:
        fewshot_data = json.load(f)

    all_prompts = []
    for i, q in enumerate(tqdm(qs)):
        dif_level = (i//10) + args.difficulty_level
        if dif_level < 0:
            continue
        with_reasoning = [d for d in fewshot_data if d['complexity_level'] == dif_level+1 and '<reasoning>' in d['output']]
        without_reasoning = [d for d in fewshot_data if d['complexity_level'] == dif_level+1 and '<reasoning>' not in d['output']]
        examples = with_reasoning[:2]
        if len(examples) < 2:
            examples += without_reasoning[:2-len(examples)]
        random.shuffle(examples)
        few_shot_examples = construct_few_shot_examples(examples, p)

        source_node = q['source']
        sink_node = q['sink']
        edges = q['edges']
        prompt_text = p['Intro'] + '\n' + \
                    p['Output_content'] + '\n' + \
                    p['Output_format'] + \
                    few_shot_examples + \
                    'Again, ' + p['Initial_question'].format(source_node=source_node, sink_node=sink_node) + ' '+\
                    p['Output_content'] + '\n' + \
                    'Follow the format in the above examples to write your answer.\n' +\
                    "\nThis is the new question you need to solve:\n\nQuestion:\n" +\
                    p['Initial_question'].format(source_node=source_node, sink_node=sink_node) +" The capacities of the network's edges are as follows: \n"
        for edge in edges:
            this_line = f"Edge from {edge['from']} to {edge['to']} has a capacity of {edge['capacity']}."
            prompt_text += this_line + '\n'
        prompt_text += '\nAnswer:\n'
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

def run_opensource_MFP(args, qs, p=mfpPrompts):
    all_prompts = []
    for i, q in enumerate(tqdm(qs)):
        source_node = q['source']
        sink_node = q['sink']
        edges = q['edges']
        prompt_text = p['Intro'] + '\n' + \
                  p['Initial_question'].format(source_node=source_node, sink_node=sink_node) + '\n' + \
                  p['Output_content'] + '\n' + \
                  p['Output_format'] + '\n\n'+\
                  "Here is a network description. The capacities of the network's edges are as follows: \n"
        for edge in edges:
            this_line = f"Edge from {edge['from']} to {edge['to']} has a capacity of {edge['capacity']}."
            prompt_text += this_line + '\n'
        prompt_text += 'Answer:\n'
        all_prompts.append(prompt_text)

    if MODEL.startswith('mistral'):
        output = run_mistral(all_prompts)
    elif MODEL.startswith('mixtral'):
        output = run_mixtral(all_prompts)
    elif MODEL.startswith('yi'):
        output = run_yi(all_prompts)
    elif MODEL.startswith('mpt'):
        output = run_mpt(all_prompts)
    elif MODEL.startswith('phi-2'):
        output = run_phi_2(all_prompts)
    elif MODEL.startswith('phi'):
        output = run_phi(all_prompts)
    elif MODEL.startswith('vicuna'):
        output = run_vicuna(all_prompts)
    elif MODEL.startswith('qwen'):
        output = run_qwen(all_prompts)
    else:
        raise NotImplementedError
    return output


if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description='Run MFP model script')

    # Add an argument for the model name
    parser.add_argument('model', type=str, help='The name of the model to run')
    parser.add_argument('--prompt_question_type', type=str, default='MFP', help='BSP or MFP')
    parser.add_argument('--difficulty_level', type=int, default=0, help="-5, -4, -3, ...")

    # Parse the argument
    args = parser.parse_args()

    # Script logic using args.model as the model name
    MODEL = str(args.model)

    DATA_PATH = '../Data/MFP/'
    RESULT_PATH = '../Results/'


    # load data
    mfpData = load_data()
    #mfpData = mfpData[:2]
    mfpResults = []
    print('number of datapoints: ', len(mfpData))

    print("Using model: {}".format(MODEL))

    outputs = run_opensource_MFP(args, mfpData)
    for q, output in zip(mfpData, outputs):
        output_dict = {}
        output_dict['output'] = output
        correctness = mfp_check(q,output)
        output_dict['correctness'] = correctness
        mfpResults.append(output_dict)
    # save the results
    with open(RESULT_PATH + MODEL + '_' + 'mfpResults.json', 'w') as f:
        f.write(json.dumps(mfpResults) + '\n')
    # with open(RESULT_PATH+MODEL+'_'+'mfpResults_{}_{}.json'.format(args.prompt_question_type, args.difficulty_level), 'w') as f:
    #     f.write(json.dumps(mfpResults) + '\n')


# if __name__ == '__main__':
#     mfpData = load_data()
#     mfpResults = []

#     print("Using model: {}".format(MODEL))

#     MAX_TRY = 10
#     for q in mfpData:
#         output_dict = {}
#         num_try = 0
#         while num_try < MAX_TRY:
#             try:
#                 llm_string = runMFP(q)
#                 output = parse_xml_to_dict(llm_string)
#                 output_dict['output'] = output
#                 output_dict['correctness'] = mfp_check(q, output)
#                 break
#             except Exception as e:
#                 print(f"Attempt {num_try + 1} failed: {e}")
#                 num_try += 1
#         if output_dict:
#             mfpResults.append(output_dict)
#         else:
#             print(f"Failed to run {q}")
#             mfpResults.append({'output': '', 'correctness': False})

#     # Save the results
#     with open(RESULT_PATH + MODEL + '_' + 'mfpResults.json', 'a') as f:
#         f.write(json.dumps(mfpResults) + '\n')
