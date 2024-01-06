import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append('../..')

from models import *
from prompts import bspPrompts
from check.check_p_BSP import *
import time

import pandas as pd
import numpy as np
import json
import argparse
from tqdm import tqdm

def load_data():
    data_path = DATA_PATH
    with open(data_path + "bsp_instances.json", 'r') as f:
        all_data = json.load(f)
    return all_data


def construct_few_shot_examples(examples):
    few_shot_examples = '\n\nBelow are 5 examples including pairs of questions and answers:\n\n'
    for i, example in enumerate(examples):
        question = 'The sorted array elements are: ' + ', '.join(str(a) for a in example['question']['array'])
        few_shot_examples += '<example{}>\nQuestion:\n'.format(i+1)+question + '\nAnswer:\n' + str(example['output']) + '\n</example{}>\n\n'.format(str(i+1))

    return few_shot_examples


def run_opensource_fewshot_BSP(args, qs, p=bspPrompts):
    FEWSHOT_DATA_PATH = '../../Data/Fewshot/FewshotExample/{}_few_shots.json'.format(args.prompt_question_type)
    with open(FEWSHOT_DATA_PATH, 'r') as f:
        fewshot_data = json.load(f)

    all_prompts = []
    for i, q in enumerate(tqdm(qs)):
        dif_level = (i//10) + args.difficulty_level
        if dif_level < 0:
            continue
        examples = [d for d in fewshot_data if d['complexity_level'] == dif_level+1][:5]
        few_shot_examples = construct_few_shot_examples(examples)
        target_value = q['target']
        # TO-DO: fix data not being sorted
        array = sorted(q['array'])
        prompt_text = p['Intro'] + '\n' + \
                  p['Initial_question'].format(target_value=target_value) + '\n' + \
                  p['Output_content'] + '\n' + \
                  p['Output_format'] + \
                  few_shot_examples + \
                  'Again, ' + p['Initial_question'].format(target_value=target_value) + '\n' + \
                  'Follow the format in the above examples to write your answer.\n' +\
                  '\nThis is the new question you need to solve:\n\nQuestion:\nThe sorted array elements are: ' + ', '.join(map(str, array)) + '\nAnswer:\n'
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
    parser = argparse.ArgumentParser(description='Run BSP model script')

    # Add an argument for the model name
    parser.add_argument('model', type=str, help='The name of the model to run')
    parser.add_argument('--prompt_question_type', type=str, default='BSP', help='BSP or MFP')
    parser.add_argument('--difficulty_level', type=int, default=0, help="-5, -4, -3, ...")

    # Parse the argument
    args = parser.parse_args()

    # Script logic using args.model as the model name
    MODEL = str(args.model)

    DATA_PATH = '../../Data/BSP/'
    RESULT_PATH = '../../Results/'

    # load data
    bspData = load_data()
    #bspData = bspData[:20]
    bspResults = []
    print('number of datapoints: ', len(bspData))

    print("Using model: {}".format(MODEL))

    outputs = run_opensource_fewshot_BSP(args, bspData)
    for q, output in zip(bspData, outputs):
        output_dict = {}
        output_dict['output'] = output
        correctness = bsp_check(q,output)
        output_dict['correctness'] = correctness
        bspResults.append(output_dict)
    # save the results
    with open(RESULT_PATH+MODEL+'_'+'bspResults_{}_{}.json'.format(args.prompt_question_type, args.difficulty_level), 'w') as f:
        f.write(json.dumps(bspResults) + '\n')
