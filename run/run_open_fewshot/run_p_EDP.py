import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append('../..')

from models import *
from prompts import edpPrompts
from check.check_p_EDP import *
from tqdm import tqdm
import time

import json
import argparse

def load_data():
    data_path = DATA_PATH
    with open(data_path + "edp_instances.json", 'r') as f:
        all_data = json.load(f)
    return all_data

def construct_few_shot_examples(examples, p):
    few_shot_examples = '\n\nBelow are 3 examples:\n\n'
    for i, example in enumerate(examples):
        string_a = example['question']['string_a']
        string_b = example['question']['string_b']
        question = p['Initial_question'].format(string_a=string_a, string_b=string_b)
        few_shot_examples += '<example{}>\nQuestion:\n'.format(i+1)+question + '\nAnswer:\n' + str(example['output']) + '\n</example{}>\n\n'.format(str(i+1))

    return few_shot_examples


def run_opensource_fewshot_EDP(args, qs, p=edpPrompts):
    FEWSHOT_DATA_PATH = '../../Data/Fewshot/FewshotExample/{}_few_shots.json'.format(args.prompt_question_type)
    with open(FEWSHOT_DATA_PATH, 'r') as f:
        fewshot_data = json.load(f)

    all_prompts = []
    for i, q in enumerate(tqdm(qs)):
        dif_level = (i//10) + args.difficulty_level
        if dif_level < 0:
            continue
        examples = [d for d in fewshot_data if d['complexity_level'] == dif_level+1][:3]
        few_shot_examples = construct_few_shot_examples(examples, p)
        string_a = q['string_a']
        string_b = q['string_b']
        prompt_text = p['Intro'] + '\n' + \
                  p['Initial_question'].format(string_a=string_a, string_b=string_b) + '\n' + \
                  p['Output_content'] + '\n' + \
                  p['Output_format'] + \
                  few_shot_examples + \
                  'Here is the question you need to solve:\n' + p['Initial_question'].format(string_a=string_a, string_b=string_b) + '\n' + \
                  'Follow the format in the above examples to write your answer.\n' +\
                  '\nAnswer:\n'
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
    parser = argparse.ArgumentParser(description='Run EDP model script')

    # Add an argument for the model name
    parser.add_argument('model', type=str, help='The name of the model to run')
    parser.add_argument('--prompt_question_type', type=str, default='EDP', help='BSP or EDP')
    parser.add_argument('--difficulty_level', type=int, default=0, help="-5, -4, -3, ...")

    # Parse the argument
    args = parser.parse_args()

    # Create the parser
    parser = argparse.ArgumentParser(description='Run EDP model script')

    # Script logic using args.model as the model name
    MODEL = str(args.model)

    DATA_PATH = '../../Data/EDP/'
    RESULT_PATH = '../../Results/'

    # load data
    edpData = load_data()
    #edpData = edpData[:2]
    edpResults = []
    print('number of datapoints: ', len(edpData))

    print("Using model: {}".format(MODEL))

    outputs = run_opensource_fewshot_EDP(args, edpData)
    for q, output in zip(edpData, outputs):
        output_dict = {}
        parsed_result, reasoning = parse_xml_to_dict(output)
        output_dict['output'] = parsed_result
        correctness = edp_check(q,parsed_result)
        output_dict['correctness'] = correctness
        edpResults.append(output_dict)
    with open(RESULT_PATH+MODEL+'_'+'edpResults_{}_{}.json'.format(args.prompt_question_type, args.difficulty_level), 'w') as f:
        f.write(json.dumps(edpResults) + '\n')
