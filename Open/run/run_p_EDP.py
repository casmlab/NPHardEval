import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import *
from prompts import edpPrompts
from check.check_p_EDP import *
from tqdm import tqdm
import time
from utils import run_opensource_models

import json
import argparse

def load_data():
    data_path = DATA_PATH
    with open(data_path + "edp_instances.json", 'r') as f:
        all_data = json.load(f)
    return all_data

def runEDP(MODEL, q, p=edpPrompts):
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


def run_opensource_EDP(args, qs, p=edpPrompts):
    all_prompts = []
    for i, q in enumerate(tqdm(qs)):
        string_a = q['string_a']
        string_b = q['string_b']
        prompt_text = p['Intro'] + '\n' + \
                    p['Initial_question'].format(string_a=string_a, string_b=string_b) + '\n' + \
                    p['Output_content'] + '\n' + \
                    p['Output_format']
        prompt_text += 'Answer:\n'
        all_prompts.append(prompt_text)

    output = run_opensource_models(args, MODEL, all_prompts)
    return output


def construct_few_shot_examples(examples, p):
    few_shot_examples = '\n\nBelow are {} examples:\n\n'.format(len(examples))
    for i, example in enumerate(examples):
        string_a = example['question']['string_a']
        string_b = example['question']['string_b']
        question = p['Initial_question'].format(string_a=string_a, string_b=string_b)
        few_shot_examples += '<example{}>\nQuestion:\n'.format(i+1)+question + '\nAnswer:\n' + str(example['output']) + '\n</example{}>\n\n'.format(str(i+1))

    return few_shot_examples


def run_opensource_fewshot_EDP(args, qs, p=edpPrompts):
    FEWSHOT_DATA_PATH = '../Data/{}/few_shot/{}_few_shots.json'.format(args.prompt_question_type, args.prompt_question_type)
    with open(FEWSHOT_DATA_PATH, 'r') as f:
        fewshot_data = json.load(f)

    all_prompts = []
    for i, q in enumerate(tqdm(qs)):
        dif_level = (i//10) + args.difficulty_level
        if dif_level < 0:
            continue
        examples = [d for d in fewshot_data if d['complexity_level'] == dif_level+1][:args.example_number]
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

    output = run_opensource_models(args, MODEL, all_prompts)
    return output


if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description='Run EDP model script')

    # Add an argument for the model name
    parser.add_argument('model', type=str, help='The name of the model to run')
    parser.add_argument('--tuned_model_dir', type=str, default='')
    parser.add_argument('--prompt_question_type', type=str, default='EDP', help='BSP or EDP')
    parser.add_argument('--difficulty_level', type=int, default=0, help="-5, -4, -3, ...")
    parser.add_argument('--example_number', type=int, default=5, help="2,3,4,5")
    parser.add_argument('--data_dir', type=str, default='../Data/EDP/', help='../Data/finetune_data/test_1/EDP/')
    parser.add_argument('--fewshot', type=bool, default=False)

    # Parse the argument
    args = parser.parse_args()

    # Create the parser
    parser = argparse.ArgumentParser(description='Run EDP model script')

    # Script logic using args.model as the model name
    MODEL = str(args.model)

    DATA_PATH = args.data_dir
    if args.tuned_model_dir:
        RESULT_PATH = '../Results/finetuned/'
        if 'test_1' in args.data_dir:
            RESULT_PATH += 'test_1/'
        elif 'test_2' in args.data_dir:
            RESULT_PATH += 'test_2/'
        else:
            RESULT_PATH += 'original/'
    else:
        RESULT_PATH = '../Results/'
        if 'test_1' in args.data_dir:
            RESULT_PATH += 'test_1/'
        elif 'test_2' in args.data_dir:
            RESULT_PATH += 'test_2/'

    # load data
    edpData = load_data()
    #edpData = edpData[:2]
    edpResults = []
    print('number of datapoints: ', len(edpData))

    print("Using model: {}".format(MODEL))

    if args.fewshot:
        outputs = run_opensource_fewshot_EDP(args, edpData)
    else:
        outputs = run_opensource_EDP(args, edpData)
    for q, output in zip(edpData, outputs):
        output_dict = {}
        parsed_result, reasoning = parse_xml_to_dict(output)
        output_dict['output'] = parsed_result
        correctness = edp_check(q,parsed_result)
        output_dict['correctness'] = correctness
        edpResults.append(output_dict)
    # save the results
    if args.fewshot:
        if args.tuned_model_dir:
            number_of_benchmarks = args.tuned_model_dir.split('_')[-1]
            with open(RESULT_PATH+MODEL+'_'+'edpResults_{}_benchmarks{}.json'.format(args.difficulty_level, number_of_benchmarks), 'w') as f:
                f.write(json.dumps(edpResults) + '\n')
        else:
            with open(RESULT_PATH+MODEL+'_'+'edpResults_{}.json'.format(args.difficulty_level), 'w') as f:
                f.write(json.dumps(edpResults) + '\n')
    else:
        if args.tuned_model_dir:
            number_of_benchmarks = args.tuned_model_dir.split('_')[-1]
            with open(RESULT_PATH+MODEL+'_'+'edpResults_benchmarks{}.json'.format(number_of_benchmarks), 'w') as f:
                f.write(json.dumps(edpResults) + '\n')
        else:
            with open(RESULT_PATH+MODEL+'_'+'edpResults.json', 'w') as f:
                f.write(json.dumps(edpResults) + '\n')

# if __name__ == '__main__':
#     edpData = load_data()
#     edpResults = []

#     print("Using model: {}".format(MODEL))

#     MAX_TRY = 10
#     for q in edpData:
#         output_dict = {}
#         num_try = 0
#         while num_try < MAX_TRY:
#             try:
#                 llm_string = runEDP(q)
#                 output, reasoning = parse_xml_to_dict(llm_string)
#                 output_dict['output'] = output
#                 output_dict['correctness'] = edp_check(q, output)
#                 output_dict['reasoning'] = reasoning
#                 break
#             except Exception as e:
#                 print(f"Attempt {num_try + 1} failed: {e}")
#                 num_try += 1
#         if output_dict:
#             edpResults.append(output_dict)
#         else:
#             print(f"Failed to run {q}")
#             edpResults.append({'output': '', 'correctness': False})

#     # Save the results
#     with open(RESULT_PATH + MODEL + '_' + 'edpResults.json', 'a') as f:
#         f.write(json.dumps(edpResults) + '\n')