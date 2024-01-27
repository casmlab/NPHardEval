import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import *
from prompts import sppPrompts
from check.check_p_SPP import *
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import argparse
from utils import run_opensource_models

def load_data():
    data_path = DATA_PATH
    with open(data_path + "spp_instances.json", 'r') as f:
        all_data = json.load(f)
    return all_data

def runSPP(MODEL,q, p=sppPrompts):
    # start_node = q['start_node']
    # end_node = q['end_node']
    # TO-DO: fix later
    start_node = q['nodes'][0]
    end_node = q['nodes'][-1]
    edges = q['edges']
    prompt_text = p['Intro'] + '\n' + \
                  p['Initial_question'].format(start_node=start_node, end_node=end_node) + '\n' + \
                  p['Output_content'] + '\n' + \
                  p['Output_format'] + \
                  "\n The graph's edges and weights are as follows: \n"
    for edge in edges:
        this_line = f"Edge from {edge['from']} to {edge['to']} has a weight of {edge['weight']}."
        prompt_text += this_line + '\n'

    if 'gpt' in MODEL:
        output = run_gpt(prompt_text, model=MODEL)
    elif 'claude' in MODEL:
        output = run_claude(prompt_text, model=MODEL)
    else:
        print('Model not found')
        return None

    return output


def run_opensource_SPP(qs, p=sppPrompts):
    all_prompts = []
    for q in tqdm(qs):
        start_node = q['nodes'][0]
        end_node = q['nodes'][-1]
        edges = q['edges']
        prompt_text = p['Intro'] + '\n' + \
                    p['Initial_question'].format(start_node=start_node, end_node=end_node) + '\n' + \
                    p['Output_content'] + '\n' + \
                    p['Output_format'] + \
                    "\n The graph's edges and weights are as follows: \n"
        for edge in edges:
            this_line = f"Edge from {edge['from']} to {edge['to']} has a weight of {edge['weight']}."
            prompt_text += this_line + '\n'
        prompt_text += 'Answer:\n'
        all_prompts.append(prompt_text)

    output = run_opensource_models(args, MODEL, all_prompts)
    return output


if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description='Run SPP model script')

    # Add an argument for the model name
    parser.add_argument('model', type=str, help='The name of the model to run')
    parser.add_argument('--tuned_model_dir', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='../Data/SPP/', help='../Data/finetune_data/test_1/SPP/')

    # Parse the argument
    args = parser.parse_args()

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
    sppData = load_data()
    #sppData = sppData[:10]
    sppResults = []
    print('number of datapoints: ', len(sppData))

    print("Using model: {}".format(MODEL))

    outputs = run_opensource_SPP(sppData)
    for q, output in zip(sppData, outputs):
        output_dict = {}
        output, reasoning = parse_xml_to_dict(output)
        output_dict['output'] = output
        output_dict['correctness'] = spp_check(q, output)
        output_dict['reasoning'] = reasoning
        sppResults.append(output_dict)
    # save the results
    if args.tuned_model_dir:
        number_of_benchmarks = args.tuned_model_dir.split('_')[-1]
        with open(RESULT_PATH+MODEL+'_'+'sppResults_benchmarks{}.json'.format(number_of_benchmarks), 'w') as f:
            f.write(json.dumps(sppResults) + '\n')
    else:
        with open(RESULT_PATH+MODEL+'_'+'sppResults.json', 'w') as f:
            f.write(json.dumps(sppResults) + '\n')



# if __name__ == '__main__':
#     sppData = load_data()
#     sppResults = []

#     print("Using model: {}".format(MODEL))

#     MAX_TRY = 10
#     for q in sppData:
#         output_dict = {}
#         num_try = 0
#         while num_try < MAX_TRY:
#             try:
#                 llm_string = runSPP(q)
#                 output = parse_xml_to_dict(llm_string)
#                 output_dict['output'] = output
#                 output_dict['correctness'] = spp_check(q, output)
#                 break
#             except Exception as e:
#                 print(f"Attempt {num_try + 1} failed: {e}")
#                 num_try += 1
#         if output_dict:
#             sppResults.append(output_dict)
#         else:
#             print(f"Failed to run {q}")
#             sppResults.append({'output': '', 'correctness': False})

#     # Save the results
#     with open(RESULT_PATH + MODEL + '_' + 'sppResults.json', 'a') as f:
#         f.write(json.dumps(sppResults) + '\n')
