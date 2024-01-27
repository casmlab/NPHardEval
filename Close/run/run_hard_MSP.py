import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import *
from prompts import mspPrompts
from check.check_hard_MSP import *

import pandas as pd
import numpy as np
import json

import argparse

# Create the parser
parser = argparse.ArgumentParser(description='Run model script')

# Add an argument for the model name
parser.add_argument('model', type=str, help='The name of the model to run')

# Parse the argument
args = parser.parse_args()

# Your script's logic here, using args.model as the model name
MODEL = str(args.model)

# MODEL = 'gpt-4-1106-preview'
# # models: gpt-4-1106-preview, gpt-3.5-turbo-1106, claude-2, claude-instant, palm-2

DATA_PATH = '../Data/MSP/'
RESULT_PATH = '../Results/'

def load_data():
    data_path = DATA_PATH
    with open(data_path+"msp_instances.json", 'r') as f:
        all_data = json.load(f)
    return all_data

def runMSP(q, p=mspPrompts): # q is the data for the HP-hard question, p is the prompt
    total_participants = q['participants']
    total_timeslots = q['time_slots']
    prompt_text = p['Intro'] + '\n' \
        + p['Initial_question'].format(total_participants=total_participants,total_timeslots=total_timeslots) + '\n' \
        + p['Output_content'] + '\n' \
        + p['Output_format'] + \
        '\n The meetings and participants details are as below: \n'
    meetings = q['meetings']
    participants = q['participants']
    for meeting in meetings:
        this_line = "Meeting {} is with duration {}.".format(meeting['id'],meeting['duration'])
        prompt_text += this_line + '\n'
    for j in participants.keys():
        this_line = "Participant {} is available at time slots {} and has meetings {}.".format(j,participants[j]['available_slots'],participants[j]['meetings'])
        prompt_text += this_line + '\n'
    # print(prompt_text)

    # get output
    if 'gpt' in MODEL:
        output = run_gpt(prompt_text,model = MODEL)
    elif 'claude' in MODEL:
        output = run_claude(text_prompt=prompt_text,model = MODEL)
    else:
        # raise error
        print('Model not found')
    return output

if __name__ == '__main__':
    mspData = load_data()
    mspResults = []

    MAX_TRY = 10 # updated MAX_TRY
    for q in mspData:
        # print("_________________________________________________________")
        # print(q)
        output_dict = {}
        num_try = 0
        while num_try < MAX_TRY:
            try:
                output = runMSP(q)
                # print(output)
                output_dict['output'] = output
                output_dict['correctness'] = mspCheck(q, output)
                break
            except Exception as e:
                print(f"Attempt {num_try+1} failed: {e}")
                num_try += 1
        if output_dict:
            mspResults.append(output_dict)
        else:
            print(f"Failed to run {q}")
            mspResults.append({'output': '', 'correctness': False})

    # save the results
    with open(RESULT_PATH+MODEL+'_'+'mspResults.json', 'a') as f:
        f.write(json.dumps(mspResults) + '\n')