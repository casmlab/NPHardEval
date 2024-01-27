import json
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import xml.etree.ElementTree as ET
import ast
from visualize_utils import *


#############################################################################################################
#### Load the results                                                                                    ####
#############################################################################################################

# ### open
# RESULT_DIR = '../Results_fewshot/BSP_self_open'
# model_performance = load_ablation_results(RESULT_DIR)

# summary_table = pd.DataFrame(columns=['model', 'example_level', 'correct', 'total'])

# for performance in model_performance:
#     model_name = performance['model']
#     problem_name = performance['problem']
#     problem_diff_level = performance['difference']
#     result_correct = performance['correct'].count(True)
#     result_len = len(performance['correct'])

#     summary_dict = {'model':model_name, 'example_level':int(problem_diff_level), 'correct':result_correct, 'total':result_len}

#     # add the summary to the summary table
#     summary_table = summary_table._append(summary_dict, ignore_index=True)

# summary_table.sort_values(by=['model', 'example_level']).to_csv('summary/bsp_self_open_summary.csv', index=False)


### close
# RESULT_DIR = '../Results_fewshot/BSP_self_close'
model_performance = load_ablation_results(RESULT_DIR)

summary_table = pd.DataFrame(columns=['model', 'example_level', 'correct', 'total'])

for performance in model_performance:
    model_name = performance['model']
    problem_name = performance['problem']
    problem_diff_level = performance['difference']
    result_correct = performance['correct'].count(True)
    result_len = len(performance['correct'])

    summary_dict = {'model':model_name, 'example_level':int(problem_diff_level), 'correct':result_correct, 'total':result_len}

    # add the summary to the summary table
    summary_table = summary_table._append(summary_dict, ignore_index=True)

summary_table.sort_values(by=['model', 'example_level']).to_csv('summary/bsp_self_close_summary.csv', index=False)
