import xml.etree.ElementTree as ET
import ast
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from path import MODEL_TYPE_PATH, HP_HARD_PATH

from models import *

def parse_xml_to_dict(xml_string: str):
    """_summary_

    Args:
        xml_string (str): llm output string

    Returns:
        dict: dictionary of llm output
    """
    # Parse the XML string
    root = ET.fromstring(xml_string)

    # Find the 'final_answer' tag
    final_answer_element = root.find('final_answer')

    # Find the 'reasoning' tag
    reasoning_element = root.find('reasoning')

    # Convert the 'final_answer' tag to a dictionary
    output = ast.literal_eval(final_answer_element.text)
    print(reasoning_element.text)
    return output

def run_opensource_models(args, MODEL, all_prompts):
    if MODEL.startswith('mistral'):
        if args.tuned_model_dir:
            output = run_mistral(all_prompts, model_name=args.tuned_model_dir)
        else:
            output = run_mistral(all_prompts)
    elif MODEL.startswith('mixtral'):
        output = run_mixtral(all_prompts)
    elif MODEL.startswith('yi'):
        output = run_yi(all_prompts)
    elif MODEL.startswith('phi-2'):
        if args.tuned_model_dir:
            output = run_phi_2(all_prompts, model_name=args.tuned_model_dir)
        else:
            output = run_phi_2(all_prompts)
    elif MODEL.startswith('mpt'):
        output = run_mpt(all_prompts)
    elif MODEL.startswith('phi'):
        output = run_phi(all_prompts)
    elif MODEL.startswith('vicuna'):
        if args.tuned_model_dir:
            output = run_vicuna(all_prompts, model_name=args.tuned_model_dir)
        else:    
            output = run_vicuna(all_prompts)
    elif MODEL.startswith('qwen'):
        if args.tuned_model_dir:
            output = run_qwen(all_prompts, model_name=args.tuned_model_dir)
        else:
            output = run_qwen(all_prompts)
    else:
        raise NotImplementedError
    return output


def find_data_path(file_path):
    """
    Determine the data file path and result file path based on a given file path.

    This function parses the file name from the provided `file_path`, identifies specific components 
    of the file name, and constructs paths for the data file and result file based on these components 
    and predefined directory structures. It will also create the result folder, if not exist.

    Parameters:
    file_path (str): The file path of the file for which the data and result paths are to be determined.

    Returns:
    tuple: A tuple containing two strings. The first string is the path for the data file, and the second 
    string is the path for the result file. 

    The function assumes a specific naming convention for the files. The file name is expected to be in 
    the format 'prefix_tasktype_taskname_suffix.ext', where 'tasktype' and 'taskname' are used to determine 
    the paths. If the suffix is 'few', it indicates a few-shot scenario, and if it is 'D', it indicates a 
    decision-related task. These suffixes modify the result file name accordingly. The paths are constructed 
    using predefined base paths ('MODEL_TYPE_PATH' and 'NP_HARD_PATH').
    """
    
    file_name = os.path.basename(file_path)
    file_name, file_extension = os.path.splitext(file_name)
    name_part = file_name.split('_')
    
    task_file_name = name_part[2]
    res_file_name = 'Results'

    if len(name_part) == 4:
        if name_part[-1] == 'few':
            res_file_name += '_fewshot'
        elif name_part[-1] == 'D':
            task_file_name += '_Decision'
    
    data_file_name = os.path.join(HP_HARD_PATH, 'Data' ,task_file_name) + os.sep
    res_file_name = os.path.join(MODEL_TYPE_PATH, res_file_name) + os.sep

    if not os.path.exists(res_file_name):
        # Create the directory along with any necessary intermediate directories
        os.makedirs(res_file_name)
        print(f"Directory created at {res_file_name}")
    else:
        # If the directory already exists
        print(f"Directory already exists at {res_file_name}")

    return data_file_name, res_file_name