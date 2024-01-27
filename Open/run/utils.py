import xml.etree.ElementTree as ET
import ast
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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