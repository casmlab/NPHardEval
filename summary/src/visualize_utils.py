"""Utils for visualization."""

################################################################################################
#### Import library                                                                         ####
################################################################################################
import json
import os
import xml.etree.ElementTree as ET
import ast


################################################################################################
#### Constants                                                                              ####
################################################################################################
problem_mapper = {
    'sppResults': 'p', 'mfpResults': 'p', 'bspResults': 'p', 'edpResults': 'p',
    'tsp_d_Results': 'np-cmp', 'gcp_d_Results': 'np-cmp', 'kspResults': 'np-cmp',
    'tspResults': 'np-hard', 'gcpResults': 'np-hard', 'mspResults': 'np-hard',
}

close_models = [
    'gpt-4-1106-preview',
    'gpt-3.5-turbo',
    'claude-2',
    'claude-instant-1.2',
    'chat-bison@001'
]

model_mapper = {
    'gpt-4-1106-preview': 'GPT 4 Turbo',
    'gpt-3.5-turbo': 'GPT 3.5 Turbo',
    'claude-2': 'Claude 2',
    'claude-instant-1.2': 'Claude Instant',
    'chat-bison@001': 'PaLM 2',
    'yi': 'Yi-34b',
    'mistral': 'Mistral-7b',
    'vicuna': 'Vicuna-13b',
    'mpt': 'MPT-30b',
    'phi': 'Phi-1.5',
    'phi-2': 'Phi-2',
    'qwen': 'Qwen-14b',
}

complexity_mapper = {
    'p': 'P',
    'np-cmp': 'NP-Complete',
    'np-hard': 'NP-Hard',
}

comp_level = {'P': 1, 'NP-Complete': 2, 'NP-Hard': 3}

model_performace = {
    "Claude 2": 2,
    "Claude Instant": 4,
    "GPT 3.5 Turbo": 3,
    "GPT 4 Turbo": 1,
    "Mistral-7b": 8,
    "MPT-30b": 10,
    "PaLM 2": 5,
    "Phi-1.5": 12,
    "Phi-2": 9,
    "Vicuna-13b": 11,
    "Yi-34b": 6,
    "Qwen-14b": 7,
}

################################################################################################
#### Helper functions                                                                       ####
################################################################################################
def append_root_tags(string):
    """Append the root tags to the XML string if necessary."""
    if not string.strip().startswith("<root>"):
        string = "<root>\n" + string
    if not string.strip().endswith("</root>"):
        string += "\n</root>"
    return string


def remove_comment_func(string):
    """Remove the comments from the XML string."""
    return string.split('//')[0].rstrip() if '//' in string else string


def parse_xml_to_dict(xml_string: str):
    """Parse the XML string to a dictionary.

    :param xml_string: The XML string to parse.
    :return: A tuple of (output, reasoning).
    """
    # Append root tags if necessary
    # print(xml_string)
    xml_string = append_root_tags(xml_string)
    xml_string = '\n'.join(remove_comment_func(line) \
                        for line in xml_string.split('\n'))

    # Parse the XML string
    root = ET.fromstring(xml_string)

    # Find the 'final_answer' tag
    final_answer_element = root.find('final_answer')

    # Find the 'reasoning' tag
    reasoning = root.find('reasoning').text.strip()

    # Convert the 'final_answer' tag to a dictionary
    output = ast.literal_eval(final_answer_element.text.strip())
    # print(reasoning_element.text)
    return output, reasoning


def select_func(x):
    """Select the correctness from the output."""
    if isinstance(x, list):
        return x[0]
    if isinstance(x, dict):
        return x.get('correctness', 'failed')
    return x


def filter_failed(x):
    """Filter the failed results."""
    return [y for y in x if y != 'failed']


def calculate_accuracy(expr_result):
    """Calculate the accuracy of the expression result."""
    # file = expr_result['file']
    model_name = expr_result['model']
    problem_name = expr_result['problem']
    difference = expr_result.get('difference', 0)
    correct_len = min(100, 100 + 10 * difference)
    correct = expr_result['correct'][-correct_len:]
    origin_level_correctness = [correct[i:i+10] for i in range(0, len(correct), 10)]
    assert len(origin_level_correctness) == min(10, 10 + difference)
    level_correctness = [filter_failed(x) for x in origin_level_correctness]
    failed_expr = [10 - len(x) for x in level_correctness]
    failed_num = sum(failed_expr)
    failed_expr = [x / 10 for x in failed_expr]
    if failed_num > 0:
        print(f'{model_name} on {problem_name} has {failed_num} failed results')
    level_accuracy = []
    for x in level_correctness:
        if len(x) == 0:
            level_accuracy.append(0)
        else:
            level_accuracy.append(sum(x) / 10)
    return {
        'model': model_name,
        'problem': problem_name,
        'difference': difference,
        'accuracy': level_accuracy,
        'failed': failed_expr,
        'level_correctness': origin_level_correctness
    }


################################################################################################
#### Load the results                                                                       ####
################################################################################################
def fetch_correctness(x):
    """Fetch the correctness from the result."""
    correctness = select_func(x.get('correctness', 'failed'))
    if not isinstance(correctness, bool):
        correctness = 'failed'
    # the output can be a string or a dictionary
    if (not correctness) and (not isinstance(x.get('output', None), dict)):
        # if it is a string, try to parse it to a dictionary
        # reusing the parse_xml_to_dict function
        # if it fails, then the case is actually failed
        if isinstance(x.get('output', None), str):
            try:
                output, _ = parse_xml_to_dict(x.get('output', None))
                if not isinstance(output, dict):
                    correctness = 'failed'
            except:
                correctness = 'failed'
    return correctness


def fetch_one_file(file, result_dir, model, problem):
    """Fetch the results from one file."""
    performance = None
    try:
        with open(result_dir + '/' + file, encoding='utf-8') as f:
            correct = []
            for line in f.readlines()[-1:]:
                data = json.loads(line)
                for x in data:
                    correctness = fetch_correctness(x)
                    correct.append(correctness)
            performance = {
                'model': model,
                'problem': problem,
                'correct': correct
            }
    except Exception as e:
        print(f'Error when reading {file}: {e}')
    return performance


def load_results(result_dir):
    """Load the results from the result directory."""
    model_performance = []
    for file in os.listdir(result_dir):
        if file.endswith('.json'):
            if "mfp" in file:
                continue
            model = file.split('_')[0]
            problem = file.split('_')
            problem = "_".join(problem[1:])
            problem = problem.split('.', maxsplit=1)[0]
            performance = fetch_one_file(file, result_dir, model, problem)
            model_performance.append(performance)
    return model_performance


def load_ablation_results(result_dir):
    """Load the results from the ablation result directory."""
    model_performance = []
    for file in os.listdir(result_dir):
        if file.endswith('.json'):
            # print(file)
            split_filename = file.split('_')
            model = split_filename[0]
            problem = split_filename[1]
            difference = int(split_filename[-1].split('.')[0])
            performance = fetch_one_file(file, result_dir, model, problem)
            performance['difference'] = difference
            model_performance.append(performance)
    return model_performance
