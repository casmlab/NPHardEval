import xml.etree.ElementTree as ET
import ast
def parse_xml_to_dict(xml_string):
    try:
        assert '<final_answer>' in xml_string
        assert '</final_answer>' in xml_string
        final_answer_start = xml_string.index('<final_answer>') + len('<final_answer>') 
        final_answer_end = xml_string.index('</final_answer>')
        final_answer_element  = xml_string[final_answer_start:final_answer_end].rstrip().strip().rstrip()
        reasoning_element = ''
        if '<reasoning>' in xml_string and '</reasoning>' in xml_string:
            reasoning_start = xml_string.index('<reasoning>') + len('<reasoning>')
            reasoning_end = xml_string.index('</reasoning>')
            reasoning_element = xml_string[reasoning_start:reasoning_end].rstrip().strip().rstrip()
        try:
            final_answer_element = ast.literal_eval(final_answer_element)
        except:
            final_answer_element = ''
    except:
        final_answer_element = ''
        reasoning_element = ''

    return final_answer_element, reasoning_element


def bsp_check(instance, solution):
    """Check if the binary search solution is valid.

    :param instance: The instance dictionary with array and target value.
    :param solution: The solution dictionary with the position of the target value.
    :return: A tuple of (is_correct, message).
    """
    array = sorted(instance['array'])
    target_value = instance['target']
    solution, reasoning = parse_xml_to_dict(solution)
    if isinstance(solution, str):
        return False, f"The solution is invalid."
    try:
        position = int(solution['Position']) 
    except:
        return False, f"The solution is invalid."
    if position == -1 or position >= len(array):
        return False, f"The solution is invalid."
    elif array[position] != target_value:
        return False, f"The target index is incorrect."
    return True, "The solution is valid."