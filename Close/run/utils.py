import xml.etree.ElementTree as ET
import ast


def append_root_tags(string):
    if not string.strip().startswith("<root>"):
        string = "<root>\n" + string
    if not string.strip().endswith("</root>"):
        string += "\n</root>"
    return string

def parse_xml_to_dict(xml_string: str):
    """Parse the XML string to a dictionary.

    :param xml_string: The XML string to parse.
    :return: A tuple of (output, reasoning).
    """
    # Append root tags if necessary
    print(xml_string)
    xml_string = append_root_tags(xml_string)

    # remove comments
    remove_comment_func = lambda string: string.split('//')[0].rstrip() if '//' in string else string
    xml_string = '\n'.join(remove_comment_func(line) for line in xml_string.split('\n'))
    
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