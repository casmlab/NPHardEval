import networkx as nx


import ast
def parse_xml_to_dict(xml_string):
    try:
        assert '<final_answer>' in xml_string
        assert '</final_answer>' in xml_string
        assert '<reasoning>' in xml_string 
        assert '</reasoning>' in xml_string
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


def read_dimacs_format(dimacs_str):
    lines = dimacs_str.strip().split('\n')
    p_line = next(line for line in lines if line.startswith('p'))
    _, _, num_vertices, num_edges = p_line.split()
    num_vertices, num_edges = int(num_vertices), int(num_edges)

    adjacency_list = {i: set() for i in range(1, num_vertices + 1)}
    for line in lines:
        if line.startswith('e'):
            _, vertex1, vertex2 = line.split()
            vertex1, vertex2 = int(vertex1), int(vertex2)
            if vertex1 in adjacency_list and vertex2 in adjacency_list:
                adjacency_list[vertex1].add(vertex2)
                adjacency_list[vertex2].add(vertex1)

    return num_vertices, adjacency_list

def gcp_greedy_solution(adjacency_list):
    """Provides a greedy solution to the GCP problem.
    
    :param adjacency_list: A dictionary of the adjacency list.
    :return: A tuple of (num_colors, coloring).
    """
    G = nx.Graph()
    G.add_nodes_from(adjacency_list.keys())
    for vertex, neighbors in adjacency_list.items():
        for neighbor in neighbors:
            G.add_edge(vertex, neighbor)
    coloring = nx.coloring.greedy_color(G, strategy='largest_first')
    num_colors = max(coloring.values()) + 1
    return num_colors, coloring


def gcp_decision_check(dimacs_str, answer, k_colors):
    """
    Check if the given GCP instance is feasible with k_colors.
    
    :param dimacs_str: The DIMACS format string of the GCP instance.
    :param answer: The answer returned by the model.
    :param k_colors: The target number of colors.
    :return: A tuple of (is_correct, message).
    """
    num_vertices, adjacency_list = read_dimacs_format(dimacs_str)
    try:
        is_feasible = answer.get('Feasible', 'no').lower() == 'yes'
    except:
        return False, "Feasible key not found"
    num_colors, coloring = gcp_greedy_solution(adjacency_list)
    exist_optimal = num_colors <= k_colors
    if is_feasible != exist_optimal:
        if exist_optimal:
            return False, f"Feasibility mismatch: {coloring}"
        else:
            return False, f"Feasibility mismatch: {is_feasible} vs {exist_optimal}"
    return True, "Feasible" if is_feasible else "Infeasible"


# # Example usage:
# dimacs_format_str = """
# p edge 14 21
# e 1 2
# e 1 5
# e 2 3
# e 2 6
# e 3 4
# e 3 7
# e 4 8
# e 5 6
# e 5 9
# e 6 7
# e 6 10
# e 7 8
# e 7 11
# e 8 12
# e 9 10
# e 9 13
# e 10 11
# e 10 14
# e 11 12
# e 11 15
# e 12 16
# """
# answer_str = {'Feasible': 'NO'}
# k_colors = 2  # The target number of colors

# print(gcp_decision_check(dimacs_format_str, answer_str, k_colors))
