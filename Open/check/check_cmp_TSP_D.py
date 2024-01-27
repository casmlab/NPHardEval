import numpy as np
import pandas as pd
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

def tsp_approx(distance_matrix):
    """Returns an approximate solution to the TSP problem.
    
    :param distance_matrix: A 2D numpy array representing the distance matrix.
    :return: A list of the cities in the order they were visited.
    """
    G = nx.from_numpy_array(distance_matrix)
    return nx.approximation.traveling_salesman_problem(G)

def tsp_decision_check(distance_matrix, threshold, tour):
    """
    Checks if a given TSP tour is valid and within the threshold distance.

    :param distance_matrix: A 2D numpy array representing the distance matrix.
    :param threshold: The maximum distance allowed.
    :param tour: A dictionary containing the feasibility.
    """
    try:
        is_feasible = tour.get('Feasible', 'no').lower() == 'yes'
    except:
        return False, "Output format incorrect"

    # Calculate the approxed distance of the tour
    tours = tsp_approx(distance_matrix)
    tour_distance = sum(distance_matrix[tours[i], tours[i + 1]] for i in range(len(tours) - 1)) + \
                    distance_matrix[tours[-1], tours[0]]

    if is_feasible != (tour_distance <= threshold):
        return False, f"Feasibility mismatch: {is_feasible} vs {tour_distance} > {threshold}"
    return True, "Feasible: {} <= {}".format(tour_distance, threshold)



# # Example usage:

# # Path to the CSV file
# file_path = '../Data/TSP_Decision/decision_data_TSP_level_0_instance_1.csv'

# # Given a tour string (replace with your tour string)
# tour_string = {'Feasible': 'yes'}

# validity, message = tsp_decision_check(file_path, tour_string)
# print(message)