# TSP
import numpy as np
import ast

def greedy_tsp(distance_matrix):
    """
    Solve the Traveling Salesman Problem using a greedy algorithm.

    :param distance_matrix: 2D numpy array where the element at [i, j] is the distance between city i and j
    :return: A tuple containing a list of the cities in the order they were visited and the total distance
    """
    num_cities = distance_matrix.shape[0]
    unvisited_cities = set(range(num_cities))
    current_city = np.random.choice(list(unvisited_cities))
    tour = [current_city]
    total_distance = 0

    while unvisited_cities:
        unvisited_cities.remove(current_city)
        if unvisited_cities:
            # Find the nearest unvisited city
            distances_to_unvisited = distance_matrix[current_city][list(unvisited_cities)]
            nearest_city = list(unvisited_cities)[np.argmin(distances_to_unvisited)]
            tour.append(nearest_city)
            # Update the total distance
            total_distance += distance_matrix[current_city, nearest_city]
            current_city = nearest_city

    # Return to start
    total_distance += distance_matrix[current_city, tour[0]]
    tour.append(tour[0])

    return tour, total_distance

# # Example usage:

# # Assuming distance_matrix is a 2D numpy array representing the distances
# # Replace this with your actual distance matrix
# distance_matrix = np.array([
#     [0, 2, 9, 10],
#     [1, 0, 6, 4],
#     [15, 7, 0, 8],
#     [6, 3, 12, 0]
# ])

# tour, total_distance = greedy_tsp(distance_matrix)
# print(f"The greedy TSP tour: {tour}")
# print(f"Total distance of the greedy TSP tour: {total_distance}")

import xml.etree.ElementTree as ET
def parse_xml_to_dict(xml_string):
    # Parse the XML string
    root = ET.fromstring(xml_string)

    # Find the 'final_answer' tag
    final_answer_element = root.find('final_answer')

    # Find the 'reasoning' tag
    reasoning_element = root.find('reasoning')

    return final_answer_element, reasoning_element

def tspCheck(distance_matrix, llm_string):
    """
    Check if the TSP solution is complete and if the distance matches the greedy solution.
    
    :param tour_string: String representing the TSP tour in the format "0->1->2->...->N->0"
    :param distance_matrix: 2D numpy array representing the distances between cities
    :return: Boolean indicating whether the tour is complete and matches the greedy distance
    """
    # convert distance_matrix to numpy array
    distance_matrix = np.array(distance_matrix) 

    # Convert the tour string to a list of integers
    # print(llm_string)
    final_answer_element, reasoning_element = parse_xml_to_dict(llm_string)
    tour_string = ast.literal_eval(final_answer_element.text)['Path']
    tour = list(map(int, tour_string.split('->')))
    # we could also prinpt `reasoning_element` to see the reasoning of the answer
    # we could also print the final distance of the tour by `final_answer_element['Distance']`
    
    # Check if tour is a cycle
    if tour[0] != tour[-1]:
        return False, "The tour must start and end at the same city."

    # Check if all cities are visited
    if len(tour) != len(distance_matrix) + 1:
        return False, "The tour does not visit all cities exactly once."

    # Calculate the distance of the provided tour
    tour_distance = sum(distance_matrix[tour[i]][tour[i + 1]] for i in range(len(tour) - 1))

    # Find the greedy tour distance for comparison
    greedy_tour, greedy_distance = greedy_tsp(distance_matrix)

    # Check if the provided tour distance is equal to the greedy tour distance
    if tour_distance != greedy_distance:
        return False, f"The tour distance ({tour_distance}) does not match the greedy solution ({greedy_distance})."
    
    return True, "The solution is complete and matches the greedy solution distance."

# Example usage:

# # Given a tour string (replace with your tour string)
# tour_string = "0->1->2->3->0"

# # Assuming distance_matrix is a previously defined numpy array representing the distances
# validity, message = tspCheck(tour_string, distance_matrix)
# print(message)