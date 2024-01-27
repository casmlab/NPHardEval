
import ast
import json
import networkx as nx


def ssp_optimal_solution(instance, source, target):
    """Provides the optimal solution for the SSP instance.

    :param instance: The SSP instance as a dictionary with 'nodes' and 'edges'.
    :param source: The source node.
    :param target: The destination node.
    :return: The optimal shortest path length and path.
    """
    G = nx.Graph()
    G.add_nodes_from(instance['nodes'])
    G.add_weighted_edges_from([(edge['from'], edge['to'], edge['weight']) for edge in instance['edges']])
    shortest_path_length = None
    shortest_path = None
    if nx.has_path(G, source=source, target=target):
        shortest_path_length = nx.shortest_path_length(G, source=source, target=target, weight='weight')
        shortest_path = nx.shortest_path(G, source=source, target=target, weight='weight')
    return shortest_path_length, shortest_path


# SPP
def spp_check(instance, solution, start_node=None, end_node=None):
    """Validate the solution of the SPP problem.

    :param instance: The instance dictionary with nodes and edges.
    :param solution: The solution dictionary with the path and total distance.
    :param start_node: The start node.
    :param end_node: The end node.
    :return: A tuple of (is_correct, message).
    """
    # Get the start and end nodes
    # Curently, the start and end nodes are the first and last nodes in the instance
    if start_node is None:
        start_node = instance['nodes'][0]
    if end_node is None:
        end_node = instance['nodes'][-1]

    # Convert solution to dictionary
    path_string = solution.get('Path', '')
    cost_string = solution.get('TotalDistance', '')

    # Calculate the optimal solution
    ssp_optimal_length, ssp_optimal_path = ssp_optimal_solution(instance, start_node, end_node)
    if ssp_optimal_length is None:
        if isinstance(cost_string, int) or cost_string.isdigit():
            return False, f"No path between from node {start_node} to node {end_node}."
        else:
            return True, "No path found from node {start_node} to node {end_node}."

    path = list(map(int, path_string.split('->')))
    total_cost = int(cost_string)

    # Check if path starts and ends with the correct nodes
    if not path or path[0] != start_node or path[-1] != end_node:
        return False, "The path does not start or end at the correct nodes."

    # Check if the path is continuous and calculate the cost
    calculated_cost = 0
    is_in_edge = lambda edge, from_node, to_node: (edge['from'] == from_node and edge['to'] == to_node) or (edge['from'] == to_node and edge['to'] == from_node)
    for i in range(len(path) - 1):
        from_node, to_node = path[i], path[i + 1]
        edge = next((edge for edge in instance['edges'] if is_in_edge(edge, from_node, to_node)), None)

        if not edge:
            return False, f"No edge found from node {from_node} to node {to_node}."

        calculated_cost += edge['weight']

    # Check if the calculated cost matches the total cost provided in the solution
    if calculated_cost != total_cost:
        return False, f"The calculated cost ({calculated_cost}) does not match the provided total cost ({total_cost})."

    if calculated_cost != ssp_optimal_length:
        spp_optimal_path = '->'.join(map(str, ssp_optimal_path))
        return False, f"The calculated cost ({calculated_cost}) does not match the optimal solution ({ssp_optimal_length}): {ssp_optimal_path}."

    return True, "The solution is valid."

# # Example usage:
# # Define an example SPP instance
# spp_instance = {
#     'nodes': [0, 1, 2, 3],
#     'edges': [
#         {'from': 0, 'to': 1, 'weight': 4},
#         {'from': 1, 'to': 2, 'weight': 1},
#         {'from': 2, 'to': 3, 'weight': 3},
#         {'from': 0, 'to': 3, 'weight': 6}
#     ],
#     'complexity_level': 1
# }

# # Define a solution for the SPP instance
# spp_solution = {
#     'Path': "0->1->2->3",
#     'TotalDistance': 8
# }

# # Validate the solution
# is_valid, message = spp_check(spp_instance, spp_solution, start_node=0, end_node=3)
# print(is_valid, message)
