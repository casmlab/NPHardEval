import json
from collections import defaultdict
import networkx as nx


def mfp_optimal_solution(num_nodes, edge_capacities, source, target):
    """Provides the optimal solution for the MFP instance.

    :param num_nodes: The number of nodes in the graph.
    :param edge_capacities: A dictionary of the edge capacities.
    :param source: The source node.
    :param target: The target node.
    :return: The optimal maximum flow.
    """
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    for edge_name, edge_capacity in edge_capacities.items():
        from_node, to_node = map(int, edge_name.split('->'))
        G.add_edge(from_node, to_node, weight=edge_capacity)
    max_flow = None
    if nx.has_path(G, source=source, target=target):
        max_flow = nx.maximum_flow_value(G, source, target, capacity='weight')
    return max_flow


def mfp_check(instance, solution):
    """Validate the solution of the MFP problem.

    :param instance: The instance dictionary with nodes, edges, source, and sink.
    :param solution: The solution dictionary with the maximum flow and flows.
    :return: A tuple of (is_correct, message).
    """
    # Get the start and end nodes
    # Curently, the start and end nodes are the first and last nodes in the instance
    num_nodes = instance['nodes']
    start_node = instance['source']
    end_node = instance['sink']

    # Initialize edge flows
    edges = instance['edges']
    edge_name_func = lambda from_node, to_node: f'{from_node}->{to_node}' if from_node < to_node else f'{to_node}->{from_node}'
    edge_capacities = defaultdict(int)
    for edge in edges:
        edge_name = edge_name_func(edge['from'], edge['to'])
        edge_capacities[edge_name] += int(edge['capacity'])
    edge_flows = {edge_name: 0 for edge_name in edge_capacities.keys()}

    # Convert solution to dictionary
    flows = solution.get('Flows', {})
    max_flow = solution.get('MaxFlow', -1)

    # Get the optimal solution
    mfp_optimal_flow = mfp_optimal_solution(num_nodes, edge_capacities, start_node, end_node)
    
    if isinstance(max_flow, str):
        if max_flow.isdigit():
            max_flow = int(max_flow)
        elif mfp_optimal_flow is None:
            return True, f"There is no path from the start node to the end node, and the solution is {max_flow}."
        else:
            return False, f"The problem should be feasible ({mfp_optimal_flow}), but the solution is {max_flow}."
    
    if mfp_optimal_flow is None:
        if max_flow > 0:
            return False, f"The problem should be infeasible."
        else:
            return True, "There is no path from the start node to the end node."
    elif max_flow < 0:
        return False, f"The problem should be feasible ({mfp_optimal_flow}), but the solution is {max_flow}."

   
    # Initialize node flows
    node_flows = [0 for _ in range(num_nodes)]
    node_flows[start_node] = max_flow
    node_flows[end_node] = -max_flow


    # Check if the flow is valid
    for edge, flow in flows.items():
        flow = int(flow)
        from_node, to_node = map(int, edge.split('->'))
        node_flows[from_node] -= flow
        node_flows[to_node] += flow
        edge_name = edge_name_func(from_node, to_node)
        edge_flow = flow
        if from_node > to_node:
            edge_flow = -flow
        if edge_name not in edge_flows:
            return False, f"Edge {edge} does not exist."
        edge_flows[edge_name] += edge_flow

    # Check the node conservation
    for node_id, node_flow in enumerate(node_flows):
        if node_flow != 0:
            return False, f"Node {node_id} is not conserved."
    
    # Check the edge capacities
    for edge_name, edge_flow in edge_flows.items():
        edge_capacity = edge_capacities[edge_name]
        if abs(edge_flow) > edge_capacity:
            return False, f"Edge {edge_name} with {edge_flow} exceeds its capacity {edge_capacity}."

    # Check if the flow is optimal
    if max_flow != mfp_optimal_flow:
        return False, f"The calculated flow ({max_flow}) does not match the optimal solution ({mfp_optimal_flow})."
    return True, "The solution is valid."


# # Example usage:
# # Define an example MFP instance
# mfp_instance =   {
#     "nodes": 4,
#     "edges": [
#         {"from": 0, "to": 3, "capacity": 3},
#         {"from": 0, "to": 3, "capacity": 4},
#         {"from": 0, "to": 3, "capacity": 2},
#         {"from": 2, "to": 3, "capacity": 3},
#         {"from": 0, "to": 2, "capacity": 4},
#         {"from": 1, "to": 2, "capacity": 3}
#     ],
#     "source": 0,
#     "sink": 3,
#     "complexity_level": 3
#   }

# # Define a solution for the MFP instance
# mfp_solution = {"MaxFlow": 12, "Flows": {"0->3": 9, "0->2": 3, "2->3": 3}}

# # Validate the solution
# is_valid, message = mfp_check(mfp_instance, mfp_solution)
# print(is_valid, message)
