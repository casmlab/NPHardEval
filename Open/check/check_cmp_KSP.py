import ast
import json


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

def ksp_optimal_solution(knapsacks, capacity):
    """Provides the optimal solution for the KSP instance with dynamic programming.
    
    :param knapsacks: A dictionary of the knapsacks.
    :param capacity: The capacity of the knapsack.
    :return: The optimal value.
    """
    num_knapsacks = len(knapsacks)

    # Create a one-dimensional array to store intermediate solutions
    dp = [0] * (capacity + 1)

    for itemId, (weight, value) in knapsacks.items():
        for w in range(capacity, weight - 1, -1):
            dp[w] = max(dp[w], value + dp[w - weight])

    return dp[capacity]


# KSP
def kspCheck(instance, solution):
    """Validates the solution for the KSP instance.

    :param instance: A dictionary of the KSP instance.
    :param solution: A dictionary of the solution.
    :return: A tuple of (is_correct, message).
    """
    # Change string key to integer key and value to boolean
    items = instance.get('items', [])
    knapsacks = {item['id']: (item['weight'], item['value']) for item in items}

    ksp_optimal_value = ksp_optimal_solution(knapsacks, instance['knapsack_capacity'])

    try:
        is_feasible = (solution.get('Feasible', '').lower() == 'yes')
    except:
        return False, f"Output format is incorrect."
    if is_feasible != (ksp_optimal_value > 0):
        return False, f"The solution is {is_feasible} but the optimal solution is {ksp_optimal_value > 0}."
    
    try:
        total_value = int(solution.get('TotalValue', -1))
        selectedItems = list(map(int, solution.get('SelectedItemIds', [])))
    except:
        return False, f"Output format is incorrect."

    if len(set(selectedItems)) != len(selectedItems):
        return False, f"Duplicate items are selected."

    total_weight = 0
    cum_value = 0

    # Calculate total weight and value of selected items
    for item in selectedItems:
        if knapsacks.get(item, False):
            weight, value = knapsacks[item]
            total_weight += weight
            cum_value += value
        else:
            return False, f"Item {item} does not exist."

    # Check if the item weight exceeds the knapsack capacity
    if total_weight > instance['knapsack_capacity']:
        return False, f"Total weight {total_weight} exceeds knapsack capacity {instance['knapsack_capacity']}."

    if total_value != cum_value:
        return False, f"The total value {total_value} does not match the cumulative value {cum_value} of the selected items."

    if total_value != ksp_optimal_value:
        return False, f"The total value {total_value} does not match the optimal value {ksp_optimal_value}."
    
    return True, f"The solution is valid with total weight {total_weight} and total value {total_value}."

# # Example usage:
# # Define an example KSP instance
# # ksp_instance = {
# #     'items': [
# #         {'id': 0, 'weight': 10, 'value': 60},
# #         {'id': 1, 'weight': 20, 'value': 100},
# #     ],
# #     'knapsack_capacity': 20
# # }

# # Define a solution for the KSP instance
# ksp_solution = {
#     "Feasible": "YES",
#     "TotalValue": 3,
#     "SelectedItemIds": [0]
# }

# # Validate the solution
# is_valid, message = kspCheck(ksp_instance, ksp_solution)
# print(is_valid, message)