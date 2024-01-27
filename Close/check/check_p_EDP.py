def compute_min_edit_distance(string_a, string_b):
    """Computes the minimum edit distance between two strings using dynamic programming."""
    m, n = len(string_a), len(string_b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif string_a[i - 1] == string_b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]

def edp_check(instance, solution):
    """Check if the edit distance solution is valid.

    :param instance: The instance dictionary with 'string_a' and 'string_b'.
    :param solution: The solution dictionary with the reported 'edit_distance'.
    :return: A tuple of (is_correct, message).
    """
    string_a = instance['string_a']
    string_b = instance['string_b']
    reported_distance = int(solution.get('Operations', -1))

    actual_distance = compute_min_edit_distance(string_a, string_b)

    if reported_distance == -1:
        return False, "No solution provided."
    elif reported_distance != actual_distance:
        return False, f"The reported edit distance ({reported_distance}) is incorrect. Actual distance: {actual_distance}."
    return True, "The solution is valid."

# # Example usage:
# # Define an example EDP instance
# edp_instance = {
#     'string_a': "kitten",
#     'string_b': "sitting"
# }

# # Define a solution for the EDP instance
# edp_solution = {
#     'Operations': 3
# }

# # Validate the solution
# is_valid, message = edp_check(edp_instance, edp_solution)
# print(0)
# print(is_valid, message)
