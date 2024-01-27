def bsp_check(instance, solution):
    """Check if the binary search solution is valid.

    :param instance: The instance dictionary with array and target value.
    :param solution: The solution dictionary with the position of the target value.
    :return: A tuple of (is_correct, message).
    """
    array = sorted(instance['array'])
    target_value = instance['target']
    position = int(solution.get('Position', -1))
    if position == -1 or position >= len(array):
        return False, f"The solution is invalid."
    elif array[position] != target_value:
        return False, f"The target index is incorrect."
    return True, "The solution is valid."