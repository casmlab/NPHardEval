import ast

import xml.etree.ElementTree as ET
def parse_xml_to_dict(xml_string):
    # Parse the XML string
    root = ET.fromstring(xml_string)

    # Find the 'final_answer' tag
    final_answer_element = root.find('final_answer')

    # Find the 'reasoning' tag
    reasoning_element = root.find('reasoning')

    return final_answer_element, reasoning_element

# MSP
def mspCheck(instance, llm_string):
    """
    Validate the MSP solution.

    Parameters:
    - instance: The MSP instance as a dictionary.
    - solution: A dictionary with meeting ids as keys and lists of scheduled time slots as values.

    Returns:
    - A tuple (is_valid, message). is_valid is True if the solution is valid, False otherwise.
      message contains information about the validity of the solution.
    """
    # print(llm_string)
    solution, reasoning_element = parse_xml_to_dict(llm_string)
    # print(solution.text)

    # convert solution to dictionary
    solution = ast.literal_eval(solution.text)
    # convert key type to int
    solution = {int(k):v for k,v in solution.items()}

    # Check if all meetings are scheduled within the available time slots
    for meeting in instance['meetings']:
        m_id = meeting['id']
        duration = meeting['duration']
        scheduled_slots = solution.get(m_id, None)

        # Check if the meeting is scheduled
        if scheduled_slots is None:
            return False, f"Meeting {m_id} is not scheduled."

        # Check if the meeting fits within the number of total time slots
        if any(slot >= instance['time_slots'] for slot in scheduled_slots):
            return False, f"Meeting {m_id} does not fit within the available time slots."

        # Check if the scheduled slots are contiguous and fit the meeting duration
        if len(scheduled_slots) != duration or not all(
                scheduled_slots[i] + 1 == scheduled_slots[i + 1] for i in range(len(scheduled_slots) - 1)):
            return False, f"Meeting {m_id} is not scheduled in contiguous time slots fitting its duration."

        # Check if all participants are available at the scheduled time
        for p_id, participant in instance['participants'].items():
            if m_id in participant['meetings']:
                if not all(slot in participant['available_slots'] for slot in scheduled_slots):
                    return False, f"Participant {p_id} is not available for meeting {m_id} at the scheduled time."

    # Check if any participant is double-booked
    participants_schedule = {p_id: [] for p_id in instance['participants']}
    for m_id, time_slots in solution.items():
        try:
            duration = next(meeting['duration'] for meeting in instance['meetings'] if meeting['id'] == m_id)
            if len(time_slots) != duration:
                return False, f"Meeting {m_id} duration does not match the number of scheduled time slots."
            for p_id, participant in instance['participants'].items():
                if m_id in participant['meetings']:
                    participants_schedule[p_id].extend(time_slots)
        except:
            return False, f"Meeting {m_id} is not in the instance or program error."

    for p_id, slots in participants_schedule.items():
        if len(slots) != len(set(slots)):
            return False, f"Participant {p_id} is double-booked."

    return True, "The solution is valid."

# # Example usage:
# # Define an example MSP instance
# # Need to figure out how to deal with each timeslot's duration
# msp_instance = {
#     'meetings': [
#         {'id': 0, 'duration': 2},
#         {'id': 1, 'duration': 1}
#     ],
#     'participants': {
#         0: {'available_slots': [0, 1, 2, 3], 'meetings': [0]},
#         1: {'available_slots': [1, 2, 3, 4], 'meetings': [0, 1]}
#     },
#     'time_slots': 5,
#     'complexity_level': 1
# }

# # Define a solution for the MSP instance
# # meeting : list of slots
# # msp_solution = {
# #     0: [1,2],  # Meeting 0 scheduled at time slot 0
# #     1: [3]   # Meeting 1 scheduled at time slot 3
# # }

# # # Validate the solution
# # is_valid, message = mspCheck(msp_instance, msp_solution)
# # print(is_valid, message)
