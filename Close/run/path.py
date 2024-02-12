import os
# path to the model type dir
# assume this file is placed in run folder of the closed and open


# Determine the directory of the current file
current_file_directory = os.path.dirname(os.path.abspath(__file__))

# Set the path to the model type directory
MODEL_TYPE_PATH = os.path.dirname(current_file_directory)
HP_HARD_PATH = os.path.dirname(os.path.dirname(current_file_directory))



