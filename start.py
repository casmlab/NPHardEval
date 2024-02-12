import sys
import os
import argparse
import subprocess

def main():
    """
    Main function to execute a specific test script based on the arguments provided.

    This script sets up an argument parser to process command line arguments for running different tests on a learning model system (LLMS). 
    The script allows users to specify the type of model (open or close), the test to run, the model name, and additional options for few-shot learning and prompt styles.

    The script dynamically constructs the command to run the appropriate test script based on the provided arguments. 
    It handles different cases for open and close model types, including adjustments for few-shot learning and prompt styles.
    IMPORTANT: please assume the current working directory is NPHARDEVAL

    Raises:
        NameError: If 'prompt_style' is not specified when required.

    Note:
        This script must be run with the necessary command line arguments. 
        It does not execute any functionality if the arguments are not provided or are incorrect.

    example:
        cd NPHARDEVAL 
        python start.py Close p_BSP gpt-4-1106-preview --fewshot True --prompt_style self
        python start.py Close hard_TSP gpt-4-1106-preview
        python start.py Close cmp_KSP gpt-4-1106-preview  

    """
    # This is the outer arg parser.
    parser = argparse.ArgumentParser(description='parse user to run different test on llms')

    parser.add_argument('model_type', 
                        metavar='model type', type=str, nargs=1,
                        choices=['Open','Close'],
                        help='''argument to choose open model or close model,
                                open for open model,
                                close for closed model''')

    parser.add_argument('test', 
                        metavar='test name', type=str, nargs=1,
                        choices=['cmp_GCP_D','cmp_KSP','cmp_TSP_D','hard_GCP',
                                'hard_MSP','hard_TSP','p_BSP','p_EDP','p_MFP','p_SPP'],
                        help='''argument to choose actual test''')

    parser.add_argument('--fewshot', 
                        metavar='few shot input', type=bool, nargs=1,
                        choices=[True,False],
                        help='''argument to choose if fewshot''')

    parser.add_argument('model', 
                        metavar='model name', type=str, nargs=1,
                        help='''argument to choose llm model''')

    parser.add_argument('--prompt_style', 
                        metavar='few shot input', type=str, nargs=1,
                        choices=['self','other'],
                        help='''argument to prompt style''')

    # args for open model
    # important please assume the current working directory is NPHARDEVAL
    parser.add_argument('--data_dir', type=str, default='Data/BSP/', help='Data/finetune_data/test_1/BSP/')
    parser.add_argument('--tuned_model_dir', type=str, default='')
    parser.add_argument('--difficulty_level', type=int, default=0, help="-5, -4, -3, ...")
    parser.add_argument('--example_number', type=int, default=5, help="2,3,4,5")

    arg = parser.parse_args()
    model_type = arg.model_type[0] 
    test_name = f'run_{arg.test[0]}'
    prompt_style = ''

    # edit _few to run file name
    if arg.fewshot and arg.fewshot[0] == True and model_type == 'Close':
        test_name +='_few'
    if arg.prompt_style:
        # the space is left to sep model name and prompt style
        prompt_style = f'{arg.prompt_style[0]}'
    test_name += '.py'

    model_name = arg.model[0]

    inner_parser_script = os.path.join(os.getcwd(), model_type,'run',test_name)

    # handle close case
    if model_type == 'Close':
        # pass in different param based on if fewshot 
        if arg.fewshot and arg.fewshot[0] == True:
            if not prompt_style or prompt_style == '':
                raise NameError('Please choose prompt_style from self and other')
            command = [sys.executable, inner_parser_script, model_name, prompt_style]
        else:
            command = [sys.executable, inner_parser_script, model_name]
    # handle open case
    else:
        command = [
        sys.executable, inner_parser_script, 
        '--data_dir', arg.data_dir,
        '--tuned_model_dir', arg.tuned_model_dir,
        '--difficulty_level', str(arg.difficulty_level),
        '--example_number', str(arg.example_number)
    ]

    subprocess.run(command)               

if __name__ == "__main__":
    main()
