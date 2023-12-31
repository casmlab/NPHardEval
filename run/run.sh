#!/bin/bash
# these are the commands to run the experiments in the paper
# we only provide examples for each model
# we do NOT suggest running commands in this file all at once as it will take a long time

# Close source models: gpt-4-1106-preview, gpt-3.5-turbo, claude-2, claude-instant-1.2, chat-bison@001
# Open source models: Mistral-7B-Instruct-v0.1, llm-yi-34b, vicuna-13b-v1.3, phi-1_5, Baichuan-13B-Chat

# Run run_hard_GCP with different close source models, zero-shot
python run_close_zeroshot/run_hard_GCP.py gpt-4-1106-preview &
python run_close_zeroshot/run_hard_GCP.py claude-2 &

# Run run_p_BSP with different close source models, few-shot self examples
python run_close_fewshot/run_p_BSP.py gpt-4-1106-preview self &
python run_close_fewshot/run_p_BSP.py claude-2 self &

# Run run_p_EDP with different close source models, few-shot with other examples
python run_close_fewshot/run_p_EDP.py gpt-4-1106-preview other &
python run_close_fewshot/run_p_EDP.py claude-2 other &

# Run run_p_BSP with different open source models, few-shot with other examples
python run_open_zeroshot/run_p_BSP.py mistral 
python run_open_zeroshot/run_p_BSP.py phi 

# Run run_p_EDP with different open source models, few-shot with other examples
python run_open_fewshot/run_p_EDP.py mistral --prompt_question_type EDP --difficulty_level 0
python run_open_fewshot/run_p_EDP.py phi --prompt_question_type EDP --difficulty_level 5

# Wait for all background jobs to finish
wait
