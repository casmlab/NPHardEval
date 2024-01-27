#!/bin/bash
# these are the commands to run the experiments in the paper
# we only provide examples for each model
# we do NOT suggest running commands in this file all at once as it will take a long time

# Close source models: gpt-4-1106-preview, gpt-3.5-turbo, claude-2, claude-instant-1.2, chat-bison@001
# Open source models: Mistral-7B-Instruct-v0.1, llm-yi-34b, vicuna-13b-v1.3, phi-1_5, Baichuan-13B-Chat

# ############### close source models ###############
# Directory - need to customize
cd Close/run

# Run run_hard_GCP with different close source models, zero-shot
python run_hard_GCP.py gpt-4-1106-preview &
python run_hard_GCP.py claude-2 &

# Run run_p_BSP with different close source models, few-shot self examples
python run_p_BSP_few.py gpt-4-1106-preview self &
python run_p_BSP_few.py claude-2 self &

# Run run_p_MFP with different close source models
# few-shot with other examples not recommended

# ############### open source models ###############
# Directory - need to customize
cd Open/run

# Run run_hard_GCP with different open source models, zero-shot
python run_cmp_KSP.py phi-2 --data_dir ../Data/KSP/
python run_p_SPP.py phi-2 --data_dir ../Data/SPP/

# ############### close source models with finetuning ###############
# Directory - need to customize
cd Open/run

# example: mistral
python finetune.py --data_up_to 4 --model mistralai/Mistral-7B-Instruct-v0.1
python finetune.py --data_up_to 5 --model mistralai/Mistral-7B-Instruct-v0.1

# example: finetune on one benchmark
python run_p_BSP.py mistral --tuned_model_dir ../../safeAGI/safeAGI/running/pretrained_models/whole_model/Mistral-7B-Instruct-v0.1_1
python run_p_EDP.py mistral --tuned_model_dir ../../safeAGI/safeAGI/running/pretrained_models/whole_model/Mistral-7B-Instruct-v0.1_1
python run_p_SPP.py mistral --tuned_model_dir ../../safeAGI/safeAGI/running/pretrained_models/whole_model/Mistral-7B-Instruct-v0.1_1
python run_cmp_GCP_D.py mistral --tuned_model_dir ../../safeAGI/safeAGI/running/pretrained_models/whole_model/Mistral-7B-Instruct-v0.1_1
python run_cmp_KSP.py mistral --tuned_model_dir ../../safeAGI/safeAGI/running/pretrained_models/whole_model/Mistral-7B-Instruct-v0.1_1
python run_cmp_TSP_D.py mistral --tuned_model_dir ../../safeAGI/safeAGI/running/pretrained_models/whole_model/Mistral-7B-Instruct-v0.1_1
python run_hard_GCP.py mistral --tuned_model_dir ../../safeAGI/safeAGI/running/pretrained_models/whole_model/Mistral-7B-Instruct-v0.1_1
python run_hard_TSP.py mistral --tuned_model_dir ../../safeAGI/safeAGI/running/pretrained_models/whole_model/Mistral-7B-Instruct-v0.1_1

python run_p_BSP.py mistral --tuned_model_dir ../../safeAGI/safeAGI/running/pretrained_models/whole_model/Mistral-7B-Instruct-v0.1_1 --data_dir ../Data/finetune_data/test_1/BSP/
python run_p_EDP.py mistral --tuned_model_dir ../../safeAGI/safeAGI/running/pretrained_models/whole_model/Mistral-7B-Instruct-v0.1_1 --data_dir ../Data/finetune_data/test_1/EDP/
python run_p_SPP.py mistral --tuned_model_dir ../../safeAGI/safeAGI/running/pretrained_models/whole_model/Mistral-7B-Instruct-v0.1_1 --data_dir ../Data/finetune_data/test_1/SPP/
python run_cmp_GCP_D.py mistral --tuned_model_dir ../../safeAGI/safeAGI/running/pretrained_models/whole_model/Mistral-7B-Instruct-v0.1_1 --data_dir ../Data/finetune_data/test_1/GCP_Decision/
python run_cmp_KSP.py mistral --tuned_model_dir ../../safeAGI/safeAGI/running/pretrained_models/whole_model/Mistral-7B-Instruct-v0.1_1 --data_dir ../Data/finetune_data/test_1/KSP/
python run_cmp_TSP_D.py mistral --tuned_model_dir ../../safeAGI/safeAGI/running/pretrained_models/whole_model/Mistral-7B-Instruct-v0.1_1 --data_dir ../Data/finetune_data/test_1/TSP_Decision/
python run_hard_GCP.py mistral --tuned_model_dir ../../safeAGI/safeAGI/running/pretrained_models/whole_model/Mistral-7B-Instruct-v0.1_1 --data_dir ../Data/finetune_data/test_1/GCP/
python run_hard_TSP.py mistral --tuned_model_dir ../../safeAGI/safeAGI/running/pretrained_models/whole_model/Mistral-7B-Instruct-v0.1_1 --data_dir ../Data/finetune_data/test_1/TSP/


