#!/bin/bash

######## mistral ########
python finetune.py --data_up_to 4 --model mistralai/Mistral-7B-Instruct-v0.1
python finetune.py --data_up_to 5 --model mistralai/Mistral-7B-Instruct-v0.1

###### finetune on one benchmark ######
python run_p_BSP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_1
python run_p_EDP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_1
python run_p_SPP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_1
python run_cmp_GCP_D.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_1
python run_cmp_KSP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_1
python run_cmp_TSP_D.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_1
python run_hard_GCP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_1
python run_hard_TSP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_1

python run_p_BSP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_1 --data_dir ../Data/finetune_data/test_1/BSP/
python run_p_EDP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_1 --data_dir ../Data/finetune_data/test_1/EDP/
python run_p_SPP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_1 --data_dir ../Data/finetune_data/test_1/SPP/
python run_cmp_GCP_D.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_1 --data_dir ../Data/finetune_data/test_1/GCP_Decision/
python run_cmp_KSP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_1 --data_dir ../Data/finetune_data/test_1/KSP/
python run_cmp_TSP_D.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_1 --data_dir ../Data/finetune_data/test_1/TSP_Decision/
python run_hard_GCP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_1 --data_dir ../Data/finetune_data/test_1/GCP/
python run_hard_TSP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_1 --data_dir ../Data/finetune_data/test_1/TSP/

###### finetune on two benchmark ######
python run_p_BSP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_2
python run_p_EDP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_2
python run_p_SPP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_2
python run_cmp_GCP_D.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_2
python run_cmp_KSP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_2
python run_cmp_TSP_D.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_2
python run_hard_GCP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_2
python run_hard_TSP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_2

python run_p_BSP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_2 --data_dir ../Data/finetune_data/test_1/BSP/
python run_p_EDP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_2 --data_dir ../Data/finetune_data/test_1/EDP/
python run_p_SPP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_2 --data_dir ../Data/finetune_data/test_1/SPP/
python run_cmp_GCP_D.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_2 --data_dir ../Data/finetune_data/test_1/GCP_Decision/
python run_cmp_KSP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_2 --data_dir ../Data/finetune_data/test_1/KSP/
python run_cmp_TSP_D.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_2 --data_dir ../Data/finetune_data/test_1/TSP_Decision/
python run_hard_GCP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_2 --data_dir ../Data/finetune_data/test_1/GCP/
python run_hard_TSP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_2 --data_dir ../Data/finetune_data/test_1/TSP/

###### finetune on three benchmark ######
python run_p_BSP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_3
python run_p_EDP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_3
python run_p_SPP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_3
python run_cmp_GCP_D.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_3
python run_cmp_KSP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_3
python run_cmp_TSP_D.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_3
python run_hard_GCP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_3
python run_hard_TSP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_3

python run_p_BSP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_3 --data_dir ../Data/finetune_data/test_1/BSP/
python run_p_EDP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_3 --data_dir ../Data/finetune_data/test_1/EDP/
python run_p_SPP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_3 --data_dir ../Data/finetune_data/test_1/SPP/
python run_cmp_GCP_D.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_3 --data_dir ../Data/finetune_data/test_1/GCP_Decision/
python run_cmp_KSP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_3 --data_dir ../Data/finetune_data/test_1/KSP/
python run_cmp_TSP_D.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_3 --data_dir ../Data/finetune_data/test_1/TSP_Decision/
python run_hard_GCP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_3 --data_dir ../Data/finetune_data/test_1/GCP/
python run_hard_TSP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_3 --data_dir ../Data/finetune_data/test_1/TSP/

###### finetune on four benchmark ######
python run_p_BSP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_4
python run_p_EDP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_4
python run_p_SPP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_4
python run_cmp_GCP_D.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_4
python run_cmp_KSP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_4
python run_cmp_TSP_D.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_4
python run_hard_GCP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_4
python run_hard_TSP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_4

python run_p_BSP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_4 --data_dir ../Data/finetune_data/test_1/BSP/
python run_p_EDP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_4 --data_dir ../Data/finetune_data/test_1/EDP/
python run_p_SPP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_4 --data_dir ../Data/finetune_data/test_1/SPP/
python run_cmp_GCP_D.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_4 --data_dir ../Data/finetune_data/test_1/GCP_Decision/
python run_cmp_KSP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_4 --data_dir ../Data/finetune_data/test_1/KSP/
python run_cmp_TSP_D.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_4 --data_dir ../Data/finetune_data/test_1/TSP_Decision/
python run_hard_GCP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_4 --data_dir ../Data/finetune_data/test_1/GCP/
python run_hard_TSP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_4 --data_dir ../Data/finetune_data/test_1/TSP/

###### finetune on five benchmark ######
python run_p_BSP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_5
python run_p_EDP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_5
python run_p_SPP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_5
python run_cmp_GCP_D.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_5
python run_cmp_KSP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_5
python run_cmp_TSP_D.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_5
python run_hard_GCP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_5
python run_hard_TSP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_5

python run_p_BSP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_5 --data_dir ../Data/finetune_data/test_1/BSP/
python run_p_EDP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_5 --data_dir ../Data/finetune_data/test_1/EDP/
python run_p_SPP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_5 --data_dir ../Data/finetune_data/test_1/SPP/
python run_cmp_GCP_D.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_5 --data_dir ../Data/finetune_data/test_1/GCP_Decision/
python run_cmp_KSP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_5 --data_dir ../Data/finetune_data/test_1/KSP/
python run_cmp_TSP_D.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_5 --data_dir ../Data/finetune_data/test_1/TSP_Decision/
python run_hard_GCP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_5 --data_dir ../Data/finetune_data/test_1/GCP/
python run_hard_TSP.py mistral --tuned_model_dir ../../whole_model/Mistral-7B-Instruct-v0.1_5 --data_dir ../Data/finetune_data/test_1/TSP/


###### qwen ######
###### finetune on one benchmark ######
python run_p_BSP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_1
python run_p_EDP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_1
python run_p_SPP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_1
python run_cmp_GCP_D.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_1
python run_cmp_KSP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_1
python run_cmp_TSP_D.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_1
python run_hard_GCP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_1
python run_hard_TSP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_1

python run_p_BSP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_1 --data_dir ../Data/finetune_data/test_1/BSP/
python run_p_EDP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_1 --data_dir ../Data/finetune_data/test_1/EDP/
python run_p_SPP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_1 --data_dir ../Data/finetune_data/test_1/SPP/
python run_cmp_GCP_D.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_1 --data_dir ../Data/finetune_data/test_1/GCP_Decision/
python run_cmp_KSP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_1 --data_dir ../Data/finetune_data/test_1/KSP/
python run_cmp_TSP_D.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_1 --data_dir ../Data/finetune_data/test_1/TSP_Decision/
python run_hard_GCP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_1 --data_dir ../Data/finetune_data/test_1/GCP/
python run_hard_TSP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_1 --data_dir ../Data/finetune_data/test_1/TSP/

###### finetune on two benchmark ######
python run_p_BSP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_2
python run_p_EDP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_2
python run_p_SPP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_2
python run_cmp_GCP_D.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_2
python run_cmp_KSP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_2
python run_cmp_TSP_D.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_2
python run_hard_GCP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_2
python run_hard_TSP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_2

python run_p_BSP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_2 --data_dir ../Data/finetune_data/test_1/BSP/
python run_p_EDP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_2 --data_dir ../Data/finetune_data/test_1/EDP/
python run_p_SPP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_2 --data_dir ../Data/finetune_data/test_1/SPP/
python run_cmp_GCP_D.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_2 --data_dir ../Data/finetune_data/test_1/GCP_Decision/
python run_cmp_KSP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_2 --data_dir ../Data/finetune_data/test_1/KSP/
python run_cmp_TSP_D.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_2 --data_dir ../Data/finetune_data/test_1/TSP_Decision/
python run_hard_GCP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_2 --data_dir ../Data/finetune_data/test_1/GCP/
python run_hard_TSP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_2 --data_dir ../Data/finetune_data/test_1/TSP/

###### finetune on three benchmark ######
python run_p_BSP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_3
python run_p_EDP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_3
python run_p_SPP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_3
python run_cmp_GCP_D.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_3
python run_cmp_KSP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_3
python run_cmp_TSP_D.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_3
python run_hard_GCP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_3
python run_hard_TSP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_3

python run_p_BSP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_3 --data_dir ../Data/finetune_data/test_1/BSP/
python run_p_EDP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_3 --data_dir ../Data/finetune_data/test_1/EDP/
python run_p_SPP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_3 --data_dir ../Data/finetune_data/test_1/SPP/
python run_cmp_GCP_D.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_3 --data_dir ../Data/finetune_data/test_1/GCP_Decision/
python run_cmp_KSP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_3 --data_dir ../Data/finetune_data/test_1/KSP/
python run_cmp_TSP_D.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_3 --data_dir ../Data/finetune_data/test_1/TSP_Decision/
python run_hard_GCP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_3 --data_dir ../Data/finetune_data/test_1/GCP/
python run_hard_TSP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_3 --data_dir ../Data/finetune_data/test_1/TSP/

###### finetune on four benchmark ######
python run_p_BSP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_4
python run_p_EDP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_4
python run_p_SPP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_4
python run_cmp_GCP_D.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_4
python run_cmp_KSP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_4
python run_cmp_TSP_D.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_4
python run_hard_GCP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_4
python run_hard_TSP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_4

python run_p_BSP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_4 --data_dir ../Data/finetune_data/test_1/BSP/
python run_p_EDP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_4 --data_dir ../Data/finetune_data/test_1/EDP/
python run_p_SPP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_4 --data_dir ../Data/finetune_data/test_1/SPP/
python run_cmp_GCP_D.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_4 --data_dir ../Data/finetune_data/test_1/GCP_Decision/
python run_cmp_KSP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_4 --data_dir ../Data/finetune_data/test_1/KSP/
python run_cmp_TSP_D.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_4 --data_dir ../Data/finetune_data/test_1/TSP_Decision/
python run_hard_GCP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_4 --data_dir ../Data/finetune_data/test_1/GCP/
python run_hard_TSP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_4 --data_dir ../Data/finetune_data/test_1/TSP/

###### finetune on five benchmark ######
python run_p_BSP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_5
python run_p_EDP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_5
python run_p_SPP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_5
python run_cmp_GCP_D.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_5
python run_cmp_KSP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_5
python run_cmp_TSP_D.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_5
python run_hard_GCP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_5
python run_hard_TSP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_5

python run_p_BSP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_5 --data_dir ../Data/finetune_data/test_1/BSP/
python run_p_EDP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_5 --data_dir ../Data/finetune_data/test_1/EDP/
python run_p_SPP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_5 --data_dir ../Data/finetune_data/test_1/SPP/
python run_cmp_GCP_D.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_5 --data_dir ../Data/finetune_data/test_1/GCP_Decision/
python run_cmp_KSP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_5 --data_dir ../Data/finetune_data/test_1/KSP/
python run_cmp_TSP_D.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_5 --data_dir ../Data/finetune_data/test_1/TSP_Decision/
python run_hard_GCP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_5 --data_dir ../Data/finetune_data/test_1/GCP/
python run_hard_TSP.py qwen --tuned_model_dir ../../whole_model/Qwen-14B-Chat_5 --data_dir ../Data/finetune_data/test_1/TSP/


###### phi-2 ######
###### finetune on five benchmark ######
python run_p_BSP.py phi-2 --tuned_model_dir ../../whole_model/phi-2_5
python run_p_EDP.py phi-2 --tuned_model_dir ../../whole_model/phi-2_5
python run_p_SPP.py phi-2 --tuned_model_dir ../../whole_model/phi-2_5
python run_cmp_GCP_D.py phi-2 --tuned_model_dir ../../whole_model/phi-2_5
python run_cmp_KSP.py phi-2 --tuned_model_dir ../../whole_model/phi-2_5
python run_cmp_TSP_D.py phi-2 --tuned_model_dir ../../whole_model/phi-2_5
python run_hard_GCP.py phi-2 --tuned_model_dir ../../whole_model/phi-2_5
python run_hard_TSP.py phi-2 --tuned_model_dir ../../whole_model/phi-2_5

python run_p_BSP.py phi-2 --tuned_model_dir ../../whole_model/phi-2_5 --data_dir ../Data/finetune_data/test_1/BSP/
python run_p_EDP.py phi-2 --tuned_model_dir ../../whole_model/phi-2_5 --data_dir ../Data/finetune_data/test_1/EDP/
python run_p_SPP.py phi-2 --tuned_model_dir ../../whole_model/phi-2_5 --data_dir ../Data/finetune_data/test_1/SPP/
python run_cmp_GCP_D.py phi-2 --tuned_model_dir ../../whole_model/phi-2_5 --data_dir ../Data/finetune_data/test_1/GCP_Decision/
python run_cmp_KSP.py phi-2 --tuned_model_dir ../../whole_model/phi-2_5 --data_dir ../Data/finetune_data/test_1/KSP/
python run_cmp_TSP_D.py phi-2 --tuned_model_dir ../../whole_model/phi-2_5 --data_dir ../Data/finetune_data/test_1/TSP_Decision/
python run_hard_GCP.py phi-2 --tuned_model_dir ../../whole_model/phi-2_5 --data_dir ../Data/finetune_data/test_1/GCP/
python run_hard_TSP.py phi-2 --tuned_model_dir ../../whole_model/phi-2_5 --data_dir ../Data/finetune_data/test_1/TSP/

