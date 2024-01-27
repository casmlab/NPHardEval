from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel,
    PeftConfig,
)
import sys
import os
from os.path import isfile, join
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json 
import argparse 
import time
import pandas as pd
import random

import torch
import math
import transformers
from accelerate import load_checkpoint_and_dispatch, init_empty_weights
import bitsandbytes as bnb
from peft.tuners.lora import LoraLayer
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaTokenizer,
    GenerationConfig,
    GPTJForCausalLM,
    T5Tokenizer,
    T5ForConditionalGeneration,
    pipeline,
    StoppingCriteria,
    StoppingCriteriaList,
    BitsAndBytesConfig,
)

from prompts import (edpPrompts, 
                     bspPrompts, 
                     gcp_dPrompts, 
                     kspPrompts, 
                     tsp_dPrompts, 
                     gcpPrompts, 
                     tspPrompts, 
                     sppPrompts)


############# util function #############
def set_seed(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--data_dir', type=str, default='../Data/finetune_data/')
    parser.add_argument("--data_up_to", type=int, default=1)
    parser.add_argument('--model', type=str, default='microsoft/phi-2', help='mistralai/Mistral-7B-Instruct-v0.1, lmsys/vicuna-13b-v1.3, Qwen/Qwen-14B-Chat')
    parser.add_argument('--cache_dir', type=str, default='../../safeAGI/safeAGI/running/pretrained_models')

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--warmup_proportion", type=float, default=0.03)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--logging_steps", type=int, default=10)

    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    parser.add_argument('--adapter_model_output_dir', type=str, default='../../adapter')
    parser.add_argument('--whole_model_output_dir', type=str, default='../../whole_model')

    args = parser.parse_args()

    return args

############# training data #############
def prompt_construction(task, q):
    if task == 'BSP':
        target_value = q['target']
        # TO-DO: fix data not being sorted
        array = sorted(q['array'])
        prompt_text = bspPrompts['Intro'] + '\n' + \
                    bspPrompts['Initial_question'].format(target_value=target_value) + '\n' + \
                    bspPrompts['Finetune_content'] + '\n' + \
                    bspPrompts['Finetune_format'] + \
                    '\n The sorted array elements are: ' + ', '.join(map(str, array)) + '\n'
    elif task == 'EDP':
        string_a = q['string_a']
        string_b = q['string_b']
        prompt_text = edpPrompts['Intro'] + '\n' + \
                    edpPrompts['Initial_question'].format(string_a=string_a, string_b=string_b) + '\n' + \
                    edpPrompts['Finetune_content'] + '\n' + \
                    edpPrompts['Finetune_format']
        prompt_text += 'Answer:\n'
    elif task == 'SPP':
        start_node = q['nodes'][0]
        end_node = q['nodes'][-1]
        edges = q['edges']
        prompt_text = sppPrompts['Intro'] + '\n' + \
                    sppPrompts['Initial_question'].format(start_node=start_node, end_node=end_node) + '\n' + \
                    sppPrompts['Finetune_content'] + '\n' + \
                    sppPrompts['Finetune_format'] + \
                    "\n The graph's edges and weights are as follows: \n"
        for edge in edges:
            this_line = f"Edge from {edge['from']} to {edge['to']} has a weight of {edge['weight']}."
            prompt_text += this_line + '\n'
        prompt_text += 'Answer:\n'
    elif task == 'GCP_Decision':
        number_of_colors = q.split('\n')[0].split()[-2] # last character of the first line
        number_of_vertices = q.split('\n')[1].split(' ')[2] # third word of the second line
        prompt_text = gcp_dPrompts['Intro'] + '\n' + \
                    gcp_dPrompts['Initial_question'].format(total_vertices=number_of_vertices, number_of_colors=number_of_colors) + '\n' + \
                    gcp_dPrompts['Finetune_content'] + '\n' + \
                    gcp_dPrompts['Finetune_format'] + '\n' + \
                        '\n The graph is below: \n'
        for line in q.split('\n')[2:]:
            vertex_list = line.split(' ')
            this_line = "Vertex {} is connected to vertex {}.".format(vertex_list[1],vertex_list[2])
            prompt_text += this_line + '\n'
    elif task == 'TSP_Decision':
        threshold = q.iloc[-1, 0] # therashold is the last row
        adj_matrix = q.iloc[:-1].values # distance matrix is the rest of the rows
        total_cities = adj_matrix.shape[0] # exclude the last row
        prompt_text = tsp_dPrompts['Intro'] + '\n' + \
                    tsp_dPrompts['Initial_question'].format(total_cities=total_cities, distance_limit=threshold) + '\n' + \
                    tsp_dPrompts['Finetune_content'] + '\n' + \
                    tsp_dPrompts['Finetune_format'] + '\n' + \
                    'The distances between cities are below: \n'
        for i in range(adj_matrix.shape[0]):
            for j in range(adj_matrix.shape[1]):
                if i < j:  # only use the upper triangle
                    this_line = "The distance between City {} and City {} is {}.".format(i, j, adj_matrix[i, j])
                    prompt_text += this_line + '\n'
    elif task == 'KSP':
        knapsack_capacity = q['knapsack_capacity']
        items = q['items']
        prompt_text = kspPrompts['Intro'] + '\n' + \
                    kspPrompts['Initial_question'].format(knapsack_capacity=knapsack_capacity) + '\n' + \
                    kspPrompts['Finetune_content'] + '\n' + \
                    kspPrompts['Finetune_format'] + \
                    '\n The items details are as below: \n'
        for item in items:
            this_line = f"Item {item['id']} has weight {item['weight']} and value {item['value']}."
            prompt_text += this_line + '\n'
    elif task == 'GCP':
        chromatic_number = q.split('\n')[0][-1] # last character of the first line
        number_of_vertices = q.split('\n')[1].split(' ')[2] # third word of the second line
        prompt_text = gcpPrompts['Intro'] + '\n' \
            + gcpPrompts['Initial_question'].format(max_vertices=number_of_vertices,max_colors=chromatic_number) + '\n' \
            + gcpPrompts['Finetune_content'] + '\n' \
            + gcpPrompts['Finetune_format'] + \
            '\n The graph is below: \n'
        for line in q.split('\n')[2:]:
            vertex_list = line.split(' ')
            this_line = "Vertex {} is connected to vertex {}.".format(vertex_list[1],vertex_list[2])
            prompt_text += this_line + '\n'
    elif task == 'TSP':
        total_cities = q.shape[0]
        prompt_text = tspPrompts['Intro'] + '\n' \
            + tspPrompts['Initial_question'].format(total_cities=total_cities) + '\n' \
            + tspPrompts['Finetune_content'] + '\n' \
            + tspPrompts['Finetune_format'] + \
            '\n The distances between cities are below: \n'
        for i in range(q.shape[0]):
            for j in range(q.shape[1]):
                if i < j: # only use the upper triangle
                    this_line = "The path between City {} and City {} is with distance {}.".format(i,j,q.iloc[i,j])
                    prompt_text += this_line + '\n'
    return prompt_text

def load_data(data_dir, task):
    if task in ['BSP', 'EDP', 'SPP', 'KSP']:
        with open(join(data_dir, '{}_instances.json'.format(task.lower())), 'r') as f:
            questions = json.load(f)
        return questions
    elif task == 'GCP_Decision':
        questions = []
        for file_num in range(1,11):
            with open(join(data_dir, "decision_data_GCP_{}.txt".format(file_num))) as f:
                data = f.read()
            questions += data.split('\n\n')[:-1]
        return questions 
    elif task == 'TSP_Decision':
        questions = []
        for level in range(1,11):
            for file_num in range(1,11):
                df = pd.read_csv(join(data_dir, "decision_data_TSP_level_{}_instance_{}.csv".format(level, file_num)),
                                header=None, 
                                index_col=False)
                questions.append(df)
        return questions 
    elif task == 'GCP':
        questions = []
        for file_num in range(1,11):
            with open(join(data_dir,"synthesized_data_GCP_{}.txt".format(file_num))) as f:
                data = f.read()
            questions += data.split('\n\n')[:-1]
        return questions
    elif task == 'TSP':
        questions = []
        for level in range(1,11):
            for file_num in range(1,11):
                df = pd.read_csv(join(data_dir,"synthesized_data_TSP_level_{}_instance_{}.csv".format(level,file_num)),
                                    header=None, 
                                    index_col=False)
                # transform df to 
                questions.append(df)
        return questions

def construct_training_data(args, tokenizer):
    files_dir = [join(args.data_dir, 'train_{}'.format(i)) for i in range(1, args.data_up_to + 1)]
    all_data = []
    for file_dir in files_dir:
        tasks = os.listdir(file_dir)
        # do not tune on MSP, no gold answer
        for task in [t for t in tasks if t != 'MSP']:
            task_dir = join(file_dir, task)
            questions = load_data(task_dir, task)
            question_prompts = [prompt_construction(task, q) for q in questions]
            # ../Data/finetune_data/train_x/EDP/EDP_answers.json
            with open(join(task_dir, '{}_answers.json'.format(task.lower().replace('ecision', ''))), 'r') as f:
                answers = json.load(f)
            for i in range(len(answers)):
                del answers[i]['level']
            answer_strings = ['<final_answer>'+str(a)+'</final_answer>' for a in answers]
            all_data += [question + answer for question, answer in zip(question_prompts, answer_strings)]
    
    random.shuffle(all_data)

    def load_custom_dataset(data):
        train_encodings = tokenizer(data, truncation=True, padding=True)

        class InputDataset(torch.utils.data.Dataset):
            def __init__(self, encodings):
                self.encodings = encodings

            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                return item

            def __len__(self):
                return len(self.encodings["input_ids"])

        train_dataset = InputDataset(train_encodings)

        return train_dataset
    
    return load_custom_dataset(all_data)
            
############# training helper function #############
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def find_all_linear_names(args, model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)

def training_steps(args, training_dataset_length):
    num_gpus = torch.cuda.device_count()
    training_steps = int(
        math.ceil(
            training_dataset_length
            / (args.gradient_accumulation_steps * args.per_device_train_batch_size)
        )
        * args.epochs
    )
    warmup_steps = int(math.ceil(training_steps * args.warmup_proportion))

    return training_steps, warmup_steps

def load_4bit_model(args, model_name):
    config = AutoConfig.from_pretrained(model_name, torch_dtype=torch.float16,trust_remote_code=True)
    config.tie_word_embeddings = True

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config,trust_remote_code=True)
        model.tie_weights()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=nf4_config,
        device_map="auto",
        offload_state_dict=True,
        cache_dir=args.cache_dir,
        trust_remote_code=True
    )
    
    if 'Qwen' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True, pad_token='<|endoftext|>',cache_dir=args.cache_dir)
    elif 'Mistral' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=args.cache_dir, trust_remote_code=True)
        tokenizer.pad_token_id = tokenizer.eos_token_id
    elif 'vicuna' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=args.cache_dir, trust_remote_code=True)
        tokenizer.pad_token = '</s>'
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir=args.cache_dir, trust_remote_code=True)
        model.config.pad_token_id = tokenizer.pad_token_id = 0
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config_eos_token_id = tokenizer.eos_token_id

    return model, tokenizer

if __name__ == '__main__':
    args = parser()

    print("loading model ...")
    model, tokenizer = load_4bit_model(args, args.model)

    try:
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
    except:
        print("gradient checkpointing not supported for model {}".format(args.model))

    modules = find_all_linear_names(args, model)
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            module = module.to(torch.float32)
        if "norm" in name:
            module = module.to(torch.float32)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                if module.weight.dtype == torch.float32:
                    module = module.to(torch.float32)

    print("model loaded.")

    train_dataset = construct_training_data(args, tokenizer)

    training_steps, warmup_steps = training_steps(args, len(train_dataset))
    print(
        """
length of training dataset: {}
number of training steps: {}
number of warmup steps: {}
    """.format(
            len(train_dataset), training_steps, warmup_steps
        )
    )

    print("start training ...")
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            max_steps=training_steps,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            fp16=True,
            logging_steps=args.logging_steps,
            output_dir="outputs",
            optim="paged_adamw_8bit",
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer, mlm=False
        ),
    )
    model.config.use_cache = (
        False  # silence the warnings. Please re-enable for inference!
    )
    trainer.train()

    adapter_model_output_dir = args.adapter_model_output_dir + '/' + args.model.split('/')[-1] + '_'+str(args.data_up_to)
    model.save_pretrained(adapter_model_output_dir)
    print("adapter model saved.")

    base_model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", cache_dir=args.cache_dir, trust_remote_code=True)
    model = PeftModel.from_pretrained(base_model, adapter_model_output_dir)
    model = model.merge_and_unload()

    whole_model_output_dir = args.whole_model_output_dir + '/' + args.model.split('/')[-1] + '_'+str(args.data_up_to)
    model.save_pretrained(whole_model_output_dir)
    tokenizer.save_pretrained(whole_model_output_dir)
    print("whole model and tokenizer saved.")