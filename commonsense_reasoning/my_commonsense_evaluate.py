# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import copy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import sys
import argparse
import time

import fire

import pandas
import torch

sys.path.append(os.path.join(os.getcwd(), "peft/src/"))
from peft import PeftModel, PeftConfig
from tqdm import tqdm
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


@dataclass
class Args:
    dataset: str = "boolq"
    base_model: str = "meta-llama/Llama-3-7B"
    lora_weights: str = "path_to_lora_weights"
    batch_size: int = 16
    load_8bit: bool = False
    output_dir: str = "sub_experiment"


def main(args: Args):
    #args = parse_args()

    def evaluate(
            instructions,
            input=None,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            max_new_tokens=32,
            **kwargs,
    ):
        prompts = [generate_prompt(instruction, input) for instruction in instructions]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences
        outputs = tokenizer.batch_decode(s, skip_special_tokens=True)
        outputs = [o.split("\n")[-1].strip() for o in outputs]
        #outputs = [o.split("### Response:")[-1].strip() for o in outputs]
        #outputs = [o.split(".")[-1].strip() for o in outputs]
        return outputs

    save_file = Path('experiment',args.output_dir ,str(args.lora_weights).split("/")[-1],f'{args.dataset}.json')
    results_file = Path('experiment',args.output_dir,str(args.lora_weights).split("/")[-1],"results.txt")
    create_dir(save_file.parent)

    dataset = load_data(args)
    batches = create_batch(dataset, args.batch_size)
    tokenizer, model = load_model(args)

    model = model.merge_and_unload()

    total = len(batches)
    correct = 0
    current = 0
    output_data = []
    pbar = tqdm(total=total)
    start_time = time.time()
    for idx, batch in enumerate(batches):
        current += len(batch)
        instructions = [data.get('instruction') for data in batch]

        outputs = evaluate(instructions)

        for data, output in zip(batch, outputs):
            label = data.get('answer')
            flag = False
            predict = extract_answer(args, output)
            if predict == "":
                predict = extract_answer2(args, output)
            if label == predict:
                correct += 1
                flag = True
            new_data = copy.deepcopy(data)
            new_data['output_pred'] = output
            new_data['pred'] = predict
            new_data['flag'] = flag
            if current < 10:
                print(data["instruction"])
                print(output)
                print('prediction:', predict)
                print('label:', label)
            output_data.append(new_data)
        print(f'\rtest:{idx + 1}/{total} | accuracy {correct}  {correct / current}')
        pbar.update(1)

    pbar.close()
    print('\n')
    accuracy = correct / current
    output_data.append({
        'accuracy': correct / current,
        'memory': torch.cuda.max_memory_allocated() /1024/1024 if torch.cuda.is_available() else 0,
        'time': time.time() - start_time
    })
    with open(save_file, 'w') as f:
        json.dump(output_data, f, indent=4)
    with open(results_file, 'a') as f:
        f.write(f'dataset: {args.dataset}, accuracy: {correct / current}, memory: {torch.cuda.max_memory_allocated() /1024/1024 if torch.cuda.is_available() else 0}, '
                f'time: {time.time() - start_time}\n')
    print('test finished')
    return accuracy , results_file


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    return

def generate_prompt(instruction, input=None):
    if input:
        return f"""{instruction}\n\n{input}\n\n""" # noqa: E501
    else:
        return f"""{instruction}\n\n""" # noqa: E501
'''

def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:
                {instruction}

                ### Input:
                {input}

                ### Response:
                """  # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

                ### Instruction:
                {instruction}

                ### Response:
                """  # noqa: E501
'''

def load_data(args) -> list:
    """
    read data from dataset file
    Args:
        args:

    Returns:

    """
    file_path = f'/work/xg24i002/x10041/LLM-Adapters/dataset/{args.dataset}/test.json'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"can not find dataset file : {file_path}")
    json_data = json.load(open(file_path, 'r'))
    return json_data

def create_batch(dataset, batch_size):
    batches = []
    num_batch = len(dataset)//batch_size if len(dataset) % batch_size == 0 else len(dataset)//batch_size + 1
    for i in range(num_batch):
        batch = dataset[i*batch_size: min((i+1)*batch_size, len(dataset))]
        batches.append(batch)
    return batches


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', required=True)
    parser.add_argument('--lora_weights', default=None, type=str)
    parser.add_argument('--lora_weights_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='sub_experiment')
    parser.add_argument('--load_8bit', action='store_true', default=False)

    return parser.parse_args()


def load_model(args) -> tuple:
    """
    load tuned model
    Args:
        args:

    Returns:
        tuple(tokenizer, model)
    """
    base_model = args.base_model
    if not base_model:
        raise ValueError(f'can not find base model name by the value: {base_model}')
    lora_weights = args.lora_weights
    if not lora_weights:
        raise ValueError(f'can not find lora weight, the value is: {lora_weights}')

    load_8bit = args.load_8bit
    
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    # Fix Llama-3.x tokenizer: add dedicated PAD token
    # Token id 0 is "!" in Llama-3 tokenizers, not a valid PAD
    # https://github.com/turboderp/exllamav2/issues/415
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == 0:
        print(f"Adding dedicated PAD token (original pad_token_id: {tokenizer.pad_token_id})")
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        print(f"New pad_token_id: {tokenizer.pad_token_id}, vocab size: {len(tokenizer)}")
    else:
        # A safer fallback is to use the EOS token if a PAD token is not defined
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    
    tokenizer.padding_side = "left"  # Allow batched inference

    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
        ) # fix zwq
        
        # Resize token embeddings to match the tokenizer's new size
        model.resize_token_embeddings(len(tokenizer))
        
        # Fix Llama-3.x tokenizer: add dedicated PAD token
        # This is a temporary workaround for the issue where the model's pad_token_id is not set
        # correctly when adding a new pad token to the tokenizer.
        if tokenizer.pad_token_id is not None and model.config.pad_token_id != tokenizer.pad_token_id:
            print(f"Updating model.config.pad_token_id from {model.config.pad_token_id} to {tokenizer.pad_token_id}")
            model.config.pad_token_id = tokenizer.pad_token_id
        
        # Fix for CorDA init error: override init_lora_weights to avoid needing eigens during inference
        peft_config = PeftConfig.from_pretrained(lora_weights)
        if getattr(peft_config, "init_lora_weights", None) == "corda":
            peft_config.init_lora_weights = True

        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            config=peft_config,
            torch_dtype=torch.float16,
            device_map={"":0}
        )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

        # unwind broken decapoda-research config
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

        if not load_8bit:
            model.half()  # seems to fix bugs for some users.

        model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)

    return tokenizer, model


def load_instruction(args) -> str:
    instruction = ''
    if not instruction:
        raise ValueError('instruct not initialized')
    return instruction

def extract_answer(args, sentence: str) -> float:
    dataset = args.dataset
    if dataset == 'boolq':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'true|false', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'piqa':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'solution1|solution2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset in ['social_i_qa', 'ARC-Challenge', 'ARC-Easy', 'openbookqa']:
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'answer1|answer2|answer3|answer4|answer5', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'hellaswag':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'ending1|ending2|ending3|ending4', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'winogrande':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'option1|option2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]

def extract_answer2(args, sentence: str) -> float:
    dataset = args.dataset
    answer = sentence.strip().split(" ")[-1]
    if dataset == 'boolq':
        if answer == "1":
           return "true"
        else:
            return "false"
    elif dataset == 'piqa':
        return "solution"+answer
    elif dataset in ['social_i_qa', 'ARC-Challenge', 'ARC-Easy', 'openbookqa']:
        return "answer"+answer
    elif dataset == 'hellaswag':
        return "ending"+answer
    elif dataset == 'winogrande':
        return "option"+answer

DATASET_CHOICES = ['piqa',"boolq", "social_i_qa", "hellaswag", "winogrande", "ARC-Challenge", "ARC-Easy", "openbookqa"]

if __name__ == "__main__":
    command_line_args = parse_args()
    if command_line_args.lora_weights_dir is None and command_line_args.lora_weights is None:
        raise ValueError("Either --lora_weights_dir or --lora_weights must be provided.")
    
    if command_line_args.lora_weights_dir is None and command_line_args.lora_weights is not None:
        # If only a single lora_weights path is provided, evaluate that directly
        for dataset_name in DATASET_CHOICES:
            args = Args(
                dataset=dataset_name,  # Default dataset, can be modified as needed
                base_model=command_line_args.base_model,
                lora_weights=command_line_args.lora_weights,
                batch_size=32,
                load_8bit=False,
                output_dir=command_line_args.output_dir
            )
            print(f"\n正在评估: {args.lora_weights}")
            main(args)
        sys.exit(0)
    # 遍历root_dir下第一层，获取以_数字结尾的文件夹

    if command_line_args.lora_weights_dir is not None:
        root_dir = Path(command_line_args.lora_weights_dir)
        import re
        pattern = re.compile(r'_\d{14}$')  # 匹配以_后跟14个数字结尾的文件夹
        
        lora_weight_dirs = []
        for item in root_dir.iterdir():
            if item.is_dir() and pattern.search(item.name):
                lora_weight_dirs.append(item.absolute())
        
        print(f"找到 {len(lora_weight_dirs)} 个符合条件的文件夹:")
        for dir_path in lora_weight_dirs:
            print(f"  {dir_path}")
        
        # 如果需要对每个文件夹执行评估，可以遍历它们
        result_data = []
        for lora_weights_path in lora_weight_dirs:
            model_name = lora_weights_path
            model_results = {'model': str(model_name).split("/")[-1]}
            
            for dataset_name in DATASET_CHOICES:
                args = Args(
                    dataset=dataset_name,  # 根据需要修改
                    base_model=command_line_args.base_model,
                    lora_weights=str(lora_weights_path),
                    batch_size=32,
                    load_8bit=False,
                    output_dir=command_line_args.output_dir
                )
                print(f"\n正在评估: {args.lora_weights} on dataset: {dataset_name}")
                acc, result_file = main(args)
                model_results[dataset_name] = acc
            
            # 计算平均准确率
            accuracies = [model_results[ds] for ds in DATASET_CHOICES]
            model_results['average'] = sum(accuracies) / len(accuracies)
            result_data.append(model_results)
        
        # 创建DataFrame并保存到CSV
        result_df = pandas.DataFrame(result_data)
        csv_output_path = Path('experiment', command_line_args.output_dir, 'all_results.csv')
        create_dir(csv_output_path.parent)
        result_df.to_csv(csv_output_path, index=False)
        print(f"\n结果已保存到: {csv_output_path}")
        print("\n汇总结果:")
        print(result_df.to_string(index=False))
            