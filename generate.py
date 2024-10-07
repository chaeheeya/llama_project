import os
import sys

import json
from tqdm import tqdm
import datetime
from pytz import timezone

import fire

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from datasets import load_dataset
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from utils.callbacks import Stream, Iteratorize
from utils.prompter import Prompter


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def test(
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights: str = "/home/user/chaehee/llama_project/output",
    prompt_template: str = "prompt",  # The prompt template to use, will default to alpaca.
    server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
    share_gradio: bool = False,

    test_on_inputs: bool = False,
    add_eos_token: bool = False,
    test_data: str = "", # data path 입력
    batch_size: int = 0,

    # generation config
    temperature: float = 0.1,
    top_p: float = 0.75,
    top_k: int = 1,
    num_beams: int= 0,
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='meta-llama/Llama-2-7b-chat-hf'"

    # 프롬프트 
    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    
    # 모델 로드
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map={"": device},
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
            device_map={"": device},
        )
    # elif device == "mps":
    #     model = LlamaForCausalLM.from_pretrained(
    #         base_model,
    #         device_map={"": device},
    #         torch_dtype=torch.float16,
    #     )
    #     model = PeftModel.from_pretrained(
    #         model,
    #         lora_weights,
    #         device_map={"": device},
    #         torch_dtype=torch.float16,
    #     )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )
    print(device)

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)

    
    def test_data_load(test_data_path):
        print(test_data_path,"입니다용요요요ㅛ요요요용")
        test_data = json.load(open("/home/user/chaehee/llama_project/data/test_dataset.json", 'r', encoding='utf-8'))
        # test_data = load_dataset("json", data_files='/home/user/chaehee/llama_project/data/test_dataset.json')
        # print(test_data[0])
        return test_data
    
    
    def gen_prompt(test_dataset, batch_size, iteration): # 배치별로 프롬프트 만들어주기
        # instruction = test_dataset['instruction']
        # input = test_dataset['input']
        # label = test_dataset['output']

        # prompt 생성하기 & 모델에 들어갈 입력 만들어주기
        start_index = batch_size * iteration

        if len(test_dataset)-start_index < batch_size:
            end_index = len(test_dataset)
        
        else:
            end_index = start_index + batch_size

        
        batch = test_dataset[start_index:end_index]
        
        instructions = [i['instruction']for i in batch]
        input_prompt = [i['input']for i in batch]
        labels = [i['output']for i in batch]
        
        prompts = []
        
        for data in batch:
            instruction = data['instruction']
            inputs = data['input']
            prompt = prompter.generate_prompt(instruction, inputs)
            prompts.append(prompt)
            
        # 프롬프트 전체 토큰화
        tokenizer.padding_side='left'
        inputs_token = tokenizer(prompts, padding = True, return_tensors="pt")
        input_ids = inputs_token["input_ids"].to(device)
        attention_mask = inputs_token["attention_mask"].to(device)
        print(inputs_token["input_ids"].size())

        return instructions, input_prompt, input_ids, attention_mask, labels
    
    
    def model_generation(input_ids,
                        attention_mask,
                        model,
                        temperature=0.7,
                        top_p=0.75,
                        top_k=3,
                        num_beams=2,
                        max_new_tokens=256,
                        repetition_penalty=4.8,
                        stream_output=False,
                        **kwargs,):
        
        # GenerationConfig 설정
            generation_config = GenerationConfig(
                do_sample = True, # greedy search: False 
                num_beams=num_beams,
                temperature=temperature,
                # top_p=top_p,
                top_k=top_k,

                # repetition_penalty=float(repetition_penalty),
                **kwargs,
            )

            generate_params = {
                "input_ids": input_ids,
                "generation_config": generation_config,
                "return_dict_in_generate": True,
                "output_scores": True,
                "max_new_tokens": max_new_tokens,
        }
            
            with torch.no_grad():
                generation_output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=max_new_tokens,
                )

                # print(generation_output.sequences)
                # print(generation_output.sequences[0])

            # print("generation output: ", len(generation_output.sequences), "\n\n\n")
            generated_output = tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=True) # 텐서 (배치사이즈 * 시퀀스길이)
            # print("Batch size:", len(generated_output))  # 배치 크기 확인

            responses = [prompter.get_response(i) for i in generated_output]

            return responses
    

    def calculate_hit(output, label):
        return ' '.join(output.strip().lower().split()) == ' '.join(label.strip().lower().split())
        
        

    def write_log(instruction, input, label, response, is_hit, log_data):
        # 로그 데이터 저장
        log_data.append({
            "instruction": instruction,
            "input": input,
            "topic_item": label,
            "generated_output": response,
            "is_hit": is_hit
        })

        return log_data
    

    if test_data:
        hit = 0
        total = 0
        log_data = []

        
        # 데이터 로드
        test_dataset = test_data_load(test_data) # 텐서로 바뀜
        testset_len = len(test_dataset)

        dataset_size = len(test_dataset)

        iteration = 0
        
        for __ in tqdm(range(0, dataset_size, batch_size),  bar_format='{percentage:3.0f} % | {bar:23} {r_bar}'):

            # 배치 생성 및 프롬프트 만들기
            instructions, inputs, input_ids, attention_mask, labels = gen_prompt(test_dataset, batch_size, iteration) # 배치 내 데이터에 대해서 instruction, input, label 모아둔것
            # Generation
            responses = model_generation(input_ids, attention_mask, model)

            iteration += 1


            for idx, response in enumerate(responses):
                instruction = instructions[idx]
                input = inputs[idx]
                response = responses[idx]
                label = labels[idx]

                # hit 계산
                is_hit = calculate_hit(response, label)
                if is_hit:
                    hit += 1
                total += 1

                # log 기록
                log_data = write_log(instruction, input, label, response, is_hit, log_data)
            
            
        # log 데이터 저장
        mdhm = str(datetime.datetime.now(timezone('Asia/Seoul')).strftime('%m-%d %H%M%S'))
        log_name = mdhm + '_' + 'd2i.json'
        log_path = os.path.join('/home/user/chaehee/llama_project/log/', log_name)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=4)

        # hit score 계산
        hit_score = hit / total
        print(f"Hit score: {hit_score * 100:.4f}%")
                

       
    
    # def evaluate(
    #     instruction,
    #     input=None,
    #     temperature=0.1,
    #     top_p=0.75,
    #     top_k=40,
    #     num_beams=4,
    #     max_new_tokens=256,
    #     repetition_penalty=4.8,
    #     stream_output=False,
    #     **kwargs,
    # ):
    #     prompt = prompter.generate_prompt(instruction, input)
    #     inputs = tokenizer(prompt, return_tensors="pt")
    #     input_ids = inputs["input_ids"].to(device)
    #     generation_config = GenerationConfig(
    #         temperature=temperature,
    #         top_p=top_p,
    #         top_k=top_k,
    #         num_beams=num_beams,
    #         repetition_penalty=float(repetition_penalty),
    #         **kwargs,
    #     )

    #     generate_params = {
    #         "input_ids": input_ids,
    #         "generation_config": generation_config,
    #         "return_dict_in_generate": True,
    #         "output_scores": True,
    #         "max_new_tokens": max_new_tokens,
    #     }

    #     if stream_output:
    #         # Stream the reply 1 token at a time.
    #         # This is based on the trick of using 'stopping_criteria' to create an iterator,
    #         # from https://github.com/oobabooga/text-generation-webui/blob/ad37f396fc8bcbab90e11ecf17c56c97bfbd4a9c/modules/text_generation.py#L216-L243.

    #         def generate_with_callback(callback=None, **kwargs):
    #             kwargs.setdefault(
    #                 "stopping_criteria", transformers.StoppingCriteriaList()
    #             )
    #             kwargs["stopping_criteria"].append(
    #                 Stream(callback_func=callback)
    #             )
    #             with torch.no_grad():
    #                 model.generate(**kwargs)

    #         def generate_with_streaming(**kwargs):
    #             return Iteratorize(
    #                 generate_with_callback, kwargs, callback=None
    #             )

    #         with generate_with_streaming(**generate_params) as generator:
    #             for output in generator:
    #                 # new_tokens = len(output) - len(input_ids[0])
    #                 decoded_output = tokenizer.decode(output)

    #                 if output[-1] in [tokenizer.eos_token_id]:
    #                     break

    #                 yield prompter.get_response(decoded_output)
    #         return  # early return for stream_output

    #     # Without streaming
    #     with torch.no_grad():
    #         generation_output = model.generate(
    #             input_ids=input_ids,
    #             generation_config=generation_config,
    #             return_dict_in_generate=True,
    #             output_scores=True,
    #             max_new_tokens=max_new_tokens,
    #         )
    #     s = generation_output.sequences[0]
    #     output = tokenizer.decode(s)
    #     yield prompter.get_response(output)




if __name__ == "__main__":
    fire.Fire(test)
