from math import exp, log
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

from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig

from utils.callbacks import Stream, Iteratorize
from utils.prompter import Prompter
from utils.args import parse_args
from utils.dataset import Test_CRS_Dataset

from nltk.translate.bleu_score import sentence_bleu
from transformers import DataCollatorForSeq2Seq
from collections import defaultdict

# device 설정
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


# def prompter_tokenizer_loader(base_model: str, prompt_template: str):
#     prompter = Prompter(prompt_template)
#     tokenizer = LlamaTokenizer.from_pretrained(base_model)
#     tokenizer.padding_side = 'left'
#     return prompter, tokenizer

# 프롬프트, 토크나이저 로드
# tokenizer = prompter_tokenizer_loader(args.base_model, args.prompt_template)


def test_data_load(test_data_path):
        print("test datasdt loading: ", test_data_path)
        test_data = json.load(open(test_data_path, 'r', encoding='utf-8'))
        return test_data
    
    
def gen_prompt(tokenizer, prompter, data): # 배치별로 프롬프트 만들어주기
    # instruction = test_dataset['instruction']
    # input = test_dataset['input']
    # label = test_dataset['output']

    # print("\nDataset 길이: ", len(test_dataset), "\n")
    # # prompt 생성하기 & 모델에 들어갈 입력 만들어주기
    # start_index = batch_size * iteration
    # if len(test_dataset)-start_index < batch_size:
    #     end_index = len(test_dataset)
    # else:
    #     end_index = start_index + batch_size
    # print("slicing index: ",start_index, end_index)

    # batch = test_dataset[start_index:end_index]
    # # print("batch: ", batch)
    
    dialogs = [i for i in data['dialog']] # 배치 내의 샘플 instruction 저장
    # input_prompts = [i for i in data['input']] # 배치 내의 샘플 input 저장
    full_prompt = [i for i in data['full_prompt']] # 배치 내의 label 저장
    
    prompts = []
    for dialog in dialogs:
        tokenizing_dialog = tokenizer(dialog).input_ids[-410:]
        dialog = tokenizer.decode(tokenizing_dialog)
    
        prompt = prompter.generate_prompt(dialog)
        prompts.append(prompt)
        
       
    # prompts에 저장된 배치 프롬프트 전체 토큰화
    tokenizer.padding_side='left'
    inputs_token = tokenizer(prompts, padding = True, return_tensors="pt")
    input_ids = inputs_token["input_ids"].to(device)
    attention_mask = inputs_token["attention_mask"].to(device)
    print(inputs_token["input_ids"].size())

    return dialogs, input_ids, attention_mask, full_prompt


def model_generation(args,
                     input_ids,
                     attention_mask,
                     model,
                     tokenizer,
                     prompter, 
                     repetition_penalty=4.8,
                     stream_output=False,
                     **kwargs,):
     
    # GenerationConfig 설정
        generation_config = GenerationConfig(
            do_sample = args.do_sample, # greedy search: False 
            num_beams=args.num_beams,
            num_return_sequences = args.num_return_sequences,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            # repetition_penalty=float(repetition_penalty),
            **kwargs,
        )

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=args.max_new_tokens,
            )

            # print(generation_output.sequences)
            # print(generation_output.sequences[0])

        # print("generation output: ", len(generation_output.sequences), "\n\n\n")
        generated_output = tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=True) # 텐서 (배치사이즈 * 시퀀스길이)
        # print("Batch size:", len(generated_output))  # 배치 크기 확인

        # print(generated_output)
        
        # 배치 내 데이터에 대해 생성된 응답 한개씩 저장
        responses_all = [prompter.get_response(i) for i in generated_output]

        responses = [] # 데이터에 대한 생성 결과 num_return_sequences 개수만큼 리스트에 저장
        for idx in range(input_ids.size(0)):
            s_index = idx*args.num_return_sequences
            e_index = s_index+args.num_return_sequences
            # print("generation 결과:", responses_all[s_index:e_index])
            responses.append(responses_all[s_index:e_index]) # response 각각이 리스트로 저장됨 (이중리스트) -> responses = [[response] * len(responses_all)]

        return responses


              
def hit_score_cal(outputs, label, topk): # 하나의 샘플에 대해서 계산산
    hit_topk = [0] * len(topk)
    for idx, k in enumerate(topk):
        hit_topk[idx] = 1 if label in outputs[:k] else 0
    return hit_topk

        
def n_gram_cal(output, label, n):
    output = ','.join(str(s) for s in output) # 리스트 -> str로 변환
    candidate_list = list(output.split())
    reference_list = list(label.split())

    n_gram_list_candidiate = []
    n_gram_list_reference= []

    # candidate list와 reference list n-gram 덩어리 만들기
    for idx in range(0, len(candidate_list) - n + 1):
        start = idx
        end = idx + n
        n_gram = candidate_list[start:end]
        n_gram_list_candidiate.append(tuple(n_gram))  # 튜플로 변환
        # print("candidate_n_gram:", n_gram_list_candidiate)

    for idx in range(0, len(reference_list) - n + 1):
        start = idx
        end = idx + n
        n_gram = reference_list[start:end]
        n_gram_list_reference.append(tuple(n_gram))
        # print("reference_n_gram:", n_gram_list_reference)

    # reference에서 각 단어들이 몇번 등장하는지 계산
    reference_dict = {}
    for word in n_gram_list_reference:
        counting: int = 0
        for element in n_gram_list_reference:
            if word == element:
                counting += 1
        reference_dict[word] = counting
    # print(reference_dict)

    # candidate 문장에서 각 단어가 몇번 등장했는지 카운트
    candidate_dict = {}
    for word in n_gram_list_candidiate:
        counting: int = 0
        for element in n_gram_list_candidiate:
            if word == element:
                counting += 1
        candidate_dict[word] = counting
    # print(candidate_dict)

    # candidate value의 값들 구하기
    candidate_sum = 0
    for value in list(candidate_dict.values()):
        candidate_sum += value


    # Modified n-gram Precision 구하기
    
    per_precision = 0
    for key in candidate_dict.keys():
        if key not in reference_dict.keys():
            per_precision += 0

        else:
            if candidate_dict[key] > reference_dict[key]:
                # print(reference_dict[key])
                per_precision += reference_dict[key]
                # precision += per_precision
            else:
                # print(candidate_dict[key])
                per_precision += candidate_dict[key]
                # precision += per_precision

    print(f"{n}_gram per_precison: {per_precision}, {candidate_sum}")
    return per_precision, candidate_sum

    # precision = 0.0
    # for key in candidate_dict.keys():
    #     if key not in reference_dict.keys():
    #         per_precision = 0.0

    #     else:
    #         if candidate_dict[key] > reference_dict[key]:
    #             # print(reference_dict[key])
    #             per_precision = reference_dict[key] 
    #             precision += per_precision
    #         else:
    #             # print(candidate_dict[key])
    #             per_precision = candidate_dict[key] 
    #             precision += per_precision

    # print(f"{n}_gram precison: {precision * 100:.2f}%")
    # return precision, candidate_sum
    


def test(
    args,
    base_model: str = "",
    lora_weights: str = "",
    prompt_template: str = "",  
    
    test_data: str = "", # data path 입력
    batch_size: int = 0,

    # generation config
    temperature: float = 0.0,
    top_p: float = 0.0,
    top_k: int = 0,
    num_beams: int= 0,
    
):
    print("argument: ", base_model, lora_weights, prompt_template, test_data, batch_size, num_beams)
    prompter = Prompter(args.prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
    
    tokenizer.padding_side = 'left'
    
    # 모델 입력
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='meta-llama/Llama-2-7b-chat-hf'"
    
    # 모델 로드
    load_8bit = False,
    if device == "cuda":
        quantization_config = BitsAndBytesConfig(
        load_in_8bit=True
    )
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            quantization_config = quantization_config,
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
        model.half()
        #model.bfloat16() #model.half()  # seems to fix bugs for some users.

    model.eval()
    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)

    
    if test_data:
        total = 0
        # hit score 계산
        topk=[1,3,5,10]
        hit_topk_total = [0] * len(topk)
        
        # bleu score 계산
        one_gram = 0.0
        two_gram = 0.0
        three_gram = 0.0
        four_gram = 0.0
        
        one_precision = 0.0
        two_precision = 0.0
        three_precision = 0.0
        four_precision = 0.0
        
        # log 저장할 파일 만들기
        md = str(datetime.datetime.now(timezone('Asia/Seoul')).strftime('%y%m%d'))
        md_dirs = os.makedirs(os.path.join('/home/user/chaehee/prefer_optim/llama/log/', md), exist_ok=True)
        
        # mdhm = str(datetime.datetime.now(timezone('Asia/Seoul')).strftime('%y%m%d-%H%M'))
        if args.log_name == '':
            if args.task == 'DIP2I':
                log_name = args.lora_weights + '_beam' + str(args.num_beams) + '.json'
            elif args.task == 'd2i':    
                log_name = args.lora_weights + '.json'           
        else:
            log_name = args.log_name
                
        args.log_file = open(os.path.join('/home/user/chaehee/prefer_optim/llama/log/', md, log_name), 'a', buffering=1, encoding='UTF-8')

        # 데이터 로드
        test_dataset = Test_CRS_Dataset(args, test_data, tokenizer, prompter)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_dataset.my_collate_fn)


        #  배치 크기별로 실행
        for data in tqdm(test_loader, bar_format='{percentage:3.0f} % | {bar:23} {r_bar}'):
            dialogs = data['instructions'] # don't touch (collate에 instructions라고 설정됨)
            inputs = data['inputs']
            topic_items = data['topics'] # don't touch (collate에 topics라고 설정됨)
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            
            # 배치 크기 만큼의 Generation -> output: (이중 list) [batch_size * [num_return_sequences]]
            responses = model_generation(args, input_ids, attention_mask, model, tokenizer, prompter, batch_size)
            
            # num_return_sequences개 생성, responses에서 하나의 데이터에 대한 response 추출하기 
            for idx, response in enumerate(responses):
                # instruction = dialogs[idx]
                topic_item = topic_items[idx]
                # input = inputs[idx]
                if "i" in args.task or "I" in args.task: # 아이템 맞추는 경우에 hit score 계산하도록 하기
                    hit_topk = hit_score_cal(response, topic_item, topk) # 각 샘플에 대한 hit@1, hit@2, hit@3를 계산
                    total+=1
                    
                    # 누적 hit@1, hit@2, hit@3
                    hit_topk_total = [i+j for i,j in zip(hit_topk_total, hit_topk)]
                    hit_topk_avg = [sum / total for sum in hit_topk_total]

                    hit_score = " | ".join(["Hit@%d_score:%.5f" % (topk[idx], i) for idx, i in enumerate(hit_topk_avg)])
                    # print(hit_score)
                                        
                    # hit score 기록
                    args.log_file.write(json.dumps(
                        {
                            "instruction:": tokenizer.decode(input_ids[idx], skip_special_tokens=True),
                            "topic_item": topic_item,
                            "generated_output": response,
                            "hit_score": hit_score
                        }, ensure_ascii=False, indent=4) + '\n')
                
                    
                # print(f"Hit score: {hit_score * 100:.4f}%")

                
                if args.task=="d2r":
                    # 각각의 응답에 대한 n-gram 계산
                    print("n-gram precision 계산")
                    answer = [answer.split()] # list로 만들어줘야함
                    total += 1
                    response = response[0].split() # ''.join(str(s) for s in response).split() # str 
                    # print(answer, response)
                    
                    
                    one = sentence_bleu(answer, response, weights=(1, 0, 0, 0))
                    one_precision += one
                    one_gram = one_precision/total
                    
                    two = sentence_bleu(answer, response, weights=(0, 1, 0, 0))
                    two_precision += two
                    two_gram = two_precision/total
                    
                    three = sentence_bleu(answer, response, weights=(0, 0, 1, 0))
                    three_precision += three
                    three_gram = three_precision/total
                    
                    four = sentence_bleu(answer, response, weights=(0, 0, 0, 1))
                    four_precision += four
                    four_gram = four_precision/total
                    
                    n_gram = f'{one_gram:.3f} | {two_gram:.3f} | {three_gram:.3f} | {four_gram:.3f}'                    
                    
                    # n-gram 기록
                    args.log_file.write(json.dumps(
                        {
                            "instruction:": instruction,
                         # /"input": input,
                            "topic_item": ' '.join(sum(answer, [])),
                            "generated_output": ' '.join(response),
                            "n_gram": n_gram,
                            # "BLEU": BLEU_score
                        }, ensure_ascii=False, indent=4) + '\n')



if __name__ == "__main__":
    fire.Fire(test)
