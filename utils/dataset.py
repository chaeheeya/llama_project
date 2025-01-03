import torch
from torch.utils.data import Dataset

import json
import pickle

from collections import defaultdict
import random

"""
pytorch Dataset 클래스 상속 받아 필요한 데이터셋 만들기
"""

class Train_CRS_Dataset(Dataset):
    def __init__(self, args, data_path, tokenizer, prompter):
        # pickle / json 데이터 파일 불러오기
        self.args = args
        
        if '.pkl' in data_path:
            self.dataset = pickle.load(open(data_path, 'rb'))
        if '.json' in data_path: 
            self.dataset = json.load(open(data_path))
        
        self.tokenizer = tokenizer
        self.prompter = [prompter, prompter, prompter]
        
        self.valid_sample = []
        
        
        
        if args.dataset_split:
            for sample in self.dataset:
                if sample['topic'] in sample['gpt_validity']['valid']:
                    self.valid_sample.append(sample)
                # if sample['topic'] in sample['gpt_validity']['neutral']:
                #     self.valid_sample.append(sample)
            self.dataset = self.valid_sample
        else:
            self.dataset = self.dataset
        
        print("train dataset size: ", len(self.dataset))
            

            
        
        # 데이터셋 전처리 -> topic이 무엇인지에 따라 다름
        if args.label == "topic":
            self.dataset = self.dataset
        
        if args.label == "valid": # valid 아이템만 가지고 학습 
            remove_sample = set()  # 중복을 방지하기 위해 set 사용
            self.items = json.load(open("/home/user/chaehee/prefer_optim/data/insp2/topicDic.txt", 'r',encoding='utf-8'))
            
            # valid 항목 처리
            for idx, sample in enumerate(self.dataset):
                for item in sample['gpt_validity']['valid']:
                    if item not in self.items['str']:
                        remove_sample.add(idx)  # 샘플의 인덱스를 set에 추가

            # invalid 항목 처리
            for idx, sample in enumerate(self.dataset):
                for item in sample['gpt_validity']['invalid']:
                    if item not in self.items['str']:
                        remove_sample.add(idx)  # 샘플의 인덱스를 set에 추가

            # neutral 항목 처리
            for idx, sample in enumerate(self.dataset):
                for item in sample['gpt_validity']['neutral']:
                    if item not in self.items['str']:
                        remove_sample.add(idx)  # 샘플의 인덱스를 set에 추가

            # remove_sample에 있는 인덱스를 가진 샘플들을 한 번에 삭제
            self.dataset = [sample for idx, sample in enumerate(self.dataset) if idx not in remove_sample]
            
            # (1) valid가 없는 sample 제외
            remove_sample = [] 
            for sample in self.dataset:
                if len(sample['gpt_validity']['valid']) == 0:
                    remove_sample.append(sample)
            
            for sample in remove_sample:
                self.dataset.remove(sample)
                
            # (2) invalid가 없는 sample 제외
            remove_sample = []
            for sample in self.dataset:
                if len(sample['gpt_validity']['invalid']) == 0:
                    remove_sample.append(sample)
            
            for sample in remove_sample:
                self.dataset.remove(sample)
        
        if args.label == "mix": # valid + topic 만들어서 학습
            # remove_sample = set()  # 중복을 방지하기 위해 set 사용
            # self.items = json.load(open("/home/user/chaehee/prefer_optim/data/insp2/topicDic.txt", 'r',encoding='utf-8'))
            # # valid 항목 처리
            # for idx, sample in enumerate(self.dataset):
            #     for item in sample['gpt_validity']['valid']:
            #         if item not in self.items['str']:
            #             remove_sample.add(idx)  # 샘플의 인덱스를 set에 추가

            # # invalid 항목 처리
            # for idx, sample in enumerate(self.dataset):
            #     for item in sample['gpt_validity']['invalid']:
            #         if item not in self.items['str']:
            #             remove_sample.add(idx)  # 샘플의 인덱스를 set에 추가

            # # neutral 항목 처리
            # for idx, sample in enumerate(self.dataset):
            #     for item in sample['gpt_validity']['neutral']:
            #         if item not in self.items['str']:
            #             remove_sample.add(idx)  # 샘플의 인덱스를 set에 추가

            # # remove_sample에 있는 인덱스를 가진 샘플들을 한 번에 삭제
            # self.dataset = [sample for idx, sample in enumerate(self.dataset) if idx not in remove_sample]
            
            # (1) valid에 topic 추가 
            for sample in self.dataset:
                if sample['topic'] not in sample['gpt_validity']['valid']:
                    sample['gpt_validity']['valid'].append(sample['topic'])
                
            # # (2) invalid가 없는 sample 제외
            # remove_sample = []
            # for sample in self.dataset:
            #     if len(sample['gpt_validity']['invalid']) == 0:
            #         remove_sample.append(sample)
            
            # for sample in remove_sample:
            #     self.dataset.remove(sample)
            
        if args.label == "except_topic": # valid에 topic 제외하고 학습
            # remove_sample = set()  # 중복을 방지하기 위해 set 사용
            # self.items = json.load(open("/home/user/chaehee/prefer_optim/data/insp2/topicDic.txt", 'r',encoding='utf-8'))
            # # valid 항목 처리
            # for idx, sample in enumerate(self.dataset):
            #     for item in sample['gpt_validity']['valid']:
            #         if item not in self.items['str']:
            #             remove_sample.add(idx)  # 샘플의 인덱스를 set에 추가

            # # invalid 항목 처리
            # for idx, sample in enumerate(self.dataset):
            #     for item in sample['gpt_validity']['invalid']:
            #         if item not in self.items['str']:
            #             remove_sample.add(idx)  # 샘플의 인덱스를 set에 추가

            # # neutral 항목 처리
            # for idx, sample in enumerate(self.dataset):
            #     for item in sample['gpt_validity']['neutral']:
            #         if item not in self.items['str']:
            #             remove_sample.add(idx)  # 샘플의 인덱스를 set에 추가

            # # remove_sample에 있는 인덱스를 가진 샘플들을 한 번에 삭제
            # self.dataset = [sample for idx, sample in enumerate(self.dataset) if idx not in remove_sample]
            
            remove_sample = []
            # (1) valid에 topic 제외 
            for sample in self.dataset:
                if sample['topic'] in sample['gpt_validity']['valid']:
                    sample['gpt_validity']['valid'].remove(sample['topic'])
                    # topic 제외 후에 valid item 개수가 없는 경우 샘플 삭제
                if len(sample['gpt_validity']['valid']) == 0:
                    remove_sample.append(sample)
            
            for sample in remove_sample:
                self.dataset.remove(sample)
                
            # # (2) invalid가 없는 sample 제외
            # remove_sample = []
            # for sample in self.dataset:
            #     if len(sample['gpt_validity']['invalid']) == 0:
            #         remove_sample.append(sample)
            
            # for sample in remove_sample:
            #     self.dataset.remove(sample)
                        
                    
        
    def __len__(self):
        # 데이터 샘플 개수를 반환
        return len(self.dataset)
        
    
    def __getitem__(self, index): # 프롬프트까지 만들어서 각 샘플을 넘겨줌, index = batch_size만큼 반복됨
        data = self.dataset[index]
        instruction = data['dialog'] # 배치 내의 샘플 dialog 저장
        topic = data['topic'] # 배치 내의 topic 저장
        if 'pkl' in self.args.data:
            input = ""
        else:
            input = data['input']

        if self.args.label != "topic":
            valid = data['gpt_validity']['valid'][random.randint(0, len(data['gpt_validity']['valid'])-1)] # valid 중에서 하나 뽑기
            # invalid = data['gpt_validity']['invalid'][random.randint(0, len(data['gpt_validity']['invalid'])-1)] # invalid 중에서 하나 뽑기
        
        tokenizing_instruction = self.tokenizer(instruction).input_ids[-self.args.cutoff_len:] #160
        instruction = self.tokenizer.decode(tokenizing_instruction, skip_special_tokens=True)
        prompt_template_idx = random.randrange(len(self.prompter))
        
        if self.args.mode == 'train':
            if self.args.label == 'topic':
                prompt = self.prompter[prompt_template_idx].generate_prompt(instruction, input, topic)
            if self.args.label == 'valid' or self.args.label == 'mix' or self.args.label == 'except_topic':
                prompt = self.prompter[prompt_template_idx].generate_prompt(instruction, input, valid)
        else:
            prompt = self.prompter[prompt_template_idx].generate_prompt(instruction)
        
        inputs_token = self.tokenizer(prompt)
        input_ids = inputs_token["input_ids"]
        attention_mask = inputs_token["attention_mask"]
        
        # eos token 프롬프트에 추가하기
        if self.args.mode == 'train':
            input_ids.append(self.tokenizer.eos_token_id)
            attention_mask.append(1)
        
        labels = input_ids.copy()
        
        if self.args.train_only_label:
            user_prompt = self.prompter[prompt_template_idx].generate_prompt(instruction, input)
            tokenized_user_prompt = self.tokenizer(user_prompt)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            # user_prompt_len
            labels = [-100] * user_prompt_len + labels[user_prompt_len:]  # could be sped up, probably
        
        return {'input_ids': input_ids, 
                'attention_mask':attention_mask, 
                'labels': labels, # 여기까지(input_ids, attention_mask, labels)는 모델 input으로 들어감 
                'instruction': instruction,
                'topic': topic
            }
        
        
            # fine-tune에는 안쓰이고 test에는 쓰임 
    def my_collate_fn(self, data):
    
        instructions = [i['instruction'] for i in data] # don't touch
        topics = [i['topic'] for i in data]
        input_ids = [i['input_ids'] for i in data]
        attention_mask =[i['attention_mask'] for i in data]
        inputs = [i['input'] for i in data]

        max_seq_length = max([len(i) for i in input_ids])
        if self.tokenizer.padding_side == 'left':
            input_ids = [[self.tokenizer.pad_token_id] * (max_seq_length-len(i)) + i for i in input_ids]
            attention_mask = [[0] * (max_seq_length-len(i)) + i for i in attention_mask]
        else:
            input_ids = [[i + self.tokenizer.pad_token_id] * (max_seq_length-len(i)) for i in input_ids]
            attention_mask = [i + [0] * (max_seq_length-len(i)) for i in attention_mask]
        
        input_ids = torch.LongTensor(input_ids)
        attention_mask = torch.LongTensor(attention_mask)
        
        context_batch = defaultdict()
        context_batch['input_ids'] = input_ids
        context_batch['attention_mask'] = attention_mask
        context_batch['topics'] = topics
        context_batch['instructions'] = instructions
        context_batch['inputs'] = inputs

        return context_batch



class Test_CRS_Dataset(Dataset):
    def __init__(self, args, data_path, tokenizer, prompter):
        # pickle / json 데이터 파일 불러오기
        self.args = args
        
        if '.pkl' in data_path:
            self.dataset = pickle.load(open(data_path, 'rb'))
        if '.json' in data_path: 
            self.dataset = json.load(open(data_path))
        
        self.tokenizer = tokenizer
        self.prompter = [prompter, prompter, prompter]
                
        # test_split 했을 때 valid, invalid sample 나눠서 성능 확인하기기
        self.valid_dataset = []
        self.neutral_dataset = []
        self.invalid_dataset = []
        
        if args.dataset_split:
            for sample in self.dataset:
                if sample['topic'] in sample['gpt_validity']['valid']:
                    self.valid_dataset.append(sample)
                if sample['topic'] in sample['gpt_validity']['neutral']:
                    self.neutral_dataset.append(sample)
                if sample['topic'] in sample['gpt_validity']['invalid']:
                    self.invalid_dataset.append(sample)
                    
            self.dataset = self.valid_dataset
        else:
            self.dataset = self.dataset
        
        print("test dataset size: ", len(self.dataset))
        
    def __len__(self):
        # 데이터 샘플 개수를 반환
        return len(self.dataset)
        
    
    def __getitem__(self, index): # 프롬프트까지 만들어서 각 샘플을 넘겨줌, index = batch_size만큼 반복됨
        data = self.dataset[index]
        instruction = data['dialog'] # 배치 내의 샘플 dialog 저장
        topic = data['topic'] # 배치 내의 topic 저장
        if 'pkl' in self.args.data:
            input = ""
        else:
            input = data['input'] # 배치 내의 label 저장
        
        # if self.args.label != "topic":
        #     valid = data['gpt_validity']['valid'][random.randint(0, len(data['gpt_validity']['valid'])-1)] # valid 중에서 하나 뽑기
        #     invalid = data['gpt_validity']['invalid'][random.randint(0, len(data['gpt_validity']['invalid'])-1)] # invalid 중에서 하나 뽑기
        
        # 토큰화
        tokenizing_instruction = self.tokenizer(instruction).input_ids[-self.args.cutoff_len:] #160
        instruction = self.tokenizer.decode(tokenizing_instruction, skip_special_tokens=True)
        prompt_template_idx = random.randrange(len(self.prompter))
        
        # if self.args.mode == 'train':
        #     if self.args.label == 'topic':
        #         prompt = self.prompter[prompt_template_idx].generate_prompt(instruction=dialog, label=topic)
        #     if self.args.label == 'valid' or self.args.label == 'mix':
        #         prompt = self.prompter[prompt_template_idx].generate_prompt(instruction=dialog, label=valid)
        # else:
        prompt = self.prompter[prompt_template_idx].generate_prompt(instruction, input)
        
        # prompt 토큰화
        prompt_tokenizing = self.tokenizer(prompt)
        input_ids = prompt_tokenizing["input_ids"]
        attention_mask = prompt_tokenizing["attention_mask"]
        
        # if self.args.mode == 'train':
        #     input_ids.append(self.tokenizer.eos_token_id)
        #     attention_mask.append(1)
        
        # labels = input_ids.copy()
        
        # if self.args.train_only_label:
        #     user_prompt = self.prompter[prompt_template_idx].generate_prompt(instruction, input)
        #     tokenized_user_prompt = self.tokenizer(user_prompt)
        #     user_prompt_len = len(tokenized_user_prompt["input_ids"])

        #     # user_prompt_len
        #     labels = [-100] * user_prompt_len + labels[user_prompt_len:]  # could be sped up, probably
        
        return {
            'input_ids': input_ids, 
            'attention_mask':attention_mask, 
            # 'labels': labels, 
            'instruction': instruction,
            'topic': topic,
            'input': input
            }


    
    
    # fine-tune에는 안쓰이고 test에는 쓰임 
    def my_collate_fn(self, data):
    
        instructions = [i['instruction'] for i in data] # don't touch
        topics = [i['topic'] for i in data]
        input_ids = [i['input_ids'] for i in data]
        attention_mask =[i['attention_mask'] for i in data]
        inputs = [i['input'] for i in data]

        max_seq_length = max([len(i) for i in input_ids])
        if self.tokenizer.padding_side == 'left':
            input_ids = [[self.tokenizer.pad_token_id] * (max_seq_length-len(i)) + i for i in input_ids]
            attention_mask = [[0] * (max_seq_length-len(i)) + i for i in attention_mask]
        else:
            input_ids = [[i + self.tokenizer.pad_token_id] * (max_seq_length-len(i)) for i in input_ids]
            attention_mask = [i + [0] * (max_seq_length-len(i)) for i in attention_mask]
        
        input_ids = torch.LongTensor(input_ids)
        attention_mask = torch.LongTensor(attention_mask)
        
        context_batch = defaultdict()
        context_batch['input_ids'] = input_ids
        context_batch['attention_mask'] = attention_mask
        context_batch['topics'] = topics
        context_batch['instructions'] = instructions
        context_batch['inputs'] = inputs

        return context_batch

 

