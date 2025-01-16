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
        
        if args.task == "DIP2I":
            self.knowledges = json.load(open('/home/user/chaehee/prefer_optim/data/insp2/en_train_know_ins2combined_new3_dpr.json', 'r', encoding='utf-8'))
            # 샘플 자르기
            self.selected_sample = []
            self.selected_knowledges = []
            for i, j in zip(self.dataset, self.knowledges):
                if i['topic'] in i['predicted_topic'][:3]:
                    self.selected_sample.append(i)
                    top3_passages = j['predicted_know'][:3]
                    for idx1, top_passages in enumerate(top3_passages):
                        for idx2, passage in enumerate(top_passages):
                            top3_passages[idx1][idx2] = self.tokenizer.decode(self.tokenizer(passage).input_ids[1:][:args.passage_cutoff]).strip()
                    
                    self.selected_knowledges.append(top3_passages)
            self.dataset = self.selected_sample

        
        # 샘플 valid, invalid, neutral (조합) 구별하기
        if args.dataset_split == 'overall':
            self.dataset = self.dataset
            
        elif args.dataset_split == 'overall_sampling':
            self.sampled_indices = random.sample(range(len(self.dataset)), args.sampling_len)
            self.dataset = [self.dataset[i] for i in self.sampled_indices]
            # print(self.sampled_indices)
        
        elif args.dataset_split == 'valid':
            train_sample = [i for i in self.dataset if i['gpt_validity']=='valid']
            self.dataset = train_sample
        
        elif args.dataset_split == 'valid_sampling':
            train_sample = [i for i in self.dataset if i['gpt_validity']=='valid']
            sampled_indices = random.sample(range(len(train_sample)), args.sampling_len)
            self.dataset = [train_sample[i] for i in sampled_indices]
        
        elif args.dataset_split == 'valid-neutral': # 중복 없음 
            train_sample = [i for i in self.datset if i['gpt_validity']=='valid' or i['gpt_validity']=='neutral']
            self.dataset = train_sample
            
        elif args.dataset_split == 'valid-invalid':
            train_sample = [i for i in self.datset if i['gpt_validity'] == 'valid' or i['gpt_validity'] == 'invalid']
            self.dataset = train_sample
            
        elif args.dataset_split == 'valid-invalid_sampling':
            valid_sample = [i for i in self.dataset if i['gpt_validity']=='valid']
            invalid_sample = [i for i in self.dataset if i['gpt_validity'] == 'invalid']
            self.sampled_indices = random.sample(range(len(invalid_sample)), 389)
            self.dataset = [invalid_sample[i] for i in self.sampled_indices] + valid_sample
        
        elif args.dataset_split=='invalid':
            invalid_sample = [i for i in self.dataset if i['gpt_validity']=='invalid']
            self.dataset = invalid_sample
            
        
        
        ## LLaMA가 D2I 생성한 결과로 valid, invalid, neutral 나누기
        elif args.dataset_split == 'topk':
            # LLaMA D2I 생성 결과
            llama_gen = json.load(open('/home/user/chaehee/prefer_optim/llama/log/250107-1749_d2i_250106-1040_d2i_topic_epoch5_lr_0.0004overall_sampling.json', 'r', encoding='utf-8'))
            
            valid_sample_indices = [idx for idx, i in enumerate(llama_gen) if i['topic_item'] in i['generated_output'][:args.topk_valid]]
            neutral_sample_indices = [idx for idx, i in enumerate(llama_gen) if i['topic_item'] in i['generated_output'][args.topk_valid:10]]
            invalid_sample_indices = [idx for idx, i in enumerate(llama_gen) if i['topic_item'] not in i['generated_output']]
            
            valid_sample = [self.dataset[i] for i in valid_sample_indices]
            neutral_sample = [self.dataset[i] for i in neutral_sample_indices]
            invalid_sample = [self.dataset[i] for i in invalid_sample_indices]
            
            if args.combination == 'valid':
                self.dataset = valid_sample
            if args.combination == 'valid-neutral':
                self.dataset = valid_sample+neutral_sample
            if args.combination == 'valid-invalid':
                self.dataset = valid_sample+invalid_sample
            if args.combination == 'neutral':
                self.dataset = neutral_sample
            if args.combination == 'neutral-invalid':
                self.dataset = neutral_sample+invalid_sample
            if args.combination == 'invalid':
                self.dataset = invalid_sample
            
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
            # (1) valid에 topic 추가 
            for sample in self.dataset:
                if sample['topic'] not in sample['gpt_validity']['valid']:
                    sample['gpt_validity']['valid'].append(sample['topic'])
            
        if args.label == "except_topic": # valid에 topic 제외하고 학습
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
                        
                    
        
    def __len__(self):
        # 데이터 샘플 개수를 반환
        return len(self.dataset)
        
    
    def __getitem__(self, index): # 프롬프트까지 만들어서 각 샘플을 넘겨줌, index = batch_size만큼 반복됨
        data = self.dataset[index]
        instruction = data['dialog'] # 배치 내의 샘플 dialog 저장
        topic = data['topic'] # 배치 내의 topic 저장
        
        if self.args.task=='d2i':
            if 'pkl' in self.args.data:
                input = ""
            else:
                input = data['input']
        
        elif self.args.task=='DIP2I':
            candidate_items_list = data['predicted_topic'][:3]
            candidate_passages = self.selected_knowledges[index]
            
            tmp_dict = dict(zip(candidate_items_list, candidate_passages))
            
            random.shuffle(candidate_items_list)
            input_candidate_items = '\n'.join([f"Item {idx + 1}. {t}" for idx, t in enumerate(candidate_items_list)])
                       
            input_candidate_passages= []
            for i in range(len(candidate_items_list)):
                prefix = f"Here are the candidate passages about Item {i+1}. {candidate_items_list[i]}"
                passage_text = '\n'.join([f"Passage {idx + 1 + i * 4}. {know}" for idx, know in enumerate(tmp_dict[candidate_items_list[i]][:4])]) # topic당 4개의 passage 만 입력 
                input_candidate_passages.append(prefix + '\n' + passage_text)
            input_candidate_passages =  '\n\n'.join(input_candidate_passages)

        
        tokenizing_instruction = self.tokenizer(instruction).input_ids[-self.args.cutoff_len:] #160
        instruction = self.tokenizer.decode(tokenizing_instruction, skip_special_tokens=True)
        prompt_template_idx = random.randrange(len(self.prompter))
        
        if self.args.label == 'topic':
            if self.args.task=='d2i':
                prompt = self.prompter[prompt_template_idx].generate_prompt(instruction=instruction, input=input, label=topic)
            elif self.args.task=='DIP2I':
                prompt = self.prompter[prompt_template_idx].generate_prompt(instruction=instruction, input=input_candidate_items, input2=input_candidate_passages, label=topic)
        
        elif self.args.label == 'valid' or self.args.label == 'mix' or self.args.label == 'except_topic':
            valid_topic = data['gpt_validity']['valid'][random.randint(0, len(data['gpt_validity']['valid'])-1)]
            prompt = self.prompter[prompt_template_idx].generate_prompt(instruction=instruction, input=input, label=valid_topic)
        
        
        inputs_token = self.tokenizer(prompt)
        input_ids = inputs_token["input_ids"]
        attention_mask = inputs_token["attention_mask"]
        
        # eos token 프롬프트에 추가하기
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
                
        if args.task == "DIP2I":
            self.knowledges = json.load(open('/home/user/chaehee/prefer_optim/data/insp2/en_test_know_ins2combined_new3_dpr.json', 'r', encoding='utf-8'))
            
        # test_split 했을 때 valid, invalid sample 나눠서 성능 확인하기 
        self.valid_dataset = []
        if args.dataset_split == 'valid':
            for sample in self.dataset:
                if sample['gpt_validity'] == 'valid':
                    self.valid_dataset.append(sample)
            self.dataset = self.valid_dataset
        
        elif args.dataset_split == 'neutral':
            for sample in self.dataset:
                if sample['gpt_validity'] == 'valid':
                    self.valid_dataset.append(sample)
                if sample['gpt_validity'] == 'valid':
                    self.valid_dataset.append(sample)
            self.dataset = self.valid_dataset

        elif args.dataset_split == 'overall':
            self.dataset = self.dataset
            
        
        print("test dataset size: ", len(self.dataset))
        
    def __len__(self):
        # 데이터 샘플 개수를 반환
        return len(self.dataset)
        
    
    def __getitem__(self, index): # 프롬프트까지 만들어서 각 샘플을 넘겨줌, index = batch_size만큼 반복됨
        data = self.dataset[index]
        instruction = data['dialog'] # 배치 내의 샘플 dialog 저장
        topic = data['topic'] # 배치 내의 topic 저장
        
        if self.args.task == 'd2i':
            if 'pkl' in self.args.data:
                input = ""
            else:
                input = data['input'] # 배치 내의 label 저장
        
        elif self.args.task=='DIP2I':
            candidate_items_list = data['predicted_topic'][:3]
            candidate_passages = self.knowledges[index]['predicted_know'][:3]
            
            # input = {candidate_items_list, candidate_passages}
            input = candidate_items_list
            tmp_dict = dict(zip(candidate_items_list, candidate_passages))
            
            # random.shuffle(candidate_items_list)
            
            input_candidate_items = '\n'.join([f"Item {idx + 1}. {t}" for idx, t in enumerate(candidate_items_list)])
            input_candidate_passages = []
            for i in range(len(candidate_items_list)):
                prefix = f"Here are the candidate passages about Item {i+1}. {candidate_items_list[i]}"
                # passage_text = '\n'.join([f"Passage {idx + 1}. {know}" for idx, know in enumerate(tmp_dict[candidate_items_list[i]][:4])])
                passage_text = '\n'.join([f"Passage {idx + 1 + i*4}. {know}" for idx, know in enumerate(candidate_passages[i][:4])])
                input_candidate_passages.append(prefix + '\n' + passage_text)
            input_candidate_passages = '\n\n'.join(input_candidate_passages)           
            
        # 토큰화
        tokenizing_instruction = self.tokenizer(instruction).input_ids[-self.args.cutoff_len:] #160
        instruction = self.tokenizer.decode(tokenizing_instruction, skip_special_tokens=True)
        prompt_template_idx = random.randrange(len(self.prompter))
        
        # prompt 만들기 
        if self.args.task == 'd2i':
            prompt = self.prompter[prompt_template_idx].generate_prompt(instruction=instruction, input=input)
        elif self.args.task == 'DIP2I':
            prompt = self.prompter[prompt_template_idx].generate_prompt(instruction=instruction, input=input_candidate_items, input2=input_candidate_passages)
        
        # prompt 토큰화
        prompt_tokenizing = self.tokenizer(prompt)
        input_ids = prompt_tokenizing["input_ids"]
        attention_mask = prompt_tokenizing["attention_mask"]
        
        
        return {
            'input_ids': input_ids, 
            'attention_mask':attention_mask,  
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

 

