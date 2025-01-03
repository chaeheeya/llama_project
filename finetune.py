import os
import sys
from typing import List

import fire #Python에서의 모든 객체를 command line interface로 만들어 줌 
import torch
import transformers
from datasets import load_dataset

import datetime
from pytz import timezone

from utils.args import parse_args
from utils.dataset import Train_CRS_Dataset

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer, TrainerCallback, BitsAndBytesConfig

from utils.prompter import Prompter



class ModelSaveCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.mdhm = str(datetime.datetime.now(timezone('Asia/Seoul')).strftime('%y%m%d-%H%M'))
    
    def on_epoch_end(self, args, state, control, **kwargs):
        param = parse_args()
        print("params: ", param.model_path, param.task, param.learning_rate, "\n")
        
        if param.model_path == '':
            # mdhm = str(datetime.datetime.now(timezone('Asia/Seoul')).strftime('%y%m%d-%H%M'))
            if param.task == "d2i":
                if param.dataset_split:
                    param.model_path = self.mdhm+'_d2i_'+ param.label+ '_epoch'+str(round(state.epoch)) + '_lr_'+str(param.learning_rate)+'_valid_split'
                else:
                    param.model_path = self.mdhm+'_d2i_'+ param.label+ '_epoch'+str(round(state.epoch)) + '_lr_'+str(param.learning_rate)+'_overall'

           
            elif param.task == "d2r":
                param.model_path = self.mdhm+'_d2r_'+ 'epoch'+str(round(state.epoch)) + '_lr_'+str(param.learning_rate)

        args.output_dir = os.path.join(args.output_dir, param.model_path)

        
        # args.output_dir = os.path.join(args.output_dir, param.model_path)
        print(f"Epoch {state.epoch} finished, saving model to {args.output_dir}")
        kwargs["model"].save_pretrained(args.output_dir)
        args.output_dir = "/home/user/chaehee/prefer_optim/llama/output/"
        
        

def train(
    # model/data params
    args,
    base_model: str = "",  # the only required argument
    data_path: str = "",
    prompt_template_name: str = "",
    output_dir: str = "",
    # training hyperparams
    batch_size: int = 4,
    micro_batch_size: int = 1,
    num_epochs: int = 0,
    learning_rate: float = 0.0,
    cutoff_len: int = 0,
    val_set_size: int = 0,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    
    # # wandb params -> 사용 안함
    # wandb_project: str = "",
    # wandb_run_name: str = "",
    # wandb_watch: str = "",  # options: false | gradients | all
    # wandb_log_model: str = "",  # options: false | true
    # resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    
):
    # os.environ['CUDA_VISIBLE_DEVICES'] = 
    print("argument: ", batch_size, num_epochs, cutoff_len, data_path, "\n")
   
    if int(os.environ.get("LOCAL_RANK", 0)) == 0: # 첫번째 프로세스에서만 설정된 파라미터
        print(
            f"Training LLaMA2 model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            # f"wandb_project: {wandb_project}\n"
            # f"wandb_run_name: {wandb_run_name}\n"
            # f"wandb_watch: {wandb_watch}\n"
            # f"wandb_log_model: {wandb_log_model}\n"
            # f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    # base_model 설정되어 있지 않으면 예외 발생 
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    # 프롬프트 
    prompter = Prompter(prompt_template_name)

    device_map = "cuda:0" 
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1 # 값이 1이 아니면 True가 되어서 ddp(distributed data parallel) 활성화
    
    # ddp -> false라서 아래 코드 안씀
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    # use_wandb = len(wandb_project) > 0 or (
    #     "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    # )
    # # Only overwrite environ if wandb param passed
    # if len(wandb_project) > 0:
    #     os.environ["WANDB_PROJECT"] = wandb_project
    # if len(wandb_watch) > 0:
    #     os.environ["WANDB_WATCH"] = wandb_watch
    # if len(wandb_log_model) > 0:
    #     os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    # 모델 및 토크나이저 로드
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True
    )
    
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    # 패딩토큰에 해당하는 ID 설정
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    ) 
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            # max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            # and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id) # eos 토큰 추가
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point): # 하나씩 처리?
        # 대화 자르기
        data_point["instruction"] = data_point["instruction"]
        data_point["instruction"]=tokenizer(data_point["instruction"])[-args.cutoff_len:]
        data_point["instruction"]=tokenizer.decode(data_point["instruction"]['input_ids'])
        
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        ) 
        # 응답을 포함하여 프롬프트를 만들어주고
        tokenized_full_prompt = tokenize(full_prompt) # 프롬프트를 토큰화
        
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(user_prompt)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1
                

            # user_prompt_len
            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
            # print(tokenized_full_prompt)
        return tokenized_full_prompt

    
    # 모델 로드
    model = prepare_model_for_int8_training(model)

    # LoRA 적용
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    # 데이터 로드 
    """
    DataLoader로 수정 -> 데이터 배치 반복
    from torch.utils.data import DataLoader
    
    train_dataloader = DataLoader(토큰화된 데이터셋, shuffle=True, batch_size=batch_size)
    """
    # if data_path.endswith(".json") or data_path.endswith(".jsonl"):
    #     data = load_dataset("json", data_files=data_path)
    # else:
    #     data = load_dataset(data_path)

    # # if resume_from_checkpoint:
    # #     # Check the available weights and load them
    # #     checkpoint_name = os.path.join(
    # #         resume_from_checkpoint, "pytorch_model.bin"
    # #     )  # Full checkpoint
    # #     if not os.path.exists(checkpoint_name):
    # #         checkpoint_name = os.path.join(
    # #             resume_from_checkpoint, "adapter_model.bin"
    # #         )  # only LoRA model - LoRA config above has to fit
    # #         resume_from_checkpoint = (
    # #             False  # So the trainer won't try loading its state
    # #         )
    
    #     # # The two files above have a different name depending on how they were saved, but are actually the same.
    #     # if os.path.exists(checkpoint_name):
    #     #     print(f"Restarting from {checkpoint_name}")
    #     #     adapters_weights = torch.load(checkpoint_name)
    #     #     set_peft_model_state_dict(model, adapters_weights)
    #     # else:
    #     #     print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.
    
    # # data split
    # if val_set_size > 0:
    #     train_val = data["train"].train_test_split(
    #         test_size=val_set_size, shuffle=True, seed=42
    #     )
    #     train_data = (
    #         train_val["train"].shuffle().map(generate_and_tokenize_prompt)
    #     )
    #     val_data = (
    #         train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    #     )
    # else:
    #     train_data = data["train"].shuffle().map(generate_and_tokenize_prompt) # 데이터의 각 요소에 generate_and_tokenize_prompt 함수 적용
    #     val_data = None

    # # train_data=train_data[:100]

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True
    
    train_dataset = Train_CRS_Dataset(args, data_path, tokenizer, prompter)
    print("train_dataset_size: ", len(train_dataset))
    # print(train_dataset)
    
    # train config 설정 
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            lr_scheduler_type="cosine",
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            save_steps=200,
            save_strategy="epoch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            eval_steps=200 if val_set_size > 0 else None,

            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            # report_to=None
            # run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        # data_collator= train_dataset.my_collate_fn,
        callbacks=[ModelSaveCallback],
    )
    model.config.use_cache = False

    # old_state_dict = model.state_dict
    # model.state_dict = (
    #     lambda self, *_, **__: get_peft_model_state_dict(
    #         self, old_state_dict()
    #     )
    # ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # 학습
    trainer.train()
    
    
    # model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    fire.Fire(train)
