import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model',         type=str,        default="meta-llama/Llama-2-7b-chat-hf")
    
    parser.add_argument('--mode',               type=str,        default="") # train / test
    parser.add_argument('--task',               type=str,        default="") # d2i / d2r
    
    parser.add_argument('--data',               type=str,        default="")
    parser.add_argument('--prompt_template',    type=str,        default="")
    parser.add_argument('--log_name',           type=str,        default="")
    parser.add_argument('--epochs',             type=int,        default=5)
    parser.add_argument('--batch_size',         type=int,        default=4)
    parser.add_argument('--learning_rate',      type=float,      default=5e-4)
    parser.add_argument('--cutoff_len',         type=int,        default=128)
    parser.add_argument('--dataset_split',          action='store_true') # 명령어에 -- test_split 입력하면, true로 설정되고, 안쓰면 자동으로 false

    # finetune
    parser.add_argument('--model_path',         type=str,        default="") # model save path
    parser.add_argument('--train_only_label',   action='store_true') # 명령어에 --train_only_label만 입력하면 true로 설정되고, 안쓰면 자동으로 false
    parser.add_argument('--label',              type=str,        default="")
    
    # generate
    parser.add_argument('--lora_weights',       type=str,        default="")
    
    
    # generation config
    parser.add_argument('--temperature',        type=float,      default=1.0)
    parser.add_argument('--top_p',              type=float,      default=1.0)
    parser.add_argument('--top_k',              type=int,        default=50)
    parser.add_argument('--num_beams',          type=int,        default=1) 
    parser.add_argument('--num_return_sequences',   type=int,        default=1) 
    parser.add_argument('--do_sample',   action='store_true') # 명령어에 --do_sample만 입력하면 true로 설정되고, 안쓰면 자동으로 false
    parser.add_argument('--max_new_tokens',     type=int,        default=128)   

    args = parser.parse_args()

    return args