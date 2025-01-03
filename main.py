import os
import fire

import utils.args

import finetune
import generate

def main():
    
    args = utils.args.parse_args()
    # args.mode = 'train'
    # args.task = 'd2i'
    # args.data = 'train_dataset_debug.json'
    # args.prompt_template = 'prompt'
    if args.mode == "train":
        finetune.train(
                args = args,
                base_model=args.base_model, # 모델 이름
                data_path=os.path.join("/home/user/chaehee/prefer_optim/data/insp2/", args.data), # data path 입력
                output_dir="/home/user/chaehee/prefer_optim/llama/output/", # 모델이 저장될 상위 디렉토리
                batch_size=args.batch_size,
                micro_batch_size=1,
                num_epochs=args.epochs,
                learning_rate=args.learning_rate,
                cutoff_len=args.cutoff_len,
                val_set_size=0,
                train_on_inputs=False,
                prompt_template_name=args.prompt_template # 프롬프트 템플릿 이름(경로X 파일 이름만 )
            )
    elif args.mode == "test":
        generate.test(
                args = args,
                base_model="meta-llama/Llama-2-7b-chat-hf",
                lora_weights= os.path.join("/home/user/chaehee/prefer_optim/llama/output/", args.lora_weights),
                prompt_template= args.prompt_template,  # The prompt template to use, will default to alpaca.
                test_data= os.path.join("/home/user/chaehee/prefer_optim/data/insp2/", args.data), # data path 입력
                batch_size=args.batch_size
                )  
        
if __name__=="__main__": 
    fire.Fire(main)
    
