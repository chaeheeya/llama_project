import utils.args

import finetune
import generate
import fire

def main(
            mode: str,
            sort: str,
            data_path: str,

        ):
    
    args = utils.args.parse_args()
    
    if args.mode == "train":
        if args.sort == "d2i":
            finetune.train(
                    base_model="meta-llama/Llama-2-7b-chat-hf", # 모델 이름
                    data_path="/home/user/chaehee/llama_project/data/train_dataset_response.json",  # 데이터 경로 (필요에 맞게 수정)
                    output_dir="./output", # 모델이 저장될 경로
                    batch_size=128,
                    micro_batch_size=4,
                    num_epochs=5,
                    learning_rate=5e-4,
                    cutoff_len=128,
                    val_set_size=0,
                    train_on_inputs=False,
                    prompt_template_name="prompt" # 프롬프트 템플릿 이름(경로X 파일 이름만 )
                )
        elif args.sort == "d2r":
            finetune.train()

        
    elif args.mode == "test":
        print("테스트 된다된다된다된다\n")
        if args.sort == "d2i":
            print("그 다음도  된다된다된다된다\n")
            generate.test(
                load_8bit=False,
                base_model="meta-llama/Llama-2-7b-chat-hf",
                lora_weights= "/home/user/chaehee/llama_project/output",
                prompt_template= "prompt",  # The prompt template to use, will default to alpaca.
                server_name= "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
                share_gradio= False,
                test_data= "/home/user/chaehee/llama_project/data/test_dataset.json", # data path 입력
                batch_size=3
                )  
        elif args.sort == "d2r":
            pass
            
            
            
            
if __name__=="__main__": 
    fire.Fire(main)
    
