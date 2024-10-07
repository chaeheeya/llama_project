import finetune
import generate
import fire

def main(
            mode: str,    
    ):
        if mode == "train":
             finetune.train(
                base_model="meta-llama/Llama-2-7b-chat-hf", # 모델 이름
                data_path="/home/user/chaehee/llama_project/data/train_dataset.json",  # 데이터 경로 (필요에 맞게 수정)
                output_dir="./output", # 모델이 저장될 경로
                batch_size=128,
                micro_batch_size=4,
                num_epochs=5,
                learning_rate=5e-4,
                # cutoff_len=256,
                val_set_size=0,
                # lora_r=8,
                # lora_alpha=16,
                # lora_dropout=0.05,
                # lora_target_modules=["q_proj", "v_proj"],
                train_on_inputs=False,
                # add_eos_token=False,
                # group_by_length=False,
                # wandb_project="",                  
                # wandb_run_name="",                 
                # wandb_watch="",                    
                # wandb_log_model="",                
                # resume_from_checkpoint=None,       
                prompt_template_name="prompt" # 프롬프트 템플릿 이름(경로X 파일 이름만 )
        )
        
        elif mode == "test":
              generate.test(
                load_8bit=False,
                base_model="meta-llama/Llama-2-7b-chat-hf",
                lora_weights= "/home/user/chaehee/llama_project/output",
                prompt_template= "prompt",  # The prompt template to use, will default to alpaca.
                server_name= "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
                share_gradio= False,
                test_data= "/home/user/chaehee/llama_project/data/test_dataset.json", # data path 입력
                batch_size=3,


                # generation config
                temperature= 0.7,
                # top_p= 0.75,
                top_k= 3,
                num_beams= 2,
        )



if __name__=="__main__": 
     fire.Fire(main)
    
