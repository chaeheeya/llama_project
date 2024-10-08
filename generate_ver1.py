import os
import sys

import json
from tqdm import tqdm

import fire
import gradio as gr
import torch
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
    log_path : str = ""
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
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
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
    
    
    # def generate_and_tokenize_prompt(data_point):
    #     full_prompt = prompter.generate_prompt(
    #         data_point["instruction"],
    #         data_point["input"],
    #         data_point["output"],
    #     )
    #     tokenized_full_prompt = tokenize(full_prompt)
    #     if not test_on_inputs:
    #         user_prompt = prompter.generate_prompt(
    #             data_point["instruction"], data_point["input"]
    #         )
    #         tokenized_user_prompt = tokenize(
    #             user_prompt, add_eos_token=add_eos_token
    #         )
    #         user_prompt_len = len(tokenized_user_prompt["input_ids"])

    #         if add_eos_token:
    #             user_prompt_len -= 1

    #         tokenized_full_prompt["labels"] = [
    #             -100
    #         ] * user_prompt_len + tokenized_full_prompt["labels"][
    #             user_prompt_len:
    #         ]  # could be sped up, probably
    #     return tokenized_full_prompt

    # if test_data:
    #     train_data = test_data["train"].shuffle().map(generate_and_tokenize_prompt)
    #     test_dataset = test_data_load(test_data)
    #     hit_score_eval(test_dataset, log_path )
    #     return
    
    
    def hit_score_eval(
            dataset, 
            log_file="",
            temperature=0.1,
            top_p=0.75,
            top_k=1,
            num_beams=4,
            max_new_tokens=256,
            repetition_penalty=4.8,
            stream_output=False,
            **kwargs,
        ):

    
        hit = 0
        total = 0
    
        log_data=[]

        for data in tqdm(dataset[total:], bar_format='{percentage:3.0f} % | {bar:23} {r_bar}'):
            instruction = data['instruction'] 
            # print(instruction, "야야야야야야")
            input = data['input']
            # print(input, "야야야야야야")
            label = data['output']
            # print(label, "야야야야야야")

            # prompt 생성하기 & 모델에 들어갈 입력 만들어주기
            prompt = prompter.generate_prompt(instruction, input)
            inputs = tokenizer(prompt, return_tensors="pt")
            # print(inputs)
            input_ids = inputs["input_ids"].to(device) # 데이터를 GPU로 이동시킬 때 텐서로 이동


            # GenerationConfig 설정
            generation_config = GenerationConfig(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_beams=num_beams,
                repetition_penalty=float(repetition_penalty),
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
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=max_new_tokens,
                )

                # print(generation_output.sequences)
                # print(generation_output.sequences[0])

            generated_output = tokenizer.decode(generation_output.sequences[0])
            output = prompter.get_response(generated_output).split('</s>')[0]


            # Hit score 계산
            if output == label:
                hit +=1
            total +=1

            # 로그 데이터 저장
            log_data.append({
                "instruction": instruction,
                "input": input,
                "topic_item": label,
                "generated_output": output,
                "is_hit": ' '.join(output.strip().lower().split()) == ' '.join(label.strip().lower().split())
            })

        # 로그 데이터를 json 파일로 저장
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=4)

        # hit score 계산
        hit_score = hit / total
        print(f"Hit score: {hit_score * 100:.4f}%")

        return hit_score


    if test_data:
        test_dataset = test_data_load(test_data)
        hit_score_eval(test_dataset, log_path )
        return
    
    
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

    # gr.Interface(
    #     fn=evaluate,
    #     inputs=[
    #         gr.components.Textbox(
    #             lines=2,
    #             label="Instruction",
    #             placeholder="Tell me about alpacas.",
    #         ),
    #         gr.components.Textbox(lines=2, label="Input", placeholder="none"),
    #         gr.components.Slider(
    #             minimum=0, maximum=1, value=0.7, label="Temperature"
    #         ),
    #         gr.components.Slider(
    #             minimum=0, maximum=1, value=0.1, label="Top p"
    #         ),
    #         gr.components.Slider(
    #             minimum=0, maximum=100, step=1, value=40, label="Top k"
    #         ),
    #         gr.components.Slider(
    #             minimum=1, maximum=4, step=1, value=4, label="Beams"
    #         ),
    #         gr.components.Slider(
    #             minimum=1, maximum=2000, step=1, value=256, label="Max tokens"
    #         ),
    #         gr.components.Slider(
    #             minimum=1., maximum=5., step=0.1, value=4.8, label="Repetation penalty"
    #         ),
    #         gr.components.Checkbox(label="Stream output", value=True),
    #     ],
    #     outputs=[
    #         gr.inputs.Textbox(
    #             lines=5,
    #             label="Output",
    #         )
    #     ],
    #     title="🦙🌲 Alpaca-LoRA",
    #     description="Alpaca-LoRA is a 7B-parameter LLaMA model finetuned to follow instructions. It is trained on the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) dataset and makes use of the Huggingface LLaMA implementation. For more information, please visit [the project's website](https://github.com/tloen/alpaca-lora).",  # noqa: E501
    # ).queue().launch(server_name="0.0.0.0", share=share_gradio)
    # # Old testing code follows.

    """
    # testing code for readme
    for instruction in [
        "Tell me about alpacas.",
        "Tell me about the president of Mexico in 2019.",
        "Tell me about the king of France in 2019.",
        "List all Canadian provinces in alphabetical order.",
        "Write a Python program that prints the first 10 Fibonacci numbers.",
        "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",  # noqa: E501
        "Tell me five words that rhyme with 'shock'.",
        "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
        "Count up from 1 to 500.",
    ]:
        print("Instruction:", instruction)
        print("Response:", evaluate(instruction))
        print()
    """


if __name__ == "__main__":
    fire.Fire(test)
