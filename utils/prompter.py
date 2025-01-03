import json
import os.path as osp
from typing import Union
from utils.args import parse_args

args = parse_args()

class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        
        if args.task == "d2i": 
            template_name = "d2i_prompt_template"
        if args.task == "d2r":
            template_name = "d2r_prompt_template"
    
        
        file_name = osp.join("/home/user/chaehee/prefer_optim/llama/templates", f"{template_name}.json")
        # /home/user/chaehee/llama_project/templates/prompt.json
        if not osp.exists(file_name):
            raise ValueError(f"Can't read prompt template {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    
    def generate_prompt(
            self,
            instruction: str,
            input: Union[None, str] = None,
            label: Union[None, str] = None,
    ) -> str:
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res
    
    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()