import os
from typing import List

class Inference4bConfig:
    """Config holder for LLaMA 4bit inference
    """
    def __init__(self, 
                 llama_q4_config_dir: str, llama_q4_model: str,
                 lora_apply_dir: str,
                 groupsize: int,
                 offloading: bool
                 ):
        """
        Args:
            llama_q4_config_dir (str): Path to the config.json, tokenizer_config.json, etc
            llama_q4_model (str): Path to the quantized model in huggingface format
            lora_apply_dir (str): Path to directory from which LoRA has to be applied before training
            groupsize (int): Group size of GPTQv2 model
            offloading (bool): Use offloading
        """
        self.llama_q4_config_dir = llama_q4_config_dir
        self.llama_q4_model = llama_q4_model
        self.lora_apply_dir = lora_apply_dir
        self.groupsize = groupsize
        self.offloading = offloading


    def __str__(self) -> str:
        s = f"\nParameters:\n{'Inference':-^20}\n" +\
        f"{self.llama_q4_config_dir=}\n{self.llama_q4_model=}\n{self.lora_apply_dir=}\n" +\
        f"{self.groupsize=}\n" +\
        f"{self.offloading=}"
        return s.replace("self.", "")
