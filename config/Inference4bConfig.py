import os
from typing import List

class Inference4bConfig:
    """Config holder for LLaMA 4bit inference
    """
    def __init__(self,
                 llama_q4_config_dir: str, llama_q4_model: str,
                 lora_apply_dir: str,
                 groupsize: int,
                 chat_mode: bool,
                 offloading: bool,
                 offload_folder: str,
                 config_file_path: str,
                 device_map : str,
                 max_memory : dict
                 ):
        """
        Args:
            llama_q4_config_dir (str): Path to the config.json, tokenizer_config.json, etc
            llama_q4_model (str): Path to the quantized model in huggingface format
            lora_apply_dir (str): Path to directory from which LoRA has to be applied before training
            chat_mode (bool): Start inference in chat mode
            groupsize (int): Group size of GPTQv2 model
            offloading (bool): Use offloading
            offload_folder (str): Offloading disk folder
            config_file_path (str): path to config file used
        """
        self.llama_q4_config_dir = llama_q4_config_dir
        self.llama_q4_model = llama_q4_model
        self.lora_apply_dir = lora_apply_dir
        self.chat_mode = chat_mode
        self.groupsize = groupsize
        self.offloading = offloading
        self.offload_folder = offload_folder
        self.config_file_path = config_file_path
        self.device_map = device_map if device_map is not None else "auto"
        self.max_memory = None
        if max_memory is not None:
            delattr(self, "max_memory")
            setattr(self, "max_memory", dict())
            for k, v in max_memory.items():
                try:
                    self.max_memory[int(k)] = v
                except:
                    self.max_memory[k] = v


    def __str__(self) -> str:
        s = f"\nParameters:\n{'Inference':-^20}\n" +\
        f"{self.llama_q4_config_dir=}\n{self.llama_q4_model=}\n{self.lora_apply_dir=}\n" +\
        f"{self.groupsize=}\n" +\
        f"{self.chat_mode=}\n" +\
        f"{self.offloading=}\n" +\
        f"{self.offload_folder=}\n" +\
        f"{self.config_file_path=}\n"
        return s.replace("self.", "")
