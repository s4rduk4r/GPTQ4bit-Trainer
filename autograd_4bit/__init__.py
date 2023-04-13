__version__ = "GPTQv2"

from colorama import init, Fore, Style

init(autoreset=True)

# Module interface
from .autograd_4bit import Autograd4bitQuantLinear
from .autograd_4bit import load_llama_model_4bit_low_ram
from .autograd_4bit import load_llama_model_4bit_low_ram_and_offload_to_cpu

# Patch
from . import lora_patch
