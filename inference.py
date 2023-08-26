from colorama import init, Style, Fore, Back
import time
import torch
from autograd_4bit import (
    load_llm, LLMGPTQv2LoaderArguments, EModelType
)

import config
config.WORK_MODE = config.EWorkModes.INFERENCE

from config.arg_parser import get_config

# ! Suppress warnings from safetensors
import warnings
warnings.filterwarnings(action="ignore", category=UserWarning, message="TypedStorage is deprecated")

# Color output
init(autoreset=True)

# ! Config
config = get_config()

config_path = config.llama_q4_config_dir
model_path = config.llama_q4_model
groupsize = config.groupsize

# * Show loaded parameters
print(f"{config}\n")

# Load Basic Model
model_args = LLMGPTQv2LoaderArguments(
    EModelType.LLaMA,
    config.llama_q4_config_dir,
    config.llama_q4_model,
    config.groupsize,
    # 2048,
    4096, # Llama-2 seqlen
    config.device_map,
    config.max_memory if config.offloading else None,
    config.offload_folder,
    config.lora_apply_dir
)
model, tokenizer = load_llm(model_args)

if not config.offloading:
    print(Fore.LIGHTYELLOW_EX + 'Apply AMP Wrapper ...')
    from amp_wrapper import AMPWrapper
    wrapper = AMPWrapper(model)
    wrapper.apply_generate()


def get_model_response(prompt: str) -> str:
    """Get response from ML-model

    Args:
        prompt (str): Valid model prompt

    Returns:
        str: raw model response
    """
    batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    batch = {k: v.cuda() for k, v in batch.items()}

    with torch.no_grad():
        generated = model.generate(inputs=batch["input_ids"],
                                do_sample=True, use_cache=True,
                                repetition_penalty=1.1,
                                max_new_tokens=4096,
                                temperature=0.7,
                                top_p=0.95,
                                #    top_k=40,
                                return_dict_in_generate=True,
                                output_attentions=False,
                                output_hidden_states=False,
                                output_scores=False)
    result_text = tokenizer.decode(generated['sequences'].cpu().tolist()[0])
    return result_text


def get_user_input() -> str:
    """Simple CLI for chat. Supports single-line and multi-line input as well as some additional features
    [HELP] - print help message
    [RAW] - show raw chat history
    [BEGIN] - start multiline input
    [END] - end multiline input

    Returns:
        str: user input
    """
    is_multiline = False

    while True:
        if not is_multiline:
            print(Fore.LIGHTCYAN_EX + "USR:>", end="")

        input_line = input()

        match(input_line):
            case "":
                continue
            case "[HELP]":
                print(Fore.LIGHTBLUE_EX + "Type your text to converse.\nMultiline mode ON: [BEGIN]\nMultiline mode OFF: [END]\nRaw chat history: [RAW]")
                continue
            case "[RAW]":
                print(Fore.LIGHTGREEN_EX + chat_history)
                continue
            case "[BEGIN]":
                user_text = ""
                is_multiline = True
                continue
            case "[END]":
                return user_text
            case _:
                if is_multiline:
                    user_text += input_line + "\n"
                else:
                    return input_line


from prompt_builders import Llama2ChatPrompt

# Default system message
prompt_system_msg = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""


system_msg = input(Fore.LIGHTYELLOW_EX + "Prompt System message:")
if system_msg.startswith("+"):
    system_msg = prompt_system_msg + system_msg.replace("+", "\n\n", 1)
elif system_msg == "":
    system_msg = prompt_system_msg

prompt_builder = Llama2ChatPrompt(system_msg)

chat_history = None

while True:
    user_text = get_user_input()
    prompt = prompt_builder.make(user_text, chat_history)

    print("Processing ⌛", end="")
    start = time.time()
    chat_history = get_model_response(prompt)
    print("\r" + Fore.LIGHTMAGENTA_EX + "AI:>" + prompt_builder.refine_output(chat_history))
    end = time.time()
    print(Fore.LIGHTGREEN_EX + f"Inference time: {end - start:.2f}s")

    if not config.chat_mode:
        break
