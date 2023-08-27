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


class LLMChat:
    def __init__(self, get_model_response_function) -> None:
        self.chat_history = None
        self.get_model_response = get_model_response_function

    def get_user_input(self) -> str:
        """Simple CLI for chat. Supports single-line and multi-line input as well as some additional features
        [HELP] - print help message

        [RAW] - show raw chat history

        [BEGIN] - start multiline input

        [END] - end multiline input

        [NEW] - forget previous conversation and start a new one

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
                    if self.chat_history is not None:
                        print(Fore.LIGHTGREEN_EX + self.chat_history)
                    continue
                case "[NEW]":
                    print(Fore.LIGHTRED_EX + "!!!New conversation started. All previous history is forgotten!!!")
                    self.chat_history = None
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

    def get_system_message(self) -> str:
        """Simple CLI to get model system message

        [HELP] - print help message

        [BEGIN] - start multiline input

        [END] - end multiline input

        [DEFAULT] - use model's default system message

        Returns:
            str: system message string or None
        """

        is_multiline = False
        while True:

            if not is_multiline:
                print(Fore.LIGHTYELLOW_EX + "Prompt System message:", end="")

            input_txt = input()

            match(input_txt):
                case "[HELP]":
                    print(Fore.LIGHTBLUE_EX + "Single-line input mode\nMultiline mode ON: [BEGIN]\nMultiline mode OFF: [END]\nDefault model system message: [DEFAULT]")
                    continue
                case "[BEGIN]":
                    system_msg = ""
                    is_multiline = True
                    continue
                case "[END]":
                    return system_msg
                case "[DEFAULT]":
                    return None
                case _:
                    if is_multiline:
                        system_msg += input_txt + "\n"
                    else:
                        return input_txt


chat = LLMChat(get_model_response)

from prompt_builders import Llama2ChatPrompt
system_msg = chat.get_system_message()
prompt_builder = Llama2ChatPrompt(system_msg)

while True:
    user_text = chat.get_user_input()
    prompt = prompt_builder.make(user_text, chat.chat_history)

    print("Processing ⌛", end="")
    start = time.time()
    chat.chat_history = get_model_response(prompt)
    print("\r" + Fore.LIGHTMAGENTA_EX + "AI:>" + prompt_builder.refine_output(chat.chat_history))
    end = time.time()
    print(Fore.LIGHTGREEN_EX + f"Inference time: {end - start:.2f}s")

    if not config.chat_mode:
        break
