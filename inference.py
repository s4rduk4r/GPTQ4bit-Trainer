from colorama import init, Style, Fore, Back
import time
import torch
from autograd_4bit import load_llama_model_4bit_low_ram_and_offload_to_cpu
from autograd_4bit import load_llama_model_4bit_low_ram
from autograd_4bit import Autograd4bitQuantLinear

# ! Suppress warnings from safetensors
import warnings
warnings.filterwarnings(action="ignore", category=UserWarning, message="TypedStorage is deprecated")

# Color output
init(autoreset=True)

# ! Config
# config_path = "/home/user/models/65b/config"
# model_path = "/home/user/models/65b/model.safetensors"
config_path = "/home/user/models/7b/config"
model_path = "/home/user/models/7b/model.safetensors"

# VRAM
# model, tokenizer = load_llama_model_4bit_low_ram(config_path, model_path, groupsize=128)

# Offload
model, tokenizer = load_llama_model_4bit_low_ram_and_offload_to_cpu(
    config_path=config_path,
    model_path=model_path,
    groupsize=128,
    max_memory={
        0 : "10Gib",
        "cpu" : "80Gib"
    }
)

print(Style.BRIGHT + Fore.LIGHTMAGENTA_EX + 'Fitting 4bit scales and zeros to half')
model.half()
for n, m in model.named_modules():
    if isinstance(m, Autograd4bitQuantLinear):
        if m.groupsize == -1:
            m.zeros = m.zeros.half()
        m.scales = m.scales.half()
        m.bias = m.bias.half()

print(Fore.LIGHTYELLOW_EX + 'Apply AMP Wrapper ...')
from amp_wrapper import AMPWrapper
wrapper = AMPWrapper(model)
wrapper.apply_generate()


prompt = '''Slavik: Define favorable outcome, CABAL.
CABAL: '''
batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
batch = {k: v.cuda() for k, v in batch.items()}

start = time.time()
with torch.no_grad():
    generated = model.generate(inputs=batch["input_ids"],
                               do_sample=True, use_cache=True,
                               repetition_penalty=1.1,
                               max_new_tokens=20,
                               temperature=0.9,
                               top_p=0.95,
                               top_k=40,
                               return_dict_in_generate=True,
                               output_attentions=False,
                               output_hidden_states=False,
                               output_scores=False)
result_text = tokenizer.decode(generated['sequences'].cpu().tolist()[0])
end = time.time()
print(result_text)
print(Fore.LIGHTGREEN_EX + f"Inference time: {end - start:.2f}s")
