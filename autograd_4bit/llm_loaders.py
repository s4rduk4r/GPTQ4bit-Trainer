from typing import Dict
from colorama import Fore, Style
import time

import torch
import torch.nn as nn
import accelerate
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer
from .autograd_4bit import make_quant_for_4bit_autograd


class EModelType:
    LLaMA = "llama"
    GPT_J = "gpt-j"
    GPT_NeoX = "gpt-neox"
    T5_Flan = "t5-flan"


class LLMGPTQv2LoaderArguments:
    def __init__(self,
                 model: EModelType,
                 config_path: str,
                 model_path: str,
                 groupsize: int,
                 seqlen: int,
                 device_map: Dict[str, int | str] = None,
                 max_memory: Dict[str, int | str] = None,
                 lora_path: str = None
                 ) -> None:
        self.model = model
        self.config_path = config_path
        self.model_path = model_path
        self.groupsize = groupsize if groupsize > 0 else -1
        self.seqlen = seqlen
        self.device_map = device_map
        self.max_memory = max_memory
        self.lora_path = lora_path


def load_llm(args: LLMGPTQv2LoaderArguments):
    match args.model:
        case EModelType.LLaMA:
            if args.max_memory is None:
                model, token = LLaMALoader.load_model_4bit_low_ram(
                    config_path=args.config_path, model_path=args.model_path,
                    groupsize=args.groupsize, seqlen=args.seqlen,
                    device_map=args.device_map
                )
            else:
                model, token = LLaMALoader.load_model_4bit_low_ram_and_offload_to_cpu(
                    config_path=args.config_path, model_path=args.model_path,
                    lora_path=args.lora_path,
                    groupsize=args.groupsize, seqlen=args.seqlen,
                    max_memory=args.max_memory
                )
        case EModelType.GPT_J:
            raise NotImplemented(f"{args.model} loader not implemented yet")
        case EModelType.GPT_NeoX:
            raise NotImplemented(f"{args.model} loader not implemented yet")
        case EModelType.T5_Flan:
            raise NotImplemented(f"{args.model} loader not implemented yet")
        case _:
            raise ValueError(f"{args.model} is unknown LLM type")

    return model, token


class LLaMALoader:

    @classmethod
    def find_layers(self, module, layers=[nn.Conv2d, nn.Linear], name=''):
        if type(module) in layers:
            return {name: module}
        res = {}
        for name1, child in module.named_children():
            res.update(LLaMALoader.find_layers(
                child, layers=layers, name=name + '.' + name1 if name != '' else name1
            ))
        return res

    @classmethod
    def load_model_4bit_low_ram(self, config_path, model_path, groupsize=-1, device_map="auto", seqlen=2048):

        print(Style.BRIGHT + Fore.CYAN + "Loading Model ...")
        t0 = time.time()

        with accelerate.init_empty_weights():
            config = LlamaConfig.from_pretrained(config_path)
            model = LlamaForCausalLM(config)
            model = model.eval()
            layers = LLaMALoader.find_layers(model)
            for name in ['lm_head']:
                if name in layers:
                    del layers[name]
            make_quant_for_4bit_autograd(model, layers, groupsize=groupsize)

        device_map = accelerate.infer_auto_device_map(
                model,
                no_split_module_classes=["LlamaDecoderLayer"]
            ) if device_map == "auto" else device_map
        model = accelerate.load_checkpoint_and_dispatch(
            model=model,
            checkpoint=model_path,
            device_map=device_map,
            no_split_module_classes=["LlamaDecoderLayer"]
        )

        model.seqlen = seqlen

        tokenizer = LlamaTokenizer.from_pretrained(config_path)
        tokenizer.truncation_side = 'left'

        print(Style.BRIGHT + Fore.GREEN + f"Loaded the model in {(time.time()-t0):.2f} seconds.")

        return model, tokenizer

    @classmethod
    def load_model_4bit_low_ram_and_offload_to_cpu(self, config_path, model_path, lora_path=None, groupsize=-1, seqlen=2048, max_memory=None):

        if max_memory is None:
            max_memory = {0: '24Gib', 'cpu': '48Gib'}

        print(Style.BRIGHT + Fore.CYAN + "Loading Model ...")
        t0 = time.time()

        with accelerate.init_empty_weights():
            config = LlamaConfig.from_pretrained(config_path)
            model = LlamaForCausalLM(config)
            model = model.eval()
            layers = LLaMALoader.find_layers(model)
            for name in ['lm_head']:
                if name in layers:
                    del layers[name]
            make_quant_for_4bit_autograd(model, layers, groupsize=groupsize)
        accelerate.load_checkpoint_in_model(model, checkpoint=model_path, device_map={'': 'cpu'})

        # # rotary_emb fix
        for n, m in model.named_modules():
            if 'rotary_emb' in n:
                cos_cached = m.cos_cached.clone().cpu()
                sin_cached = m.sin_cached.clone().cpu()
                break

        if lora_path is not None:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, lora_path, device_map={'': 'cpu'}, torch_dtype=torch.float32)
            print(Style.BRIGHT + Fore.GREEN + '{} Lora Applied.'.format(lora_path))

        model.seqlen = seqlen


        print(Style.BRIGHT + Fore.BLUE + 'Dispatching model ...')
        device_map = accelerate.infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=["LlamaDecoderLayer"])
        model = accelerate.dispatch_model(model, device_map=device_map, offload_buffers=True, main_device=0)
        torch.cuda.empty_cache()
        print(Style.BRIGHT + Fore.YELLOW + 'Total {:.2f} Mib VRAM used.'.format(torch.cuda.memory_allocated() / (1024 ** 2)))

        # rotary_emb fix
        for n, m in model.named_modules():
            if 'rotary_emb' in n:
                if getattr(m, '_hf_hook', None):
                    if isinstance(m._hf_hook, accelerate.hooks.SequentialHook):
                        hooks = m._hf_hook.hooks
                    else:
                        hooks = [m._hf_hook]
                    for hook in hooks:
                        if hook.offload:
                            if n + '.sin_cached' not in hook.weights_map.dataset.state_dict.keys():
                                hook.weights_map.dataset.state_dict[n + '.sin_cached'] = sin_cached.clone().cpu()
                                hook.weights_map.dataset.state_dict[n + '.cos_cached'] = cos_cached.clone().cpu()

        tokenizer = LlamaTokenizer.from_pretrained(config_path)
        tokenizer.truncation_side = 'left'

        print(Style.BRIGHT + Fore.GREEN + f"Loaded the model in {(time.time()-t0):.2f} seconds.")

        return model, tokenizer
