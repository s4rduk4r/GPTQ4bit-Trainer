# GPTQ4bit Trainer
This is a heavily modified fork of [johnsmith0031/alpaca_lora_4bit](https://github.com/johnsmith0031/alpaca_lora_4bit) concentrating on Triton implementation of 4bit GPTQv2 model finetuning. 

**Note:** If you're interested in GPTQv1 models or CUDA support, please use [johnsmith0031/alpaca_lora_4bit](https://github.com/johnsmith0031/alpaca_lora_4bit)

## Main differences:
1. No CUDA support. And there are no plans to add it in the future
2. No GPTQv1 support
3. Minimal inference code. Just enough to test the models produced
4. Adoption of the most recent upstream [peft](https://github.com/huggingface/peft) library
5. Will focus on producing a standalone python package


# Requirements
**OS:** Linux, Windows WSL

**Libraries:**
- `torch`
- `accelerate`
- `bitsandbytes`
- `transformers`
- `datasets`
- `sentencepiece`
- `safetensors`
- `triton`
- `peft` fork

# Install
```
git clone https://github.com/s4rduk4r/alpaca_lora_4bit
cd alpaca_lora_4bit
pip install -r requirements.txt
```

# Finetune
List all arguments
```sh
python finetune.py -h
```

Command-line arguments
```sh
python finetune.py [args_list]
```

Config file
```sh
python finetune.py --config-file /path/to/config.json
```

# Inference
List all arguments
```sh
python finetune.py -h
```
**Note:** only a subset of arguments is supported:
- llama-q4-config_dir
- llama-q4-model
- groupsize
- lora-apply-dir

Command-line arguments
```
python inference.py [args_list]
```

Config file
```sh
python finetune.py --config-file /path/to/config.json
```

# References:
1. Original code by [johnsmith0031/alpaca_lora_4bit](https://github.com/johnsmith0031/alpaca_lora_4bit)
2. Original triton kernels by [qwopqwop200/GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa/blob/triton/quant.py)
