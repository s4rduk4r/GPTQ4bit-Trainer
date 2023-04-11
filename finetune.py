"""
    llama-4b trainer with support of Stanford Alpaca-like JSON datasets (short for SAD)
    Intended to use with https://github.com/johnsmith0031/alpaca_lora_4bit

    SAD structure:
    [
        {
            "instruction": "Give null hypothesis",
            "input": "6 subjects were given a drug (treatment group) and an additional 6 subjects a placebo (control group).",
            "output": "Drug is equivalent of placebo"
        },
        {
            "instruction": "What does RNA stand for?",
            "input": "",
            "output": "RNA stands for ribonucleic acid."
        }
    ]
"""

import torch
import transformers
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from autograd_4bit import load_llama_model_4bit_low_ram

# ! Suppress warnings from safetensors
import warnings
warnings.filterwarnings(action="ignore", category=UserWarning, message="TypedStorage is deprecated")

# ! Config
from config.arg_parser import get_config
import train_data

ft_config = get_config()

# * Show loaded parameters
if ft_config.local_rank == 0:
    print(f"{ft_config}\n")

if ft_config.gradient_checkpointing:
    print('Disable Dropout.')

# Load Basic Model
model, tokenizer = load_llama_model_4bit_low_ram(ft_config.llama_q4_config_dir,
                                                 ft_config.llama_q4_model,
                                                 device_map=ft_config.device_map,
                                                 groupsize=ft_config.groupsize
                                                )

# Config Lora
lora_config = LoraConfig(
    r=ft_config.lora_r,
    lora_alpha=ft_config.lora_alpha,
    target_modules=ft_config.lora_target_modules,
    lora_dropout=ft_config.lora_dropout,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False
)
if ft_config.lora_apply_dir is None:
    model = get_peft_model(model, lora_config)
else:
    if ft_config.ddp:
        model = PeftModel.from_pretrained(model, ft_config.lora_apply_dir, is_trainable=True, device_map="auto", torch_dtype=torch.float32)
    else:
        model = PeftModel.from_pretrained(model, ft_config.lora_apply_dir, is_trainable=True, device_map={'': 0}, torch_dtype=torch.float32)
    print(ft_config.lora_apply_dir, 'loaded')

# Print trainable parameters
model.print_trainable_parameters()


# Set tokenizer
tokenizer.pad_token_id = 0

if not ft_config.skip:
    # Load Data
    data = None
    if ft_config.ds_type == "txt" and not ft_config.skip:
        #### LLaMa
        data = train_data.TrainTxt(ft_config.dataset, ft_config.val_set_size, tokenizer, ft_config.cutoff_len)
    elif ft_config.ds_type == "alpaca" and not ft_config.skip:
        #### Stanford Alpaca-like Data
        data = train_data.TrainSAD(ft_config.dataset, ft_config.val_set_size, tokenizer, ft_config.cutoff_len)
    elif ft_config.ds_type == "gpt4all" and not ft_config.skip:
        #### GPT4All Data
        data = train_data.TrainGPT4All(ft_config.dataset, ft_config.val_set_size, tokenizer, ft_config.cutoff_len)
    else:
        raise NotImplementedError("ERROR: Unknown dataset format")
    data.prepare_data(thd=ft_config.txt_row_thd, use_eos_token=ft_config.use_eos_token)
    ####

    # Use gradient checkpointing
    if ft_config.gradient_checkpointing:
        print('Applying gradient checkpointing ...')
        from gradient_checkpointing import apply_gradient_checkpointing
        apply_gradient_checkpointing(model, checkpoint_ratio=ft_config.gradient_checkpointing_ratio)

    # Disable Trainer's DataParallel for multigpu
    if not ft_config.ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    training_arguments = transformers.TrainingArguments(
        per_device_train_batch_size=ft_config.mbatch_size,
        gradient_accumulation_steps=ft_config.gradient_accumulation_steps,
        warmup_steps=ft_config.warmup_steps,
        optim="adamw_torch",
        num_train_epochs=ft_config.epochs,
        learning_rate=ft_config.lr,
        fp16=True,
        logging_steps=ft_config.logging_steps,
        evaluation_strategy="no",
        save_strategy="steps",
        eval_steps=None,
        save_steps=ft_config.save_steps,
        output_dir=ft_config.lora_out_dir,
        save_total_limit=ft_config.save_total_limit,
        load_best_model_at_end=False,
        ddp_find_unused_parameters=False if ft_config.ddp else None,
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=data.train_data,
        eval_dataset=data.val_data,
        args=training_arguments,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False

    # Set Verbose
    if ft_config.verbose:
        transformers.logging.set_verbosity_info()

    # Run Trainer
    if ft_config.resume_checkpoint:
        print('Resuming from {} ...'.format(ft_config.resume_checkpoint))
        trainer.train(ft_config.resume_checkpoint)
    else:
        trainer.train()

    print('Train completed.')

# Save Model
model.save_pretrained(ft_config.lora_out_dir)

if ft_config.checkpoint:
    print("Warning: Merge model + LoRA and save the whole checkpoint not implemented yet.")

print('Model Saved.')
