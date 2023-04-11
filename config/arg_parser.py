import os
import argparse

from . import WORK_MODE, EWorkModes

# global WORK_MODE


if WORK_MODE == EWorkModes.FINETUNE:

    from .Finetune4bConfig import Finetune4bConfig

    def parse_commandline():
        parser = argparse.ArgumentParser(
            prog=__file__.split(os.path.sep)[-1],
            description="Produce LoRA in 4bit training",
            usage="%(prog)s [config] [training]\n\nAll arguments are optional"
        )

        parser.add_argument("dataset", nargs="?",
            default="./dataset.json", 
            help="Path to dataset file. Default: %(default)s"
        )

        parser_config = parser.add_argument_group("config")
        parser_training = parser.add_argument_group("training")

        # Config file
        parser.add_argument("--config-file", default=None, required=False,
            help="Read config from file.json. File structure repeats the arguments name passed via command line. Default: %(default)s"
        )

        # Config args group
        parser_config.add_argument("--ds-type", choices=["txt", "alpaca", "gpt4all"], default="alpaca", required=False,
            help="Dataset structure format. Default: %(default)s"
        )
        parser_config.add_argument("--lora-out-dir", default="alpaca_lora", required=False,
            help="Directory to place new LoRA. Default: %(default)s"
        )
        parser_config.add_argument("--lora-apply-dir", default=None, required=False,
            help="Path to directory from which LoRA has to be applied before training. Default: %(default)s"
        )
        parser_training.add_argument("--resume-checkpoint", default=None, required=False,
            help="Resume training from specified checkpoint. Default: %(default)s"
        )
        parser_config.add_argument("--llama-q4-config-dir", default="./llama-13b-4bit/", required=False,
            help="Path to the config.json, tokenizer_config.json, etc. Default: %(default)s"
        )
        parser_config.add_argument("--llama-q4-model", default="./llama-13b-4bit.pt", required=False,
            help="Path to the quantized model in huggingface format. Default: %(default)s"
        )

        # Training args group
        parser_training.add_argument("--mbatch-size", default=1, type=int, help="Micro-batch size. Default: %(default)s")
        parser_training.add_argument("--batch-size", default=2, type=int, help="Batch size. Default: %(default)s")
        parser_training.add_argument("--epochs", default=3, type=int, help="Epochs. Default: %(default)s")
        parser_training.add_argument("--lr", default=2e-4, type=float, help="Learning rate. Default: %(default)s")
        parser_training.add_argument("--cutoff-len", default=256, type=int, help="Default: %(default)s")
        parser_training.add_argument("--lora-r", default=8, type=int, help="Default: %(default)s")
        parser_training.add_argument("--lora-alpha", default=16, type=int, help="Default: %(default)s")
        parser_training.add_argument("--lora-dropout", default=0.05, type=float, help="Default: %(default)s")
        parser_training.add_argument("--lora-target_modules", default="q_proj, v_proj", type=str, help="Default: %(default)s")
        parser_training.add_argument("--grad-chckpt", action="store_true", required=False, help="Use gradient checkpoint. For 30B model. Default: %(default)s")
        parser_training.add_argument("--grad-chckpt-ratio", default=1, type=float, help="Gradient checkpoint ratio. Default: %(default)s")
        parser_training.add_argument("--val-set-size", default=0.2, type=float, help="Validation set size. Default: %(default)s")
        parser_training.add_argument("--warmup-steps", default=50, type=int, help="Default: %(default)s")
        parser_training.add_argument("--save-steps", default=50, type=int, help="Default: %(default)s")
        parser_training.add_argument("--save-total-limit", default=3, type=int, help="Default: %(default)s")
        parser_training.add_argument("--logging-steps", default=10, type=int, help="Default: %(default)s")
        parser_training.add_argument("-c", "--checkpoint", action="store_true", help="Produce checkpoint instead of LoRA. Default: %(default)s")
        parser_training.add_argument("--skip", action="store_true", help="Don't train model. Can be useful to produce checkpoint from existing LoRA. Default: %(default)s")
        parser_training.add_argument("--verbose", action="store_true", help="If output log of training. Default: %(default)s")

        # Data args
        parser_training.add_argument("--txt-row-thd", default=-1, type=int, help="Custom thd for txt rows.")
        parser_training.add_argument("--use-eos-token", action="store_false", help="Use eos token instead of padding with 0. Default: %(default)s")

        # V2 model support
        parser_training.add_argument("--groupsize", type=int, default=-1, help="Groupsize of v2 model, use -1 to load v1 model")

        # Multi GPU Support
        parser_training.add_argument("--local-rank", type=int, default=0, help="local rank if using torch.distributed.launch")

        return vars(parser.parse_args())


    def get_config() -> Finetune4bConfig:
        args = parse_commandline()

        if args["config_file"]:
            from . import config_loader as cl
            return cl.get_config(args["config_file"])

        return Finetune4bConfig(
            dataset=args["dataset"], 
            ds_type=args["ds_type"], 
            lora_out_dir=args["lora_out_dir"], 
            lora_apply_dir=args["lora_apply_dir"],
            resume_checkpoint=args["resume_checkpoint"],
            llama_q4_config_dir=args["llama_q4_config_dir"],
            llama_q4_model=args["llama_q4_model"],
            mbatch_size=args["mbatch_size"],
            batch_size=args["batch_size"],
            epochs=args["epochs"], 
            lr=args["lr"],
            cutoff_len=args["cutoff_len"],
            lora_r=args["lora_r"], 
            lora_alpha=args["lora_alpha"], 
            lora_dropout=args["lora_dropout"],
            lora_target_modules=[w.strip() for w in args["lora_target_modules"].split(",")],
            val_set_size=args["val_set_size"],
            gradient_checkpointing=args["grad_chckpt"],
            gradient_checkpointing_ratio=args["grad_chckpt_ratio"],
            warmup_steps=args["warmup_steps"],
            save_steps=args["save_steps"],
            save_total_limit=args["save_total_limit"],
            logging_steps=args["logging_steps"],
            checkpoint=args["checkpoint"],
            skip=args["skip"],
            verbose=args["verbose"],
            txt_row_thd=args["txt_row_thd"],
            use_eos_token=args["use_eos_token"],
            groupsize=args["groupsize"],
            local_rank=args["local_rank"],
        )

elif WORK_MODE == EWorkModes.INFERENCE:

    from .Inference4bConfig import Inference4bConfig

    def parse_commandline():
        parser = argparse.ArgumentParser(
            prog=__file__.split(os.path.sep)[-1],
            description="Inference with or without LoRA in 4bit",
            usage="%(prog)s [config]\n\nAll arguments are optional"
        )

        # Config file
        parser.add_argument("--config-file", default=None, required=False,
            help="Read config from file.json. File structure repeats the arguments name passed via command line. Default: %(default)s"
        )
        parser.add_argument("--lora-apply-dir", default=None, required=False,
            help="Path to directory from which LoRA has to be applied before training. Default: %(default)s"
        )
        parser.add_argument("--llama-q4-config-dir", default="./llama-13b-4bit/", required=False,
            help="Path to the config.json, tokenizer_config.json, etc. Default: %(default)s"
        )
        parser.add_argument("--llama-q4-model", default="./llama-13b-4bit.safetensors", required=False,
            help="Path to the quantized model in huggingface format. Default: %(default)s"
        )
        parser.add_argument("--groupsize", type=int, default=-1, help="Groupsize of GPTQv2 model. Use -1 for models created without groupsize argument. Default: %(default)s")
        parser.add_argument("--offloading", action="store_true", help="Use offloading for inference")

        return vars(parser.parse_args())

    def get_config() -> Inference4bConfig:
        args = parse_commandline()

        if args["config_file"]:
            from . import config_loader as cl
            return cl.get_config(args["config_file"])

        return Inference4bConfig(
            llama_q4_config_dir=args["llama_q4_config_dir"],
            llama_q4_model=args["llama_q4_model"],
            lora_apply_dir=args["lora_apply_dir"],
            groupsize=args["groupsize"],
            offloading=args["offloading"]
        )
