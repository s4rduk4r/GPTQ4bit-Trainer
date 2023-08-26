import json

from . import WORK_MODE, EWorkModes


if WORK_MODE == EWorkModes.FINETUNE:

    from .Finetune4bConfig import Finetune4bConfig
    
    def get_config(path: str) -> Finetune4bConfig:

        with open(path, "r") as f:
            config = json.load(f)
            return Finetune4bConfig(
                dataset=config["dataset"], 
                ds_type=config["ds_type"], 
                lora_out_dir=config["lora_out_dir"], 
                lora_apply_dir=config["lora_apply_dir"],
                resume_checkpoint=config["resume_checkpoint"],
                llama_q4_config_dir=config["llama_q4_config_dir"],
                llama_q4_model=config["llama_q4_model"],
                mbatch_size=config["mbatch_size"],
                batch_size=config["batch_size"],
                epochs=config["epochs"], 
                lr=config["lr"],
                cutoff_len=config["cutoff_len"],
                lora_r=config["lora_r"], 
                lora_alpha=config["lora_alpha"], 
                lora_dropout=config["lora_dropout"],
                lora_target_modules=config["lora_target_modules"],
                val_set_size=config["val_set_size"],
                gradient_checkpointing=config["grad_chckpt"],
                gradient_checkpointing_ratio=config["grad_chckpt_ratio"],
                warmup_steps=config["warmup_steps"],
                save_steps=config["save_steps"],
                save_total_limit=config["save_total_limit"],
                logging_steps=config["logging_steps"],
                checkpoint=config["checkpoint"],
                skip=config["skip"],
                verbose=config["verbose"],
                txt_row_thd=config["txt_row_thd"],
                use_eos_token=config["use_eos_token"],
                groupsize=config["groupsize"],
                local_rank=config["local_rank"],
                config_file_path=path
            )

elif WORK_MODE == EWorkModes.INFERENCE:
    
    from .Inference4bConfig import Inference4bConfig
    
    def get_config(path: str) -> Inference4bConfig:
        with open(path, "r") as f:
            config = json.load(f)
            return Inference4bConfig(
                llama_q4_config_dir=config["llama_q4_config_dir"],
                llama_q4_model=config["llama_q4_model"],
                lora_apply_dir=config["lora_apply_dir"],
                chat_mode=config["chat_mode"],
                groupsize=config["groupsize"],
                offloading=config["offloading"],
                offload_folder=config["offload_folder"],
                config_file_path=path,
                device_map=config["device_map"],
                max_memory=config["max_memory"]
            )
