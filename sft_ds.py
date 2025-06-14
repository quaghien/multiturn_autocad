import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset
from dotenv import load_dotenv
from huggingface_hub import login
from trl import SFTConfig, SFTTrainer, apply_chat_template
from accelerate import Accelerator
import wandb
import os
import torch.distributed as dist


try:
    accelerator = Accelerator()
    device = accelerator.device
    print(f"Using device: {device}")

    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        raise ValueError("Please set HF_TOKEN environment variable")
    login(token=hf_token)

    wandb_token = os.getenv('WANDB_API_KEY')
    if not wandb_token:
        raise ValueError("Please set WANDB_API_KEY environment variable")
    wandb.login(key=wandb_token)

    wandb.init(project="stage1-cad-29-05")

    dtype = torch.bfloat16
    max_length = 6000
    num_epochs = 1
    learning_rate = 2e-5
    num_proc = 16
    model_name = "1e_full" #   meta-llama/Llama-3.1-8B-Instruct    Qwen/Qwen2.5-7B-Instruct
    output_dir = f"{model_name.split('/')[-1]}_{num_epochs}epoch_stage1_10-06"

    print(f"Training model: {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        model_max_length=max_length,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        use_cache=False
    )

    raw_train_dataset = load_dataset("wanhin/DEEPCAD-stage1", split="train_en_vi", keep_in_memory=True)
    raw_val_dataset = load_dataset("wanhin/DEEPCAD-stage1", split="val_en_vi_400", keep_in_memory=True)

    # raw_train_dataset = raw_train_dataset.to_pandas()
    # raw_train_dataset = [{"prompt": str(item["prompt"]), "completion": str(item["completion"])} for _, item in raw_train_dataset[300000:].iterrows()]

    train_dataset_dict = {
        "prompt": [[{"role": "user", "content": item["prompt"]}] for item in raw_train_dataset],
        "completion": [[{"role": "assistant", "content": item["completion"]}] for item in raw_train_dataset]
    }
    
    val_dataset_dict = {
        "prompt": [[{"role": "user", "content": item["prompt"]}] for item in raw_val_dataset],
        "completion": [[{"role": "assistant", "content": item["completion"]}] for item in raw_val_dataset]
    }
    
    train_dataset = Dataset.from_dict(train_dataset_dict)
    train_dataset = train_dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer}, num_proc=num_proc)
    
    val_dataset = Dataset.from_dict(val_dataset_dict)
    val_dataset = val_dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer}, num_proc=num_proc)

    training_args = SFTConfig(
        dataset_num_proc = num_proc,
        dataloader_num_workers=2,
        max_length = max_length,
        completion_only_loss = True,
        output_dir=f"./train_results/{output_dir}",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=32,
        learning_rate=learning_rate,
        bf16=True,
        save_strategy="steps",
        save_steps=250,
        save_total_limit=2,
        save_safetensors=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=250,
        remove_unused_columns=True,
        use_liger_kernel=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        warmup_steps=0,
        report_to="wandb",
        save_only_model=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    try:
        trainer.train()
    except Exception as e:
        print(e)
        print("Training failed")
        exit(1)
    
    accelerator.wait_for_everyone()
    print("Training and saving completed!")

finally:
    if wandb.run is not None:
        wandb.finish()
        print("wandb finished")

# accelerate launch --config_file {path/to/config/my_config_file.yaml} {script_name.py}