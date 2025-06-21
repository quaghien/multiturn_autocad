import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset
from dotenv import load_dotenv
from huggingface_hub import login
from trl import SFTConfig, SFTTrainer, apply_chat_template
import wandb
import os
from peft import LoraConfig, get_peft_model

try:
    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        raise ValueError("Please set HF_TOKEN environment variable")
    login(token=hf_token)

    wandb_token = os.getenv('WANDB_API_KEY')
    if not wandb_token:
        raise ValueError("Please set WANDB_API_KEY environment variable")
    wandb.login(key=wandb_token)

    wandb.init(project="stage1-cad-completion")
    
    dtype = torch.bfloat16
    max_length = 6000
    num_epochs = 1
    learning_rate = 1e-4
    num_proc = 1
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    output_dir = f"{model_name.split('/')[-1]}_lora_r64"

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
        device_map="cuda",
        use_cache=False
    )

    # Configure LoRA
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    
    # Chỉ mở gradient cho các layer LoRA
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    model.print_trainable_parameters()

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
        gradient_checkpointing=True,
        optim="adamw_torch",
        warmup_steps=10,
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
        print("Training failed")
        print(f"Error: {e}") 

    model.push_to_hub(f"wanhin/{output_dir}")
    tokenizer.push_to_hub(f"wanhin/{output_dir}")

finally:
    if wandb.run is not None:
        wandb.finish()
        print("wandb finished")
