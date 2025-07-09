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

    wandb.init(project="lora-reasoning")
    
    dtype = torch.bfloat16
    max_length = 32000
    num_epochs = 1
    learning_rate = 1e-4
    num_proc = 16
    model_name = "wanhin/cad_reasoning_1_2e"
    output_dir = f"{model_name.split('/')[-1]}_lora_reasoning"

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
        r=32,
        lora_alpha=64,
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
    
    # Enable training mode explicitly
    model.train()
    
    # The PEFT model should automatically handle requires_grad for LoRA parameters
    # Let's verify but not override the automatic handling
    print("\n***Calculating trainable parameters...")
    trainable_params = 0
    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(f"Trainable params: {trainable_params:,}")
    print(f"Total params: {total_params:,}")
    print(f"Trainable%: {100 * trainable_params / total_params:.4f}%\n")
    
    model.print_trainable_parameters()

    raw_train_dataset = load_dataset("wanhin/train_lora_multiturn", split="train", keep_in_memory=True)
    raw_val_dataset = load_dataset("wanhin/train_lora_multiturn", split="validation", keep_in_memory=True)
    
    # raw_train_dataset = raw_train_dataset.to_pandas()
    # raw_train_dataset = [{"prompt": str(item["prompt"]), "completion": str(item["completion"])} for _, item in raw_train_dataset[310000:].iterrows()]

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
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=32,
        learning_rate=learning_rate,
        bf16=True,
        save_strategy="steps",
        save_steps=3,
        save_total_limit=2,
        save_safetensors=True,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=1,
        remove_unused_columns=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
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
        print("Training failed")
        print(f"Error: {e}") 
finally:
    if wandb.run is not None:
        wandb.finish()
        print("wandb finished")
