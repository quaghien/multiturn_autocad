import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from liger_kernel.transformers import apply_liger_kernel_to_qwen2
from datasets import load_dataset, interleave_datasets
import os
from dotenv import load_dotenv
from huggingface_hub import login
from trl import SFTConfig, SFTTrainer

def load_and_process_datasets(num_proc):
    train_en = load_dataset("TruongSinhAI/DEEPCAD-Text2Json-EnVi", split="train_en", num_proc=3)
    train_vi = load_dataset("TruongSinhAI/DEEPCAD-Text2Json-EnVi", split="train_vi", num_proc=3)
    val_en = load_dataset("TruongSinhAI/DEEPCAD-Text2Json-EnVi", split="val_en", num_proc=1)
    val_vi = load_dataset("TruongSinhAI/DEEPCAD-Text2Json-EnVi", split="val_vi", num_proc=1)

    train_dataset = interleave_datasets([train_en, train_vi])
    val_dataset = interleave_datasets([val_en, val_vi])

    def format_example(sample):
        return {
            "prompt": sample["prompt_fix"],
            "completion": sample["output"]
        }

    train_dataset = train_dataset.map(
        format_example, 
        batched=True,
        remove_columns=train_dataset.column_names,
        num_proc = num_proc
    )

    val_dataset = val_dataset.map(
        format_example, 
        batched=True,
        remove_columns=val_dataset.column_names,
        num_proc = num_proc
    )
    return train_dataset, val_dataset

def save_model(model, tokenizer, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"Model saved to {output_dir}")

def load_trained_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_safetensors=True,
        attn_implementation="flash_attention_2",
        use_cache=False
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        padding_side="left"
    )
    print(f'Loaded model and tokenizer from {model_path}')
    return model, tokenizer

if __name__ == "__main__":
    
    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        raise ValueError("Please set HF_TOKEN environment variable")
    login(token=hf_token)

    dtype = torch.bfloat16
    max_length = 8192
    num_epochs = 2
    learning_rate = 1e-5
    num_proc = 16
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    output_dir = f"{model_name.split('/')[-1]}_{num_epochs}epoch_{max_length}maxlength"

    print(f"Training model: {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        padding_side="left",
        model_max_length=max_length
    )

    apply_liger_kernel_to_qwen2(
        rope=True,
        swiglu=True,
        cross_entropy=False,
        fused_linear_cross_entropy=True,
        rms_norm=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        device_map="auto",
        attn_implementation="flash_attention_2",
        use_cache=False
    )

    train_dataset, val_dataset = load_and_process_datasets(num_proc)

    training_args = SFTConfig(
        dataset_num_proc = num_proc,
        max_length = max_length,
        padding_free = True,
        completion_only_loss = True,
        # neftune_noise_alpha=5,
        output_dir=f"./train_results/{output_dir}",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=learning_rate,
        bf16=True,
        # save_strategy="epoch",
        max_steps=10,
        save_strategy="steps",
        save_steps=4,
        # eval_strategy="steps",
        # eval_steps=4,
        logging_steps=2,
        remove_unused_columns=True,
        use_liger_kernel=True,
        gradient_checkpointing=True,
        max_grad_norm=0.1,
        optim="galore_adamw_8bit_layerwise",
        optim_target_modules=[r".*.attn.*", r".*.mlp.*"],
        optim_args="rank=16, scale=1.0, update_proj_gap=100",
        warmup_steps=2,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()

    save_model(model, tokenizer, f"./final_model/{output_dir}")

    model.push_to_hub(f"hienhq/{output_dir}")
    tokenizer.push_to_hub(f"hienhq/{output_dir}")

    my_model, my_tokenizer = load_trained_model(f"./final_model/{output_dir}")
