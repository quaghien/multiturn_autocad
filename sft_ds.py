import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from liger_kernel.transformers import apply_liger_kernel_to_qwen2
from datasets import load_dataset, interleave_datasets
import os
from dotenv import load_dotenv
from huggingface_hub import login
import yaml
from trl import SFTConfig, SFTTrainer
from accelerate import Accelerator
from accelerate.state import AcceleratorState

def load_and_process_datasets(num_proc):
    train_en = load_dataset("TruongSinhAI/DEEPCAD-Text2Json-EnVi", split="train_en", num_proc=3)
    train_vi = load_dataset("TruongSinhAI/DEEPCAD-Text2Json-EnVi", split="train_vi", num_proc=3)
    val_en = load_dataset("TruongSinhAI/DEEPCAD-Text2Json-EnVi", split="val_en")
    val_vi = load_dataset("TruongSinhAI/DEEPCAD-Text2Json-EnVi", split="val_vi")

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

def save_model(model, tokenizer, trainer, output_dir):
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

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    accelerator = Accelerator()
    accelerator.print(f"{AcceleratorState()}")

    with open('default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    dtype = torch.bfloat16
    max_length = 8192
    num_epochs = 2
    learning_rate = 1e-5
    num_proc = 16
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    output_dir = f"./{model_name.split('/')[-1]}_{num_epochs}epoch_{max_length}max_length"

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
        use_cache=False,
    )

    train_dataset, val_dataset = load_and_process_datasets(num_proc)

    training_args = SFTConfig(
        dataset_num_proc = num_proc,
        max_length = max_length,
        padding_free = True,
        completion_only_loss = True,
        output_dir=f"./train_results/{output_dir}",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        # gradient_accumulation_steps=1,
        learning_rate=learning_rate,
        bf16=True,
        save_strategy="epoch",
        # max_steps=3,
        # save_strategy="steps",
        save_steps=2,
        eval_strategy="steps",
        eval_steps=4,
        logging_steps=2,
        
        remove_unused_columns=True,
        use_liger_kernel=True,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        # optim="adamw_8bit",
        optim="galore_adamw_8bit_layerwise",
        optim_target_modules=[r".*.attn.*", r".*.mlp.*"],
        optim_args="rank=512, scale=1.0, update_proj_gap=100",
        warmup_steps=1,
        deepspeed=config["deepspeed_config"]["deepspeed_config_file"]
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()

    save_model(model, tokenizer, trainer, f"./final_model/{output_dir}")

    model.push_to_hub(f"hienhq/{output_dir}")
    tokenizer.push_to_hub(f"hienhq/{output_dir}")

    my_model, my_tokenizer = load_trained_model(f"./final_model/{output_dir}")



# accelerate launch --config_file {path/to/config/my_config_file.yaml} {script_name.py}