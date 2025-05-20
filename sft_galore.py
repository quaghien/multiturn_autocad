import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from liger_kernel.transformers import apply_liger_kernel_to_qwen2
from datasets import load_dataset, Dataset
from dotenv import load_dotenv
from huggingface_hub import login
from trl import SFTConfig, SFTTrainer, apply_chat_template
import wandb
import os

def save_model(model, tokenizer, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"Model saved to {output_dir}")

def load_trained_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        use_safetensors=True,
        use_cache=False
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
    )
    print(f'Loaded model and tokenizer from {model_path}')
    return model, tokenizer

if __name__ == "__main__":
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

        wandb.init(project="sft-stage1-cad-completion")
        
        dtype = torch.bfloat16
        max_length = 6000
        num_epochs = 3
        learning_rate = 1e-5
        num_proc = 30
        model_name = "Qwen/Qwen2.5-7B-Instruct"
        output_dir = f"{model_name.split('/')[-1]}_{num_epochs}epoch_{max_length}maxlength"

        print(f"Training model: {output_dir}")

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            model_max_length=max_length,
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
            device_map="cuda",
            use_cache=False
        )

        raw_dataset = load_dataset("wanhin/DEEPCAD-completion-sft", split="train")
        
        dataset_dict = {
            "prompt": [[{"role": "user", "content": item["prompt"]}] for item in raw_dataset],
            "completion": [[{"role": "assistant", "content": item["completion"]}] for item in raw_dataset]
        }
        
        train_dataset = Dataset.from_dict(dataset_dict)
        train_dataset = train_dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})
        
        # print(train_dataset)
        # print("Dataset format example:")
        # print(train_dataset[0])

        training_args = SFTConfig(
            dataset_num_proc = num_proc,
            max_length = max_length,
            completion_only_loss = True,
            output_dir=f"./train_results/{output_dir}",
            num_train_epochs=num_epochs,
            per_device_train_batch_size=4,
            learning_rate=learning_rate,
            bf16=True,
            save_strategy="epoch",
            # max_steps=100,
            # save_strategy="steps",
            # save_steps=100,
            logging_steps=50,
            use_liger_kernel=True,
            gradient_checkpointing=True,
            optim="galore_adamw_8bit_layerwise",
            optim_target_modules=[r".*.attn.*", r".*.mlp.*"],
            optim_args = "rank=16, scale=1.0, update_proj_gap=200",
            warmup_steps=50,
            report_to="wandb"
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset
        )

        trainer.train()

        save_model(model, tokenizer, f"./final_model/{output_dir}")

        # model.push_to_hub(f"wanhin/{output_dir}")
        # tokenizer.push_to_hub(f"wanhin/{output_dir}")

    finally:
        if wandb.run is not None:
            wandb.finish()
            print("wandb finished")
