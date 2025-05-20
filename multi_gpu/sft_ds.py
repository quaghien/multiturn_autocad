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

# Set environment variables for NCCL
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"

# def load_and_process_datasets(num_proc):
#     train_en = load_dataset("TruongSinhAI/DEEPCAD-Text2Json-EnVi", split="train_en", num_proc=3)
#     train_vi = load_dataset("TruongSinhAI/DEEPCAD-Text2Json-EnVi", split="train_vi", num_proc=3)
#     val_en = load_dataset("TruongSinhAI/DEEPCAD-Text2Json-EnVi", split="val_en")
#     val_vi = load_dataset("TruongSinhAI/DEEPCAD-Text2Json-EnVi", split="val_vi")

#     train_dataset = interleave_datasets([train_en, train_vi])
#     val_dataset = interleave_datasets([val_en, val_vi])

#     def format_example(sample):
#         return {
#             "prompt": sample["prompt_fix"],
#             "completion": sample["output"]
#         }

#     train_dataset = train_dataset.map(
#         format_example, 
#         batched=True,
#         remove_columns=train_dataset.column_names,
#         num_proc = num_proc
#     )

#     val_dataset = val_dataset.map(
#         format_example, 
#         batched=True,
#         remove_columns=val_dataset.column_names,
#         num_proc = num_proc
#     )
#     return train_dataset, val_dataset

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

    accelerator = Accelerator()
    accelerator.print(f"{AcceleratorState()}")

    with open('default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    dtype = torch.bfloat16
    max_length = 6000
    num_epochs = 2
    learning_rate = 1e-5
    num_proc = 30
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
        use_cache=False,
    )

    train_dataset = load_dataset("wanhin/DEEPCAD-completion-sft", split="train")

    training_args = SFTConfig(
        dataset_num_proc = num_proc,
        max_length = max_length,
        padding_free = False,
        completion_only_loss = True,
        output_dir=f"./train_results/{output_dir}",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        bf16=True,
        max_steps=10,
        save_strategy="steps",
        save_steps=4,
        logging_steps=1,
        remove_unused_columns=True,
        use_liger_kernel=True,
        gradient_checkpointing=True,
        max_grad_norm=0.1,
        optim="adamw_8bit",
        warmup_steps=1,
        report_to=None
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()

    save_model(model, tokenizer, f"./final_model/{output_dir}")

    model.push_to_hub(f"wanhin/{output_dir}")
    tokenizer.push_to_hub(f"wanhin/{output_dir}")

    my_model, my_tokenizer = load_trained_model(f"./final_model/{output_dir}")

    prompt ='''<objective>
    Generate a JSON file describing the sketching and extrusion steps needed to construct a 3D CAD model. Generate only the JSON file, no other text.
    </objective>

    <instruction>
    You will be given a natural language description of a CAD design task. Your goal is to convert it into a structured JSON representation, which includes sketch geometry and extrusion operations.

    The JSON must follow the structure defined in the <template> section, and the extrusion <operation> must be one of the following:

    1. <NewBodyFeatureOperation>: Creates a new solid body.
    2. <JoinFeatureOperation>: Fuses the shape with an existing body.
    3. <CutFeatureOperation>: Subtracts the shape from an existing body.
    4. <IntersectFeatureOperation>: Keeps only the overlapping volume between the new shape and existing body.

    Ensure all coordinates, geometry, and extrusion depths are extracted accurately from the input.
    </instruction>

    <description>
    **Part 1: Three-Dimensional Rectangular Prism with Tapered Top and Bottom** Begin by creating a new coordinate system for part 1 with the following parameters: * Euler Angles: [0.0, 0.0, -90.0] * Translation Vector: [0.0, 0.75, 0.0] For the 2D sketch, create a face (face\\_1) on the XY plane: * Loop 1: + Line 1: Start Point [0.0, 0.0], End Point [0.5, 0.0] + Line 2: Start Point [0.5, 0.0], End Point [0.5, 0.25] + Line 3: Start Point [0.5, 0.25], End Point [0.25, 0.25] + Line 4: Start Point [0.25, 0.25], End Point [0.25, 0.625] + Line 5: Start Point [0.25, 0.625], End Point [0.0, 0.625] + Line 6: Start Point [0.0, 0.625], End Point [0.0, 0.0] After drawing the 2D sketch, scale it using the sketch\\_scale parameter with a value of 0.625. Then, transform the scaled 2D sketch into 3D using the provided euler angles and translation vector. Extrude the sketch with the following parameters: * extrude\\_depth\\_towards\\_normal: 0.75 * extrude\\_depth\\_opposite\\_normal: 0.0 * sketch\\_scale: 0.625 This completes the first part, which is a three-dimensional rectangular prism with a slightly tapered top and bottom. The part has a length of 0.625, a width of 0.75, and a height of 0.625. **Part 2: Three-Dimensional Rectangular Prism with Flat Top and Bottom** Create a new coordinate system for part 2 with the following parameters: * Euler Angles: [0.0, 0.0, 0.0] * Translation Vector: [0.25, 0.5, 0.25] For the 2D sketch, create a face (face\\_1) on the XY plane: * Loop 1: + Line 1: Start Point [0.0, 0.25], End Point [0.25, 0.0] + Line 2: Start Point [0.25, 0.0], End Point [0.25, 0.25] + Line 3: Start Point [0.25, 0.25], End Point [0.0, 0.25] After drawing the 2D sketch, extrude it with the following parameters: * extrude\\_depth\\_towards\\_normal: 0.0 * extrude\\_depth\\_opposite\\_normal: 0.5 * sketch\\_scale: 0.25 Use the cut boolean operation for this part. This results in a three-dimensional rectangular prism with a flat top and bottom. The sides are parallel, and the top and bottom faces are perpendicular to the sides. The dimensions of the prism are 2 units by 4 units by 6 units. The part has a length of 0.25, a width of 0.25, and a height of 0.5. **Part 3: Rectangular Prism with Flat Top and Bottom** Create a new coordinate system for part 3 with the following parameters: * Euler Angles: [-90.0, 0.0, -90.0] * Translation Vector: [0.25, 0.5, 0.25] For the 2D sketch, create a face (face\\_1) on the YZ plane: * Loop 1: + Line 1: Start Point [0.0, 0.375], End Point [0.25, 0.0] + Line 2: Start Point [0.25, 0.0], End Point [0.25, 0.375] + Line 3: Start Point [0.25, 0.375], End Point [0.0, 0.375] After drawing the 2D sketch, extrude it with the following parameters: * extrude\\_depth\\_towards\\_normal: 0.0 * extrude\\_depth\\_opposite\\_normal: 0.75 * sketch\\_scale: 0.375 Use the cut boolean operation for this part. This results in a rectangular prism with a flat top and bottom. The sides are parallel, and the top and bottom faces are perpendicular to the sides. The dimensions of the prism are 3 units in length, 2 units in width, and 1 unit in height. The part has a length of 0.75, a width of 0.375, and a height of 0.375. **Final CAD Model:** By combining all the parts, you will create a final CAD model, which is a three-dimensional rectangular prism with a tapered top and bottom and a hollowed-out center. This model combines the characteristics of each individual part, forming a single, cohesive structure. * The model has a slightly tapered top and bottom with the dimensions of the outermost prism being 2 units by 6 units by 1.25 units. * The hollowed-out center is formed by combining parts 2 and 3, creating an internal space in the middle of the overall structure. * This structure can be used as a container or a base for other assemblies, highlighting the importance of mastering the creation of individual parts and their integration into a cohesive final design.
    </description>'''

    model_inputs = my_tokenizer([prompt], return_tensors="pt").to(my_model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens= 8192
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(response)
# finally:
#     # Ensure wandb is properly closed
#     if wandb.run is not None:
#         wandb.finish()
#         print("wandb finished")

# accelerate launch --config_file {path/to/config/my_config_file.yaml} {script_name.py}