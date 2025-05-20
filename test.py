import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from dotenv import load_dotenv
from huggingface_hub import login

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
    # Load environment variables and login to HF
    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        raise ValueError("Please set HF_TOKEN environment variable")
    login(token=hf_token)

    # Load the trained model
    model_path = "./final_model/Qwen2.5-7B-Instruct_2epoch_6000maxlength"  # Update this path to match your saved model
    # model_path = "Qwen/Qwen2.5-7B-Instruct"
    my_model, my_tokenizer = load_trained_model(model_path)
    # my_model.push_to_hub(f"wanhin/Qwen2.5-7B-Instruct_2epoch_6000maxlength")
    # my_tokenizer.push_to_hub(f"wanhin/Qwen2.5-7B-Instruct_2epoch_6000maxlength")

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

    # Format the input using chat template
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant that generates CAD model descriptions in JSON format."},
        {"role": "user", "content": prompt}
    ]
    
    formatted_prompt = my_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Prepare model inputs
    model_inputs = my_tokenizer([formatted_prompt], return_tensors="pt").to(my_model.device)

    # Generate response
    print("Generating response...")
    generated_ids = my_model.generate(
        **model_inputs,
        max_new_tokens=6000,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=my_tokenizer.pad_token_id,
        eos_token_id=my_tokenizer.eos_token_id
    )

    # Process generated output
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = my_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("\nGenerated Response:")
    print(response)