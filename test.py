import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from dotenv import load_dotenv
from huggingface_hub import login
from trl import apply_chat_template

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
    model_path = "wanhin/cad_reasoning_1_2e"  # Update this path to match your saved model

    my_model, my_tokenizer = load_trained_model(model_path)
    # my_model.push_to_hub(f"wanhin/Qwen2.5-7B-Instruct_1e_fullfinetune")
    # my_tokenizer.push_to_hub(f"wanhin/Qwen2.5-7B-Instruct_1e_fullfinetune")

    prompt ='''<objective>
    Generate a JSON file describing the sketching and extrusion steps needed to construct a 3D CAD model based on the description provided. The output should include a reasoning section within the <think> tag and the corresponding JSON in the <json> tag. Do not provide any additional text outside of the tags.
    </objective>

    <instruction>
    You will be given a natural language description of a CAD design task enclosed within <description> </description>. Your task is to:
    1. Analyze the description and extract the relevant geometric and extrusion information.
    2. In the <think> tag, explain how you derived each field and value in the JSON from the description. This includes the geometric properties (e.g., coordinates, shapes) and extrusion operations. The reasoning should clarify how the geometry is mapped to the JSON structure and the chosen extrusion operation.
    3. Based on the reasoning in the <think> tag, generate the corresponding JSON structure for the CAD model in the <json> tag.

    The extrusion <operation> must be one of the following:
    1. <NewBodyFeatureOperation>: Creates a new solid body.
    2. <JoinFeatureOperation>: Fuses the shape with an existing body.
    3. <CutFeatureOperation>: Subtracts the shape from an existing body.
    4. <IntersectFeatureOperation>: Keeps only the overlapping volume between the new shape and existing body.
    </instruction>

    <description>
    **Construct a cylindrical object with a central hole, resembling a washer or a ring.** Create a new coordinate system with euler angles of (0, 0, 0) and a translation vector of (0, 0, 0). Begin by creating a new sketch on the XY plane defined by the coordinate system. Within this sketch, create the first face (face\\_1). This face consists of two loops, loop\\_1 and loop\\_2. 1. Loop\\_1: - Create a circle (circle\\_1) with a center at (0.375, 0.375) and a radius of 0.375. 2. Loop\\_2: - Create a circle (circle\\_1) with a center at (0.375, 0.375) and a radius of 0.1364. Apply a scaling factor of 0.75 to the entire sketch. Transform the 2D sketch into a 3D sketch using a rotation of 0°, 0°, and 0° around the X, Y, and Z axes, respectively, and a translation vector of (0, 0, 0). Extrude the 2D sketch to generate the 3D model, using an extrusion depth of 0.2386 towards the normal and no extrusion of 0.0 towards the opposite direction of the normal. The height of this part is 0.2386, the width is 0.75, and the length is 0.75. This part represents a cylindrical object with a central hole, resembling a washer or a ring.
    </description>'''

    # prompt ='''Bạn có hiểu tiếng việt không?'''

    # # Format the input using chat template
    # example = {
    #     "prompt": [{"role": "user", "content": "What color is the sky?"}],
    #     "completion": [{"role": "assistant", "content": "It is blue."}]
    # }

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant that generates CAD model descriptions in JSON format."},
        {"role": "user", "content": prompt}
    ]
    text = my_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    # completion_content = '''{"final_shape": "", "parts": {"part_1": {"coordinate_system": {"Euler Angles": [0.0, 0.0, -90.0], "Translation Vector": [0.0, 0.075, 0.0]}, "sketch": {"face_1": {"loop_1": {"line_1": {"Start Point": [0.0, 0.0], "End Point": [0.75, 0.0]}, "line_2": {"Start Point": [0.75, 0.0], "End Point": [0.75, 0.75]}, "line_3": {"Start Point": [0.75, 0.75], "End Point": [0.0, 0.75]}, "line_4": {"Start Point": [0.0, 0.75], "End Point": [0.0, 0.0]}}}}, "extrusion": {"extrude_depth_towards_normal": 0.075, "extrude_depth_opposite_normal": 0.0, "sketch_scale": 0.75, "operation": "NewBodyFeatureOperation"}, "description": {"shape": "", "length": 0.75, "width": 0.075, "height": 0.75}}}}'''

    # example = {
    #     "prompt": [{"role": "user", "content": "prompt_setup"}],
    #     "completion": [{"role": "assistant", "content": "training_content"}]
    # }

    # formatted_prompt = apply_chat_template(example, my_tokenizer)

    # print(formatted_prompt)

    #Prepare model inputs
    model_inputs = my_tokenizer(text, return_tensors="pt").to(my_model.device)

    # Generate response
    print("Generating response...")
    generated_ids = my_model.generate(
        **model_inputs,
        max_new_tokens=10000,
        do_sample=False,
        # temperature=0.7,
        # top_p=0.9,
        # top_k=20,
        # pad_token_id=my_tokenizer.pad_token_id,
        # eos_token_id=my_tokenizer.eos_token_id
    )

    # Process generated output
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = my_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print()
    print(response)