import json
import re
import os
from datasets import load_dataset
from tqdm import tqdm
import google.generativeai as genai
from huggingface_hub import login
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure Gemini API
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")

genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel('gemini-2.0-flash')

# Configure HuggingFace login
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN not found in environment variables. Please set it in your .env file.")

login(token=hf_token)

def create_prompt_gemini(prompt, completion):
    """Create the Gemini prompt with proper escaping for JSON content"""
    return f"""
I have a CAD JSON file (used to create CAD) and a description of the JSON:

{prompt}

{completion}

*S1: Remove the excess description immediately after the extrusion description.

**Example:
<description>
Part 1: Three-dimensional rectangular prism with a flat top and bottom. Create a new coordinate system with the following properties: * Euler Angles: (0.0, 0.0, -90.0) * Translation Vector: (0.0, 0.0316, 0.0). Draw a 2D sketch on the XY plane of the coordinate system. Create a face containing one closed loop made up of 4 lines: * Line 1: Start Point (0.0, 0.0), End Point (0.75, 0.0) * Line 2: Start Point (0.75, 0.0), End Point (0.75, 0.6772) * Line 3: Start Point (0.75, 0.6772), End Point (0.0, 0.6772) * Line 4: Start Point (0.0, 0.6772), End Point (0.0, 0.0). Scale the 2D sketch by a factor of 0.75. Transform the scaled 2D sketch into a 3D sketch using the defined coordinate system. Extrude the 3D sketch by 0.0316 units in the positive Z direction. The height of this part is 0.75 units, the width is 0.0316 units, and the length is 0.75 units. This completes the three-dimensional rectangular prism part.
</description>

-> You need to remove the line: "The height of this part is 0.75 units, the width is 0.0316 units, and the length is 0.75 units. This completes the three-dimensional rectangular prism part." Keep the </description> tag.
The output you need to return after removing the excess description would look like:

<description>
Part 1: Three-dimensional rectangular prism with a flat top and bottom. Create a new coordinate system with the following properties: * Euler Angles: (0.0, 0.0, -90.0) * Translation Vector: (0.0, 0.0316, 0.0). Draw a 2D sketch on the XY plane of the coordinate system. Create a face containing one closed loop made up of 4 lines: * Line 1: Start Point (0.0, 0.0), End Point (0.75, 0.0) * Line 2: Start Point (0.75, 0.0), End Point (0.75, 0.6772) * Line 3: Start Point (0.75, 0.6772), End Point (0.0, 0.6772) * Line 4: Start Point (0.0, 0.6772), End Point (0.0, 0.0). Scale the 2D sketch by a factor of 0.75. Transform the scaled 2D sketch into a 3D sketch using the defined coordinate system. Extrude the 3D sketch by 0.0316 units in the positive Z direction.
</description>
**

*S2: Check if the newly created description matches the json. If it does, create "<valid>Yes</valid>", if it doesn't, create "<valid>No</valid>"

*S3: Create a sample reasoning data enclosed in <think> ... </think>. The reasoning data should follow two steps:
Step 1: Reason out the components that will be in the JSON based on the given description.
Step 2: Check the logic, arithmetic correctness, and make corrections (if necessary) from Step 1.

**Example 1 sample:

***Input:
<json> {{"parts": {{"part_1": {{"coordinate_system": {{"Euler Angles": [0.0, 0.0, -90.0], "Translation Vector": [0.0, 0.0316, 0.0]}}, "sketch": {{"face_1": {{"loop_1": {{"line_1": {{"Start Point": [0.0, 0.0], "End Point": [0.75, 0.0]}}, "line_2": {{"Start Point": [0.75, 0.0], "End Point": [0.75, 0.6772]}}, "line_3": {{"Start Point": [0.75, 0.6772], "End Point": [0.0, 0.6772]}}, "line_4": {{"Start Point": [0.0, 0.6772], "End Point": [0.0, 0.0]}}}}}}, "extrusion": {{"extrude_depth_towards_normal": 0.0316, "extrude_depth_opposite_normal": 0.0, "sketch_scale": 0.75, "operation": "NewBodyFeatureOperation"}}}}}}}} </json>
<description> Part 1: Three-dimensional rectangular prism with a flat top and bottom. Create a new coordinate system with the following properties: * Euler Angles: (0.0, 0.0, -90.0) * Translation Vector: (0.0, 0.0316, 0.0). Draw a 2D sketch on the XY plane of the coordinate system. Create a face containing one closed loop made up of 4 lines: * Line 1: Start Point (0.0, 0.0), End Point (0.75, 0.0) * Line 2: Start Point (0.75, 0.0), End Point (0.75, 0.6772) * Line 3: Start Point (0.75, 0.6772), End Point (0.0, 0.6772) * Line 4: Start Point (0.0, 0.6772), End Point (0.0, 0.0). Scale the 2D sketch by a factor of 0.75. Transform the scaled 2D sketch into a 3D sketch using the defined coordinate system. Extrude the 3D sketch by 0.0316 units in the positive Z direction. </description>

***Output:
S1:
<description>
Part 1: Three-dimensional rectangular prism with a flat top and bottom. Create a new coordinate system with the following properties: * Euler Angles: (0.0, 0.0, -90.0) * Translation Vector: (0.0, 0.0316, 0.0). Draw a 2D sketch on the XY plane of the coordinate system. Create a face containing one closed loop made up of 4 lines: * Line 1: Start Point (0.0, 0.0), End Point (0.75, 0.0) * Line 2: Start Point (0.75, 0.0), End Point (0.75, 0.6772) * Line 3: Start Point (0.75, 0.6772), End Point (0.0, 0.6772) * Line 4: Start Point (0.0, 0.6772), End Point (0.0, 0.0). Scale the 2D sketch by a factor of 0.75. Transform the scaled 2D sketch into a 3D sketch using the defined coordinate system. Extrude the 3D sketch by 0.0316 units in the positive Z direction.
</description>
S2:
<valid>Yes</valid>
S3:
<think>
***Step 1: Infer the components that will be in the json based on the provided description:

parts:From the description, we have a single part labeled "part_1" describing a three-dimensional rectangular prism. This will be reflected in the json as "part_1".

part_1: (Rectangular Prism)
-coordinate_system:
--Euler Angles: [0.0, 0.0, -90.0] (Derived from the description: "Create a new coordinate system with the following properties: * Euler Angles: (0.0, 0.0, -90.0)").
--Translation Vector: [0.0, 0.0316, 0.0] (Derived from the description: "Translation Vector: (0.0, 0.0316, 0.0)").
-sketch:
--face_1:
---loop_1:
----line_1: Start [0.0, 0.0], End [0.75, 0.0] (Derived from the description: "Line 1: Start Point (0.0, 0.0), End Point (0.75, 0.0)").
----line_2: Start [0.75, 0.0], End [0.75, 0.6772] (Derived from the description: "Line 2: Start Point (0.75, 0.0), End Point (0.75, 0.6772)").
----line_3: Start [0.75, 0.6772], End [0.0, 0.6772] (Derived from the description: "Line 3: Start Point (0.75, 0.6772), End Point (0.0, 0.6772)").
----line_4: Start [0.0, 0.6772], End [0.0, 0.0] (Derived from the description: "Line 4: Start Point (0.0, 0.6772), End Point (0.0, 0.0)").
-extrusion:
--extrude_depth_towards_normal: 0.0316 (Derived from the description: "Extrude the 3D sketch by 0.0316 units in the positive Z direction").
--extrude_depth_opposite_normal: 0.0 (Not mentioned in the description, but inferred as 0 because no extrusion is required in the opposite direction).
--sketch_scale: 0.75 (Derived from the description: "Scale the 2D sketch by a factor of 0.75").
--operation: "NewBodyFeatureOperation" (Inferred from the description: "Create a 3D model of the rectangular prism").

***Step 2: Check the logic, arithmetic correctness, and make corrections (if necessary) from Step 1:

-Check the lines (line_1, line_2, line_3, line_4):
line_1 (Start [0.0, 0.0], End [0.75, 0.0]) connects correctly with line_2 (Start [0.75, 0.0], End [0.75, 0.6772]).
line_2 (Start [0.75, 0.0], End [0.75, 0.6772]) connects correctly with line_3 (Start [0.75, 0.6772], End [0.0, 0.6772]).
line_3 (Start [0.75, 0.6772], End [0.0, 0.6772]) connects correctly with line_4 (Start [0.0, 0.6772], End [0.0, 0.0]).
line_4 (Start [0.0, 0.6772], End [0.0, 0.0]) connects correctly with line_1 (Start [0.0, 0.0], End [0.75, 0.0]).

--Result: The sketch loop is valid and closed.

-Check the extrusion:
Extrusion of the 3D sketch with a depth of 0.0316 units in the positive Z direction, as specified in the description.
No extrusion is needed in the opposite direction, so extrude_depth_opposite_normal is correctly set to 0.

--Result: The extrusion is correct and matches the requirements.

-Check dimensions consistency:
The values for the Start/End points of the lines are correctly represented and match the 2D sketch described.
The extrusion depth and scale factor are consistent with the description.

--Result: The dimensions are correct and consistent.

-Correction: This inferred data is accurate and does not require further modifications.
</think>
**
"""

def extract_tags(text, tag_name):
    """Extract content from XML-like tags"""
    pattern = f"<{tag_name}>(.*?)</{tag_name}>"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[0] if matches else None

def process_dataset(start=0, end=500, old_split="range_0_500_en"):
    """Process the dataset with Gemini model"""
    
    # Load dataset
    print(f"Loading dataset 'wanhin/cad_hqh1' with split '{old_split}'...")
    dataset = load_dataset("wanhin/cad_hqh1", split=old_split)
    
    # Filter dataset based on start and end indices
    filtered_dataset = dataset.select(range(start, min(end, len(dataset))))
    print(f"Processing {len(filtered_dataset)} samples from index {start} to {min(end, len(dataset))}")
    
    new_data = []
    invalid_indices = []
    
    # Process each sample
    for idx, sample in enumerate(tqdm(filtered_dataset, desc="Processing samples")):
        try:
            # Create prompt for Gemini
            prompt_text = create_prompt_gemini(sample['prompt'], sample['completion'])
            
            # Get response from Gemini
            response = model.generate_content(prompt_text)
            response_text = response.text
            
            # Extract validation result
            valid_tag = extract_tags(response_text, "valid")
            if valid_tag and valid_tag.strip() == "Yes":
                # Extract description and reasoning
                description_tag = extract_tags(response_text, "description")
                think_tag = extract_tags(response_text, "think")
                
                if description_tag and think_tag:
                    # Create new sample
                    new_sample = {
                        "description": f"<description>{description_tag}</description>",
                        "reasoning": f"<think>{think_tag}</think>",
                        "completion": sample['completion']
                    }
                    new_data.append(new_sample)
                else:
                    invalid_indices.append(start + idx)
            else:
                invalid_indices.append(start + idx)
                
        except Exception as e:
            print(f"Error processing sample {start + idx}: {str(e)}")
            invalid_indices.append(start + idx)
    
    # Save invalid indices to JSON
    invalid_file = f"valid_{old_split}_{start}_{end}.json"
    with open(invalid_file, 'w') as f:
        json.dump({"invalid_indices": invalid_indices}, f, indent=2)
    
    print(f"Processed {len(new_data)} valid samples")
    print(f"Found {len(invalid_indices)} invalid samples")
    print(f"Invalid indices saved to {invalid_file}")
    
    # Create new dataset
    new_split_name = f"{old_split}_{start}_{end}"
    
    # Push to hub
    print(f"Uploading dataset to 'wanhin/cad_reason_1' with split '{new_split_name}'...")
    from datasets import Dataset, DatasetDict
    
    new_dataset = Dataset.from_list(new_data)
    dataset_dict_obj = DatasetDict({new_split_name: new_dataset})
    
    dataset_dict_obj.push_to_hub("wanhin/cad_reason_1")
    print("Dataset uploaded successfully!")

if __name__ == "__main__":
    # Example usage
    process_dataset(start=0, end=1500, old_split="range_1000_1500_en")