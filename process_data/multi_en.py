import json
import re
import os
import time
import threading
from datasets import load_dataset, Dataset
from huggingface_hub import login
from dotenv import load_dotenv
import google.generativeai as genai
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Prompt template for single sample - clear and comprehensive
single_prompt_template = '''
PROCESSING RULES FOR EACH <input_i>:

1. Analyze each sample json input to count the number of parts.
2. Create custom multi-turn data according to the number of parts in each sample json input: (json_1, prompt_1, think_1 -> ... -> json_i, prompt_i, think_i)

DETAILED RULES:
- Create progressive json: json_1 (only part_1), json_2 (part_1 + part_2), ..., json_n (part_1 + part_2 + ... + part_n) (if a part doesn't exist, don't create it)
- Place each json within <json_i></json_i> tags
- Create user input descriptions for each part within <prompt_i></prompt_i> tags (describe shapes, no technical specifications) (describe user modifications from previous part to achieve this part)
- Create 2-step reasoning within <think_i></think_i> tags:
  * Step 1: Reason about the components that will be included in the JSON based on the provided description
  * Step 2: Check logic, mathematical correctness, and perform modifications (if necessary) from Step 1

OUTPUT FORMAT FOR EACH SAMPLE (create each part individually, don't create if part doesn't exist):
<sample_i>
<json_1>[json with part_1]</json_1>
<prompt_1>[user description to create part_1]</prompt_1>
<think_1>[2-step reasoning to create json_1]</think_1>
<json_2>[json with part_1 + part_2]</json_2>
<prompt_2>[user description to modify part_1 to get part 1 + part 2]</prompt_2> (only create if part_2 exists)
<think_2>[2-step reasoning to create json_2]</think_2>
...continue to json_n (if exists) ...
</sample_i>

SPECIFIC EXAMPLE:
***
INPUT:
<input_1>
{"parts":{"part_1":{"coordinate_system":{"Euler Angles":[0.0,0.0,0.0],"Translation Vector":[0.0,0.0,0.1607]},"sketch":{"face_1":{"loop_1":{"line_1":{"Start Point":[0.0,0.0],"End Point":[0.3214,0.0]},"line_2":{"Start Point":[0.3214,0.0],"End Point":[0.3214,0.3214]},"line_3":{"Start Point":[0.3214,0.3214],"End Point":[0.0,0.3214]},"line_4":{"Start Point":[0.0,0.3214],"End Point":[0.0,0.0]}}}},"extrusion":{"extrude_depth_towards_normal":0.0804,"extrude_depth_opposite_normal":0.0804,"sketch_scale":0.3214,"operation":"NewBodyFeatureOperation"}},"part_2":{"coordinate_system":{"Euler Angles":[0.0,0.0,0.0],"Translation Vector":[0.3214,0.0,0.1607]},"sketch":{"face_1":{"loop_1":{"line_1":{"Start Point":[0.0,0.0],"End Point":[0.2679,0.0]},"line_2":{"Start Point":[0.2679,0.0],"End Point":[0.2679,0.3214]},"line_3":{"Start Point":[0.2679,0.3214],"End Point":[0.0,0.3214]},"line_4":{"Start Point":[0.0,0.3214],"End Point":[0.0,0.0]}}}},"extrusion":{"extrude_depth_towards_normal":0.0268,"extrude_depth_opposite_normal":0.0268,"sketch_scale":0.3214,"operation":"JoinFeatureOperation"}},"part_3":{"coordinate_system":{"Euler Angles":[0.0,0.0,0.0],"Translation Vector":[0.5893,0.0,0.1607]},"sketch":{"face_1":{"loop_1":{"arc_1":{"Start Point":[0.0,0.0],"Mid Point":[0.1607,0.1607],"End Point":[0.0,0.3214]},"line_1":{"Start Point":[0.0,0.3214],"End Point":[0.0,0.0]}}}},"extrusion":{"extrude_depth_towards_normal":0.0268,"extrude_depth_opposite_normal":0.0268,"sketch_scale":0.3214,"operation":"JoinFeatureOperation"}}}}
</input_1>

OUTPUT:

<sample_1>
<json_1>
{"parts":{"part_1":{"coordinate_system":{"Euler Angles":[0.0,0.0,0.0],"Translation Vector":[0.0,0.0,0.0]},"sketch":{"face_1":{"loop_1":{"line_1":{"Start Point":[0.0,0.0],"End Point":[0.75,0.0]},"line_2":{"Start Point":[0.75,0.0],"End Point":[0.75,0.2812]},"line_3":{"Start Point":[0.75,0.2812],"End Point":[0.0,0.2812]},"line_4":{"Start Point":[0.0,0.2812],"End Point":[0.0,0.0]}},"loop_2":{"circle_1":{"Center":[0.1716,0.1406],"Radius":0.0769}},"loop_3":{"circle_1":{"Center":[0.5784,0.1406],"Radius":0.0769}}},"face_2":{"loop_1":{"circle_1":{"Center":[0.1716,0.1406],"Radius":0.0769}},"loop_2":{"circle_1":{"Center":[0.1716,0.1406],"Radius":0.0675}}},"face_3":{"loop_1":{"circle_1":{"Center":[0.5784,0.1406],"Radius":0.0769}},"loop_2":{"circle_1":{"Center":[0.5784,0.1406],"Radius":0.0675}}}},"extrusion":{"extrude_depth_towards_normal":0.0562,"extrude_depth_opposite_normal":0.0,"sketch_scale":0.75,"operation":"NewBodyFeatureOperation"}}}}
</json_1>
<prompt_1>
I want to create a rectangular flat plate. On the plate, there should be two circular holes, and on both side faces of the plate, at the positions of those holes, I want to have two concentric circular grooves.
</prompt_1>
<think_1>
Step 1: Reason about the components that will be included in the JSON based on the provided description
From the requirement "I want to create a rectangular flat plate. On the plate, there should be two circular holes, and on both side faces of the plate, at the positions of those holes, I want to have two concentric circular grooves", I will analyze and determine the necessary elements, listing the specific JSON keys.

**parts**: Top-level key to contain the model components.
**part_1**: Defines the first part of the model.
**coordinate_system**: Component to position and orient this part in space.
  **Euler Angles**: I will assume no initial rotation. I set it as `[0.0,0.0,0.0]`.
  **Translation Vector**: I will assume a default position at the coordinate origin. I set it as `[0.0,0.0,0.0]`.
**sketch**: Component defining the base 2D sketch.
  **face_1**: Represents the plane containing the main sketch of the plate and holes.
    **loop_1**: Represents the outer rectangle of the plate.
      **line_1, line_2, line_3, line_4**: I will set the `Start Point` and `End Point` to create a rectangle. For example: `line_1:{"Start Point":[0.0,0.0],"End Point":[0.75,0.0]}`, `line_2:{"Start Point":[0.75,0.0],"End Point":[0.75,0.2812]}`, `line_3:{"Start Point":[0.75,0.2812],"End Point":[0.0,0.2812]}`, `line_4:{"Start Point":[0.0,0.2812],"End Point":[0.0,0.0]}`.
    **loop_2, loop_3**: Represent the two circular holes on the plate.
      **circle_1**: The circular hole shape.
        **Center**: I will set the center positions for the two holes, for example: `loop_2:{"circle_1":{"Center":[0.1716,0.1406]}}` and `loop_3:{"circle_1":{"Center":[0.5784,0.1406]}}`.
        **Radius**: I will set the radius for the two holes, for example: `0.0769`.
  **face_2, face_3**: Represent the two concentric circular grooves on the side faces. Each groove will be a separate `face`.
    **loop_1**: Represents the outer ring of the groove.
      **circle_1**: The outer ring shape.
        **Center**: I will set the center of the outer ring to coincide with the corresponding hole center, for example: `face_2:{"loop_1":{"circle_1":{"Center":[0.1716,0.1406]}}}` and `face_3:{"loop_1":{"circle_1":{"Center":[0.5784,0.1406]}}}`.
        **Radius**: I will set the radius of the outer ring, for example: `0.0769`.
    **loop_2**: Represents the inner ring of the groove.
      **circle_1**: The inner ring shape.
        **Center**: I will set the center of the inner ring to coincide with the outer ring center, for example: `face_2:{"loop_2":{"circle_1":{"Center":[0.1716,0.1406]}}}` and `face_3:{"loop_2":{"circle_1":{"Center":[0.5784,0.1406]}}}`.
        **Radius**: I will set the radius of the inner ring, slightly smaller than the outer ring, for example: `0.0675`.
**extrusion**: Component defining the extrusion process to create 3D shape.
  **extrude_depth_towards_normal**: I will set a reasonable thickness for the plate, for example `0.0562`.
  **extrude_depth_opposite_normal**: I reason that the extrusion process occurs in only one direction. I set it as `0.0`.
  **sketch_scale**: I will set a scale so the plate doesn't get distorted, for example `0.75`.
  **operation**: I reason that this is an operation to create a new complete 3D body. I set it as `"NewBodyFeatureOperation"`.

Step 2: Check logic, mathematical correctness, and perform modifications (if necessary) from Step 1
I check the reasonableness of the parameters reasoned from Step 1.

**coordinate_system**:
  **Euler Angles:[0.0,0.0,0.0]**: Reasoning: No rotation. Conclusion: Reasonable.
  **Translation Vector:[0.0,0.0,0.0]**: Reasoning: At coordinate origin. Conclusion: Reasonable.
**sketch**:
  **face_1**:
    **loop_1 (rectangle)**: The coordinate points create a reasonable rectangle. Conclusion: Reasonable.
    **loop_2, loop_3 (circular holes)**: The centers and radii of the holes are placed symmetrically and reasonably on the plate. Conclusion: Reasonable.
  **face_2, face_3 (concentric grooves)**:
    **loop_1, loop_2**: The concentric circles with different radii create grooves. The center positions coinciding with the holes on `face_1` is reasonable for the description "on both side faces of the plate, at the positions of those holes". Conclusion: Reasonable.
**extrusion**:
  **extrude_depth_towards_normal:0.0562**: Reasoning: Determines the thickness of the plate. Conclusion: Reasonable.
  **extrude_depth_opposite_normal:0.0**: Reasoning: Only extrude in one direction. Conclusion: Reasonable.
  **sketch_scale:0.75**: Reasoning: Sketch scale so the plate doesn't get distorted. Conclusion: Reasonable.
  **operation:"NewBodyFeatureOperation"**: Reasoning: Create new body. Conclusion: Reasonable.
Perform modifications (if necessary): No parameters need modification.
</think_1>

<json_2>
{"parts":{"part_1":{"coordinate_system":{"Euler Angles":[0.0,0.0,0.0],"Translation Vector":[0.0,0.0,0.0]},"sketch":{"face_1":{"loop_1":{"line_1":{"Start Point":[0.0,0.0],"End Point":[0.75,0.0]},"line_2":{"Start Point":[0.75,0.0],"End Point":[0.75,0.2812]},"line_3":{"Start Point":[0.75,0.2812],"End Point":[0.0,0.2812]},"line_4":{"Start Point":[0.0,0.2812],"End Point":[0.0,0.0]}},"loop_2":{"circle_1":{"Center":[0.1716,0.1406],"Radius":0.0769}},"loop_3":{"circle_1":{"Center":[0.5784,0.1406],"Radius":0.0769}}},"face_2":{"loop_1":{"circle_1":{"Center":[0.1716,0.1406],"Radius":0.0769}},"loop_2":{"circle_1":{"Center":[0.1716,0.1406],"Radius":0.0675}}},"face_3":{"loop_1":{"circle_1":{"Center":[0.5784,0.1406],"Radius":0.0769}},"loop_2":{"circle_1":{"Center":[0.5784,0.1406],"Radius":0.0675}}}},"extrusion":{"extrude_depth_towards_normal":0.0562,"extrude_depth_opposite_normal":0.0,"sketch_scale":0.75,"operation":"NewBodyFeatureOperation"}},"part_2":{"coordinate_system":{"Euler Angles":[0.0,0.0,0.0],"Translation Vector":[0.0947,0.0637,0.0562]},"sketch":{"face_1":{"loop_1":{"circle_1":{"Center":[0.0769,0.0769],"Radius":0.0769}},"loop_2":{"circle_1":{"Center":[0.0769,0.0769],"Radius":0.0675}}},"face_2":{"loop_1":{"circle_1":{"Center":[0.0769,0.0769],"Radius":0.0675}}},"face_3":{"loop_1":{"circle_1":{"Center":[0.4837,0.0769],"Radius":0.0769}},"loop_2":{"circle_1":{"Center":[0.4837,0.0769],"Radius":0.0675}}},"face_4":{"loop_1":{"circle_1":{"Center":[0.4837,0.0769],"Radius":0.0675}}}},"extrusion":{"extrude_depth_towards_normal":0.0,"extrude_depth_opposite_normal":0.0187,"sketch_scale":0.5606,"operation":"CutFeatureOperation"}}}}
</json_2>
<prompt_2>
I want to add two more circular grooves on the two remaining sides of the rectangular plate, located near the original circular holes. These grooves should also be annular and shallow.
</prompt_2>
<think_2>
Step 1: Reason about the components that will be included in the JSON based on the provided description
From the requirement "I want to add two more circular grooves on the two remaining sides of the rectangular plate, located near the original circular holes. These grooves should also be annular and shallow", I will analyze and determine the necessary elements, listing the specific JSON keys for `part_2`.

**part_2**: Defines the two new circular grooves.
**coordinate_system**: Component to position and orient this part in space.
  **Euler Angles**: I will assume no initial rotation. I set it as `[0.0,0.0,0.0]`.
  **Translation Vector**: I will set the position of the grooves. The Z-coordinate will equal the plate thickness (`0.0562`) so they are on the top surface of the plate. X and Y will be shifted slightly so the grooves are near the original circular holes but on the other side, for example `[0.0947,0.0637,0.0562]`.
**sketch**: Component defining the 2D annular sketch (two concentric circles) for each groove.
  **face_1**: Represents the first groove.
    **loop_1**: Represents the outer ring of the groove.
      **circle_1**: The outer ring shape.
        **Center**: I will set the center of the outer ring, for example `[0.0769,0.0769]`.
        **Radius**: I will set the radius of the outer ring, for example `0.0769`.
    **loop_2**: Represents the inner ring of the groove.
      **circle_1**: The inner ring shape.
        **Center**: I will set the center of the inner ring to coincide with the outer ring center, for example `[0.0769,0.0769]`.
        **Radius**: I will set the radius of the inner ring, for example `0.0675`.
  **face_2**: Represents the cut portion of the first groove (if there is an inner ring). I reason that this is a circle concentric with the outer ring to create an annular groove.
    **loop_1**:
      **circle_1**:
        **Center**: I will set the center of the inner ring to coincide with the outer ring center, for example `[0.0769,0.0769]`.
        **Radius**: I will set the radius of the inner ring, for example `0.0675`.
  **face_3**: Represents the second groove.
    **loop_1**: Represents the outer ring of the groove.
      **circle_1**: The outer ring shape.
        **Center**: I will set the center of the outer ring, for example `[0.4837,0.0769]`.
        **Radius**: I will set the radius of the outer ring, for example `0.0769`.
    **loop_2**: Represents the inner ring of the groove.
      **circle_1**: The inner ring shape.
        **Center**: I will set the center of the inner ring to coincide with the outer ring center, for example `[0.4837,0.0769]`.
        **Radius**: I will set the radius of the inner ring, for example `0.0675`.
  **face_4**: Represents the cut portion of the second groove (if there is an inner ring). I reason that this is a circle concentric with the outer ring to create an annular groove.
    **loop_1**:
      **circle_1**:
        **Center**: I will set the center of the inner ring to coincide with the outer ring center, for example `[0.4837,0.0769]`.
        **Radius**: I will set the radius of the inner ring, for example `0.0675`.
**extrusion**: Component defining the extrusion process to create grooves (cutting).
  **extrude_depth_towards_normal**: I reason that the groove cutting process will go against the surface normal direction. I set it as `0.0`.
  **extrude_depth_opposite_normal**: I will set the groove depth to be very shallow, for example `0.0187`.
  **sketch_scale**: I will set a scale so the grooves don't get distorted, for example `0.5606`.
  **operation**: I reason that this is an operation to cut the existing body. I set it as `"CutFeatureOperation"`.

Step 2: Check logic, mathematical correctness, and perform modifications (if necessary) from Step 1
I check the reasonableness of the parameters reasoned from Step 1.

**coordinate_system**:
  **Euler Angles:[0.0,0.0,0.0]**: Reasoning: No rotation. Conclusion: Reasonable.
  **Translation Vector:[0.0947,0.0637,0.0562]**: Reasoning: Position on the top surface of the plate with slight displacement. Conclusion: Reasonable.
**sketch**:
  **face_1, face_3 (outer rings)**:
    **circle_1**: Center and radius create outer rings. Conclusion: Reasonable.
  **face_1, face_3 (inner rings)**:
    **circle_1**: Center and radius create concentric inner rings that are smaller. Conclusion: Reasonable.
  **face_2, face_4**: These grooves seem to be defined by a single circle (only having `loop_1` and `circle_1`), but the description is "annular". It's possible that `loop_2` of `face_1` and `face_3` already created the inner rings, and `face_2` and `face_4` are just copies of those inner rings or are sketch planes for cutting. Based on the JSON, `face_2` and `face_4` only have one `loop` with one `circle` having radius `0.0675`, which is exactly the radius of the inner rings from `face_1` and `face_3`. This indicates that `face_2` and `face_4` are used to define the cutting holes for the annular grooves. Conclusion: Reasonable for creating annular grooves.
**extrusion**:
  **extrude_depth_towards_normal:0.0**: Reasoning: No extrusion towards the normal direction. Conclusion: Reasonable.
  **extrude_depth_opposite_normal:0.0187**: Reasoning: Very shallow groove depth. Conclusion: Reasonable.
  **sketch_scale:0.5606**: Reasoning: Sketch scale so grooves don't get distorted. Conclusion: Reasonable.
  **operation:"CutFeatureOperation"**: Reasoning: Perform cutting operation. Conclusion: Reasonable.
Perform modifications (if necessary): The structure of `face_2` and `face_4` in the JSON seems somewhat redundant if the purpose is to create annular grooves. Typically, annular grooves are defined by two concentric circles within the same `loop` or `face`. However, with the current structure, it can still work if the CAD software understands that cutting based on the smaller circle will create a hollow section inside the larger circle. For the purpose of reasoning from the description, I will assume this is how the user wants to create annular grooves.
</think_2>
</sample_1>
***
'''

# Simplified error logging - only track failed sample IDs
error_log_lock = threading.Lock()
failed_sample_ids = []

def log_failed_sample_ids(sample_ids):
    """Log failed sample IDs in batch"""
    global failed_sample_ids
    with error_log_lock:
        failed_sample_ids.extend(sample_ids)

def save_error_log(dataset_name, split_name, start_index):
    """Save simple error log with failed sample IDs"""
    global failed_sample_ids
    with error_log_lock:
        if failed_sample_ids:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"failed_ids_{dataset_name.replace('/', '_')}_{split_name}_{start_index}_{timestamp}.json"
            
            error_data = {
                "dataset": dataset_name,
                "split": split_name,
                "start_index": start_index,
                "failed_count": len(failed_sample_ids),
                "failed_sample_ids": failed_sample_ids,
                "timestamp": timestamp
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(error_data, f, indent=2, ensure_ascii=False)
            
            print(f"Failed IDs saved: {filename}")
            print(f"Total failed samples: {len(failed_sample_ids)}")
        else:
            print("No failed samples to log!")

# Prompt template for batch processing (dynamic number of samples)
def create_batch_prompt_template(batch_size, single_template, samples_data):
    """Create batch prompt template with dynamic number of samples"""
    
    # Create input sections with actual data (no placeholders)
    input_sections = []
    for i in range(1, batch_size + 1):
        sample_data = samples_data.get(f'input_{i}', '')
        input_sections.append(f"<input_{i}>\n{sample_data}\n</input_{i}>")
    
    # Create output format sections
    output_sections = []
    for i in range(1, batch_size + 1):
        output_sections.append(f"<sample_{i}>\noutput_{i}\n</sample_{i}>")
    
    template = f'''
I will provide you with {batch_size} sample JSON inputs and you need to process each sample separately.

JSON INPUT SAMPLES:
{chr(10).join(input_sections)}

REQUIRED OUTPUT:
Process each sample JSON input according to the example below and place results in corresponding tags:

{chr(10).join(output_sections)}

IMPORTANT NOTES:
- Each <sample_i> contains complete: json, prompt, think
- All tags must be created according to the rules in the example below.

PROCESSING RULES AND EXAMPLE:
{single_template}
'''
    return template

def load_environment():
    """Load environment variables"""
    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        raise ValueError("Please set HF_TOKEN environment variable")
    login(token=hf_token)

def extract_json_from_completion(completion_text):
    """Extract JSON from completion field"""
    try:
        json_match = re.search(r'<json>\s*(.*?)\s*</json>', completion_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            return json.loads(json_str)
        return None
    except:
        return None

def count_parts_in_json(json_data):
    """Count number of parts in JSON"""
    if not json_data or 'parts' not in json_data:
        return 0
    
    parts = json_data['parts']
    part_count = 0
    for key in parts.keys():
        if re.match(r'part_\d+', key):
            part_count += 1
    
    return part_count

def survey_dataset(dataset_name, split_name, expected_parts=None):
    """Survey dataset to find maximum number of parts and check consistency"""
    print(f"Loading dataset {dataset_name} with split {split_name}...")
    
    dataset = load_dataset(dataset_name, split=split_name)
    print(f"Dataset loaded with {len(dataset)} samples")
    
    max_parts = 0
    part_counts = {}
    valid_samples = 0
    consistent_samples = 0
    
    for idx, sample in enumerate(dataset):
        completion = sample.get('completion', '')
        json_data = extract_json_from_completion(completion)
        
        if json_data:
            parts_count = count_parts_in_json(json_data)
            if parts_count > 0:
                valid_samples += 1
                max_parts = max(max_parts, parts_count)
                
                if parts_count in part_counts:
                    part_counts[parts_count] += 1
                else:
                    part_counts[parts_count] = 1
                
                # Check consistency with expected parts
                if expected_parts is not None and parts_count == expected_parts:
                    consistent_samples += 1
    
    print(f"\nSurvey Results:")
    print(f"Total samples: {len(dataset)}")
    print(f"Valid samples with JSON: {valid_samples}")
    print(f"Maximum number of parts found: {max_parts}")
    
    # Check consistency
    is_consistent = False
    if expected_parts is not None:
        print(f"\nExpected parts: {expected_parts}")
        print(f"Samples with expected parts: {consistent_samples}/{valid_samples}")
        
        if consistent_samples == valid_samples and len(part_counts) == 1 and expected_parts in part_counts:
            is_consistent = True
            print(f"✅ Dataset is CONSISTENT - All samples have exactly {expected_parts} parts")
        else:
            print(f"❌ Dataset is NOT CONSISTENT")
            print(f"Part count distribution:")
            for count in sorted(part_counts.keys()):
                percentage = (part_counts[count] / valid_samples) * 100 if valid_samples > 0 else 0
                print(f"  {count} parts: {part_counts[count]} samples ({percentage:.1f}%)")
    else:
        print(f"\nPart count distribution:")
        for count in sorted(part_counts.keys()):
            percentage = (part_counts[count] / valid_samples) * 100 if valid_samples > 0 else 0
            print(f"  {count} parts: {part_counts[count]} samples ({percentage:.1f}%)")
    
    return max_parts, part_counts, is_consistent

def extract_tags_from_response(response_text, max_parts):
    """Extract json_n, prompt_n, think_n tags from Gemini response"""
    extracted_data = {}
    
    # Extract all possible tags (up to max_parts + some extra to ensure coverage)
    for i in range(1, max_parts + 5):  # Add buffer
        # Extract json_n
        json_pattern = rf'<json_{i}>\s*(.*?)\s*</json_{i}>'
        json_match = re.search(json_pattern, response_text, re.DOTALL)
        if json_match:
            extracted_data[f'json_{i}'] = json_match.group(1).strip()
        
        # Extract prompt_n
        prompt_pattern = rf'<prompt_{i}>\s*(.*?)\s*</prompt_{i}>'
        prompt_match = re.search(prompt_pattern, response_text, re.DOTALL)
        if prompt_match:
            extracted_data[f'input_{i}'] = prompt_match.group(1).strip()
        
        # Extract think_n
        think_pattern = rf'<think_{i}>\s*(.*?)\s*</think_{i}>'
        think_match = re.search(think_pattern, response_text, re.DOTALL)
        if think_match:
            extracted_data[f'think_{i}'] = think_match.group(1).strip()
    
    return extracted_data

def extract_samples_from_batch_response(response_text, batch_size):
    """Extract individual sample responses from batch response"""
    samples = {}
    
    for i in range(1, batch_size + 1):  # Extract sample_1 to sample_batch_size
        sample_pattern = rf'<sample_{i}>\s*(.*?)\s*</sample_{i}>'
        sample_match = re.search(sample_pattern, response_text, re.DOTALL)
        if sample_match:
            sample_content = sample_match.group(1).strip()
            samples[f'sample_{i}'] = {
                'content': sample_content,
                'length': len(sample_content)
            }
    
    return samples

def process_batch_with_gemini(samples_batch, api_key, max_parts, batch_size, thread_id, start_sample_index):
    """Process a batch of samples with Gemini API"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Get actual batch size (might be less than max batch_size for last batch)
        actual_batch_size = len(samples_batch)
        
        # Prepare input data for batch prompt
        input_data = {}
        for i, sample in enumerate(samples_batch, 1):
            input_data[f'input_{i}'] = sample['completion']
        
        # Create batch prompt with data embedded directly (no .format() needed)
        prompt = create_batch_prompt_template(actual_batch_size, single_prompt_template, input_data)
        
        response = model.generate_content(prompt)
        
        if not response or not hasattr(response, 'text'):
            return [None] * len(samples_batch)
            
        response_text = response.text
        
        if not response_text or len(response_text.strip()) == 0:
            return [None] * len(samples_batch)
        
        # Extract individual samples from batch response
        extracted_samples = extract_samples_from_batch_response(response_text, actual_batch_size)
        
        if not extracted_samples:
            return [None] * len(samples_batch)
        
        # Process each sample and extract tags
        results = []
        for i, sample in enumerate(samples_batch, 1):
            sample_key = f'sample_{i}'
            
            if sample_key in extracted_samples:
                sample_content = extracted_samples[sample_key]['content']
                sample_length = extracted_samples[sample_key]['length']
                
                # Extract tags from this sample's content
                extracted_data = extract_tags_from_response(sample_content, max_parts)
                extracted_data['new_length'] = sample_length
                
                results.append(extracted_data)
            else:
                # If sample not found, return empty result
                results.append({'new_length': 0})
        
        return results
            
    except Exception as e:
        return [None] * len(samples_batch)

def process_thread(dataset_samples, thread_id, api_key_index, max_parts, samples_per_thread, wait_seconds, start_idx, end_idx, batch_size):
    """Process samples in a thread using batch processing with intelligent timing"""
    
    # Load API key
    api_key = os.getenv(f'GEMINI_API_KEY_{api_key_index}')
    if not api_key:
        return []
    
    processed_data = []
    thread_failed_ids = []
    
    # Process samples in batches
    dataset_list = list(dataset_samples)
    total_samples = len(dataset_list)
    
    for batch_start in tqdm(range(0, total_samples, batch_size), desc=f"Thread {thread_id}"):
        batch_end = min(batch_start + batch_size, total_samples)
        samples_batch = dataset_list[batch_start:batch_end]
        actual_batch_size = len(samples_batch)
        
        # Calculate actual sample indices
        start_sample_index = start_idx + batch_start
        
        # Measure API call timing
        api_start_time = time.time()
        
        # Process batch with Gemini
        batch_results = process_batch_with_gemini(samples_batch, api_key, max_parts, batch_size, thread_id, start_sample_index)
        
        api_end_time = time.time()
        api_elapsed_time = api_end_time - api_start_time
        
        batch_failed_ids = []
        
        # Process each sample in the batch
        for idx, (sample, result) in enumerate(zip(samples_batch, batch_results)):
            # Extract sample ID 
            sample_id = sample.get('id', f'unknown_{start_sample_index + idx}')
            
            # Count parts in original completion
            original_json = extract_json_from_completion(sample['completion'])
            original_parts_count = count_parts_in_json(original_json) if original_json else 0
            
            # Create new sample with original data (always include, even if processing failed)
            new_sample = {
                'id': sample.get('id', f'unknown_{start_sample_index + idx}'),
                'original_completion': sample['completion']
            }
            
            # Track sample success/failure and count generated samples
            generated_samples_count = 0
            is_failed = False
            
            if result is None:
                is_failed = True
                new_sample['new_length'] = 0
                # Initialize all fields with empty strings for failed samples
                for i in range(1, max_parts + 1):
                    new_sample[f'input_{i}'] = ''
                    new_sample[f'think_{i}'] = ''
                    new_sample[f'json_{i}'] = ''
            else:
                # Add new_length field
                new_sample['new_length'] = result.get('new_length', 0)
                
                # Initialize all possible fields based on max_parts and count generated samples
                for i in range(1, max_parts + 1):
                    input_val = result.get(f'input_{i}', '')
                    think_val = result.get(f'think_{i}', '')
                    json_val = result.get(f'json_{i}', '')
                    
                    new_sample[f'input_{i}'] = input_val
                    new_sample[f'think_{i}'] = think_val
                    new_sample[f'json_{i}'] = json_val
                    
                    # Count if all three fields are non-empty (successful generation)
                    if input_val.strip() and think_val.strip() and json_val.strip():
                        generated_samples_count += 1
                
                # Check if generated correct number of parts or no output
                if generated_samples_count != original_parts_count or generated_samples_count == 0:
                    is_failed = True
            
            # Track failed sample ID
            if is_failed:
                batch_failed_ids.append(sample_id)
            
            processed_data.append(new_sample)
        
        # Log failed IDs for this batch
        if batch_failed_ids:
            thread_failed_ids.extend(batch_failed_ids)
        
        # Intelligent wait based on actual API time
        if batch_end < total_samples:  # Don't wait after last batch
            if api_elapsed_time < wait_seconds:
                remaining_wait = wait_seconds - api_elapsed_time
                time.sleep(remaining_wait)
    
    # Log all failed IDs for this thread
    if thread_failed_ids:
        log_failed_sample_ids(thread_failed_ids)
        print(f"Thread {thread_id}: {len(thread_failed_ids)} failed samples")
    
    return processed_data

def create_multi_turn_dataset(dataset_name, split_name, new_dataset_name, num_threads, samples_per_thread, wait_seconds, start_index=0, batch_size=5, expected_parts=None, api_key_start_index=1):
    """Main function to create multi-turn dataset"""
    
    # Load environment
    load_environment()
    
    # Survey dataset first
    print("Surveying dataset...")
    max_parts, distribution, is_consistent = survey_dataset(dataset_name, split_name, expected_parts)
    
    if not max_parts:
        print("No valid samples found!")
        return
    
    # Check if we should proceed with API processing
    if expected_parts is not None:
        if not is_consistent:
            print(f"\n❌ PROCESSING STOPPED!")
            print(f"Dataset is not consistent with expected {expected_parts} parts.")
            print(f"Cannot proceed with Gemini API processing.")
            print(f"Please choose a dataset with all samples having exactly {expected_parts} parts.")
            return
        else:
            print(f"\n✅ Dataset validation passed!")
            print(f"All samples have exactly {expected_parts} parts. Proceeding with processing...")
    else:
        print(f"\n⚠️ No expected parts specified. Proceeding with max_parts={max_parts}...")
    
    # Calculate total fields: id + original_completion + new_length + (input_i + think_i + json_i) * max_parts  
    total_fields = 3 + (max_parts * 3)
    print(f"Creating dataset fields for up to {max_parts} parts")
    print(f"Dataset will have {total_fields} fields: 3 original + {max_parts}*3 = {total_fields} total fields")
    
    # Load dataset
    dataset = load_dataset(dataset_name, split=split_name)
    
    # Calculate samples per thread starting from start_index
    available_samples = len(dataset) - start_index
    total_samples = min(available_samples, num_threads * samples_per_thread)
    actual_samples_per_thread = total_samples // num_threads
    
    print(f"Dataset has {len(dataset)} samples, starting from index {start_index}")
    print(f"Processing {total_samples} samples with {num_threads} threads ({actual_samples_per_thread} samples per thread)")
    print(f"Using batch processing: {batch_size} samples per API call")
    print(f"API keys: GEMINI_API_KEY_{api_key_start_index} to GEMINI_API_KEY_{api_key_start_index + num_threads - 1}")
    
    # Split dataset for threads, starting from start_index
    thread_datasets = []
    for i in range(num_threads):
        thread_start_idx = start_index + (i * actual_samples_per_thread)
        thread_end_idx = min(start_index + ((i + 1) * actual_samples_per_thread), start_index + total_samples)
        thread_samples = dataset.select(range(thread_start_idx, thread_end_idx))
        thread_datasets.append((thread_samples, thread_start_idx, thread_end_idx))
    
    # Process with threads
    all_processed_data = []
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Create mapping of future to thread info
        future_to_info = {}
        
        for thread_id, (thread_samples, start_idx, end_idx) in enumerate(thread_datasets):
            api_key_index = api_key_start_index + thread_id  # Use consecutive API keys starting from api_key_start_index
            
            future = executor.submit(
                process_thread,
                thread_samples,
                thread_id,
                api_key_index,
                max_parts,
                actual_samples_per_thread,
                wait_seconds,
                start_idx,
                end_idx,
                batch_size
            )
            future_to_info[future] = (thread_id, start_idx, end_idx)
        
        # Collect results
        for future in as_completed(future_to_info.keys()):
            thread_id, start_idx, end_idx = future_to_info[future]
            try:
                thread_results = future.result()
                all_processed_data.extend(thread_results)
                print(f"Thread {thread_id} completed: {len(thread_results)} samples (samples {start_idx}-{end_idx})")
            except Exception as e:
                print(f"Thread {thread_id} failed with exception: {str(e)}")

    # Save error log
    save_error_log(dataset_name, split_name, start_index)
    
    if not all_processed_data:
        print("No data was processed successfully!")
        return

    print(f"Total processed samples: {len(all_processed_data)}")
    
    # Ensure all samples have the same structure before creating dataset
    if all_processed_data:
        # Create a template with all required fields
        field_template = {
            'id': '',
            'original_completion': '',
            'new_length': 0
        }
        for i in range(1, max_parts + 1):
            field_template[f'input_{i}'] = ''
            field_template[f'think_{i}'] = ''
            field_template[f'json_{i}'] = ''
        
        # Ensure all samples have all fields
        for sample in all_processed_data:
            for field_name, default_value in field_template.items():
                if field_name not in sample:
                    sample[field_name] = default_value
        
        print(f"Dataset structure validated. Each sample has {len(field_template)} fields.")
    
    # Create new dataset
    new_dataset = Dataset.from_list(all_processed_data)
    
    # Create split name based on consistency
    end_index = start_index + total_samples
    if is_consistent and expected_parts is not None:
        new_split_name = f"{split_name}_{start_index}_{end_index}_{expected_parts}parts"
        print(f"✅ Dataset is consistent with {expected_parts} parts")
    else:
        new_split_name = f"{split_name}_{start_index}_{end_index}"
        if expected_parts is not None:
            print(f"❌ Dataset is not consistent with expected {expected_parts} parts")
        else:
            print(f"ℹ️ No expected parts specified")
    
    print(f"Final split name will be: {new_split_name}")
    print(f"Pushing dataset to {new_dataset_name} with split {new_split_name}")
    
    new_dataset.push_to_hub(new_dataset_name, split=new_split_name)
    print("Dataset pushed successfully!")
    print(f"📤 Uploaded split: {new_split_name}")

# Example usage
if __name__ == "__main__":
    # Parameters
    dataset_name = "wanhin/cad_reasoning_part"
    split_name = "en_part_1"
    new_dataset_name = "wanhin/output_reasoning_part"
    num_threads = 4
    samples_per_thread = 9000  # Total samples per thread
    wait_seconds = 70  # Minimum interval between API calls (intelligent wait - only waits if API call < this time)
    start_index = 72000  # Starting index parameter
    batch_size = 9  # Number of samples per batch/API call
    expected_parts = 1  # Expected number of parts in dataset (None if not checking)
    api_key_start_index = 2  # Start from GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc.
    
    create_multi_turn_dataset(
        dataset_name=dataset_name,
        split_name=split_name,
        new_dataset_name=new_dataset_name,
        num_threads=num_threads,
        samples_per_thread=samples_per_thread,
        wait_seconds=wait_seconds,
        start_index=start_index,
        batch_size=batch_size,
        expected_parts=expected_parts,
        api_key_start_index=api_key_start_index
    )