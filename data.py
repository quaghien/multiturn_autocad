from datasets import load_dataset, interleave_datasets
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()
        hf_token = os.getenv('HF_TOKEN')
        if not hf_token:
            raise ValueError("Please set HF_TOKEN environment variable")
        login(token=hf_token)

train_en = load_dataset("TruongSinhAI/DEEPCAD-Text2Json-EnVi", split="train_en")
train_vi = load_dataset("TruongSinhAI/DEEPCAD-Text2Json-EnVi", split="train_vi")

dataset = interleave_datasets([train_en, train_vi])

messages_prompt ='''<objective>
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
</instruction>'''

def process_example(example):
    # Convert input list to string if it's a list
    input_text = example["input"]
    if isinstance(input_text, list):
        input_text = "\n".join(input_text)
    
    return {
        "prompt": messages_prompt + "\n\n<description>\n" + input_text + "\n</description>",
        "completion": example["output"]
    }

new_dataset = dataset.map(process_example, batched=False, num_proc=40, remove_columns=dataset.column_names)

print(new_dataset[0])
new_dataset.push_to_hub("wanhin/DEEPCAD-completion-sft")


