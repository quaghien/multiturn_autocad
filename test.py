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
    # load_dotenv()
    # hf_token = os.getenv('HF_TOKEN')
    # if not hf_token:
    #     raise ValueError("Please set HF_TOKEN environment variable")
    # login(token=hf_token)

    # Load the trained model
    # model_path = "train_results/Qwen2.5-7B-Instruct_3epoch_6000maxlength/checkpoint-105813"  # Update this path to match your saved model
    model_path = "wanhin/Qwen2.5-7B-Instruct_1epoch_6000maxlength"
    my_model, my_tokenizer = load_trained_model(model_path)
    # my_model.push_to_hub(f"wanhin/Qwen2.5-7B-Instruct_1epoch_6000maxlength")
    # my_tokenizer.push_to_hub(f"wanhin/Qwen2.5-7B-Instruct_1epoch_6000maxlength")
    # print("Pushed model and tokenizer to hub")

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
    **Đối tượng phẳng, tròn có lỗ trung tâm** Bắt đầu bằng cách tạo hệ tọa độ mới cho phần đầu tiên với các thuộc tính sau: * Góc Euler: [0,0, 0,0, 0,0] * Vector dịch: [0,0, 0,0, 0,0375] Đối với bản phác thảo của phần này, tạo một mặt mới (mặt\\_1) và xác định các vòng lặp và đường cong như sau: **Mặt 1** *Vòng lặp 1* * Vòng tròn 1: + Tâm: [0,375, 0,375] + Bán kính: 0,375 *Vòng 2* * Dòng 1: + Điểm bắt đầu: [0,1429, 0,5537] + Điểm kết thúc: [0,1692, 0,5537] * Dòng 2: + Điểm bắt đầu: [0,1692, 0,5537] + Điểm kết thúc: [0,1692, 0,5884] * Dòng 3: + Điểm bắt đầu: [0,1692, 0,5884] + Điểm kết thúc: [0,1429, 0,5884] * Dòng 4: + Điểm bắt đầu: [0,1429, 0,5884] + Điểm kết thúc: [0,1429, 0,5537] *Vòng 3* * Vòng 1: + Giữa: [0,375, 0,375] + Bán kính: 0,075 *Vòng 4* * Dòng 1: + Điểm bắt đầu: [0,542, 0,2212] + Điểm kết thúc: [0,5669, 0,2212] * Dòng 2: + Điểm bắt đầu: [0,5669, 0,2212] + Điểm kết thúc: [0,5669, 0,2545] * Dòng 3: + Điểm bắt đầu: [0,5669, 0,2545] + Điểm kết thúc: [0,542, 0,2545] * Dòng 4: + Điểm bắt đầu: [0,542, 0,2545] + Điểm kết thúc: [0,542, 0,2212] *Vòng 5* * Dòng 1: + Điểm bắt đầu: [0,5517, 0,5842] + Điểm cuối: [0,5669, 0,5842] * Dòng 2: + Điểm bắt đầu: [0,5669, 0,5842] + Điểm cuối: [0,5669, 0,605] * Dòng 3: + Điểm bắt đầu: [0,5669, 0,605] + Điểm cuối: [0,5517, 0,605] * Đường 4: + Điểm bắt đầu: [0.5517, 0.605] + Điểm kết thúc: [0.5517, 0.5842] Bản phác thảo đã hoàn tất. Chia tỷ lệ bản phác thảo bằng cách sử dụng tham số chia tỷ lệ sketch\\_scale (0,75), chuyển đổi nó thành 3D, sau đó đùn bản phác thảo để tạo mô hình 3D. Áp dụng NewBodyFeatureOperation để tạo hình dạng cuối cùng. Hình dạng cuối cùng là một vật thể phẳng, hình tròn có lỗ ở giữa, bề mặt nhẵn, cong và đối xứng qua trục trung tâm của nó. Lỗ nằm ở tâm của vật thể và có đường kính bằng chính vật thể đó. Vật thể có vẻ là một khối ba chiều, không có kết cấu hoặc hoa văn rõ ràng. Vật thể có kích thước như sau: * Chiều dài: 0,75 m * Chiều rộng: 0,75 m * Chiều cao: 0,0375 m
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
    # formatted_prompt = apply_chat_template(example, my_tokenizer)

    # print(formatted_prompt)

    #Prepare model inputs
    model_inputs = my_tokenizer(text, return_tensors="pt").to(my_model.device)

    # Generate response
    print("Generating response...")
    generated_ids = my_model.generate(
        **model_inputs,
        max_new_tokens=6000,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=20,
        pad_token_id=my_tokenizer.pad_token_id,
        eos_token_id=my_tokenizer.eos_token_id
    )

    # Process generated output
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = my_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print()
    print(response)