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
Tôi có một file JSON CAD (được sử dụng để tạo CAD) và mô tả của JSON:

{prompt}

{completion}

*Bước 1: Loại bỏ phần mô tả thừa ngay sau phần mô tả đùn.

**Ví dụ:
<description>
Phần 1: Lăng trụ hình chữ nhật ba chiều với đỉnh và đáy phẳng. Tạo một hệ tọa độ mới với các thuộc tính sau: * Góc Euler: (0.0, 0.0, -90.0) * Vector Dịch chuyển: (0.0, 0.0316, 0.0). Vẽ một bản phác thảo 2D trên mặt phẳng XY của hệ tọa độ. Tạo một mặt chứa một vòng lặp kín được tạo thành từ 4 đường: * Đường 1: Điểm Bắt đầu (0.0, 0.0), Điểm Kết thúc (0.75, 0.0) * Đường 2: Điểm Bắt đầu (0.75, 0.0), Điểm Kết thúc (0.75, 0.6772) * Đường 3: Điểm Bắt đầu (0.75, 0.6772), Điểm Kết thúc (0.0, 0.6772) * Đường 4: Điểm Bắt đầu (0.0, 0.6772), Điểm Kết thúc (0.0, 0.0). Thu nhỏ bản phác thảo 2D theo hệ số 0.75. Chuyển đổi bản phác thảo 2D đã thu nhỏ thành bản phác thảo 3D sử dụng hệ tọa độ đã định nghĩa. Đùn bản phác thảo 3D theo 0.0316 đơn vị theo hướng Z dương. Chiều cao của phần này là 0.75 đơn vị, chiều rộng là 0.0316 đơn vị, và chiều dài là 0.75 đơn vị. Điều này hoàn thành phần lăng trụ hình chữ nhật ba chiều.
</description>

-> Bạn cần loại bỏ dòng: "Chiều cao của phần này là 0.75 đơn vị, chiều rộng là 0.0316 đơn vị, và chiều dài là 0.75 đơn vị. Điều này hoàn thành phần lăng trụ hình chữ nhật ba chiều." Giữ lại thẻ </description>.
Kết quả bạn cần trả về sau khi loại bỏ phần mô tả thừa sẽ như sau:

<description>
Phần 1: Lăng trụ hình chữ nhật ba chiều với đỉnh và đáy phẳng. Tạo một hệ tọa độ mới với các thuộc tính sau: * Góc Euler: (0.0, 0.0, -90.0) * Vector Dịch chuyển: (0.0, 0.0316, 0.0). Vẽ một bản phác thảo 2D trên mặt phẳng XY của hệ tọa độ. Tạo một mặt chứa một vòng lặp kín được tạo thành từ 4 đường: * Đường 1: Điểm Bắt đầu (0.0, 0.0), Điểm Kết thúc (0.75, 0.0) * Đường 2: Điểm Bắt đầu (0.75, 0.0), Điểm Kết thúc (0.75, 0.6772) * Đường 3: Điểm Bắt đầu (0.75, 0.6772), Điểm Kết thúc (0.0, 0.6772) * Đường 4: Điểm Bắt đầu (0.0, 0.6772), Điểm Kết thúc (0.0, 0.0). Thu nhỏ bản phác thảo 2D theo hệ số 0.75. Chuyển đổi bản phác thảo 2D đã thu nhỏ thành bản phác thảo 3D sử dụng hệ tọa độ đã định nghĩa. Đùn bản phác thảo 3D theo 0.0316 đơn vị theo hướng Z dương.
</description>
**

*Bước 2: Kiểm tra xem mô tả mới được tạo có khớp với json không. Nếu có, tạo "<valid>Yes</valid>", nếu không, tạo "<valid>No</valid>"

*Bước 3: Tạo dữ liệu lý luận mẫu được bao trong <think> ... </think>. Dữ liệu lý luận nên tuân theo hai bước:
Bước 1: Lý luận ra các thành phần sẽ có trong JSON dựa trên mô tả đã cho.
Bước 2: Kiểm tra logic, tính đúng đắn về số học, và thực hiện các sửa đổi (nếu cần thiết) từ Bước 1.

**Ví dụ 1 mẫu:

***Đầu vào:
<json> {{"parts": {{"part_1": {{"coordinate_system": {{"Euler Angles": [0.0, 0.0, -90.0], "Translation Vector": [0.0, 0.0316, 0.0]}}, "sketch": {{"face_1": {{"loop_1": {{"line_1": {{"Start Point": [0.0, 0.0], "End Point": [0.75, 0.0]}}, "line_2": {{"Start Point": [0.75, 0.0], "End Point": [0.75, 0.6772]}}, "line_3": {{"Start Point": [0.75, 0.6772], "End Point": [0.0, 0.6772]}}, "line_4": {{"Start Point": [0.0, 0.6772], "End Point": [0.0, 0.0]}}}}}}, "extrusion": {{"extrude_depth_towards_normal": 0.0316, "extrude_depth_opposite_normal": 0.0, "sketch_scale": 0.75, "operation": "NewBodyFeatureOperation"}}}}}}}} </json>
<description> Phần 1: Lăng trụ hình chữ nhật ba chiều với đỉnh và đáy phẳng. Tạo một hệ tọa độ mới với các thuộc tính sau: * Góc Euler: (0.0, 0.0, -90.0) * Vector Dịch chuyển: (0.0, 0.0316, 0.0). Vẽ một bản phác thảo 2D trên mặt phẳng XY của hệ tọa độ. Tạo một mặt chứa một vòng lặp kín được tạo thành từ 4 đường: * Đường 1: Điểm Bắt đầu (0.0, 0.0), Điểm Kết thúc (0.75, 0.0) * Đường 2: Điểm Bắt đầu (0.75, 0.0), Điểm Kết thúc (0.75, 0.6772) * Đường 3: Điểm Bắt đầu (0.75, 0.6772), Điểm Kết thúc (0.0, 0.6772) * Đường 4: Điểm Bắt đầu (0.0, 0.6772), Điểm Kết thúc (0.0, 0.0). Thu nhỏ bản phác thảo 2D theo hệ số 0.75. Chuyển đổi bản phác thảo 2D đã thu nhỏ thành bản phác thảo 3D sử dụng hệ tọa độ đã định nghĩa. Đùn bản phác thảo 3D theo 0.0316 đơn vị theo hướng Z dương. </description>

***Đầu ra:
Bước 1:
<description>
Phần 1: Lăng trụ hình chữ nhật ba chiều với đỉnh và đáy phẳng. Tạo một hệ tọa độ mới với các thuộc tính sau: * Góc Euler: (0.0, 0.0, -90.0) * Vector Dịch chuyển: (0.0, 0.0316, 0.0). Vẽ một bản phác thảo 2D trên mặt phẳng XY của hệ tọa độ. Tạo một mặt chứa một vòng lặp kín được tạo thành từ 4 đường: * Đường 1: Điểm Bắt đầu (0.0, 0.0), Điểm Kết thúc (0.75, 0.0) * Đường 2: Điểm Bắt đầu (0.75, 0.0), Điểm Kết thúc (0.75, 0.6772) * Đường 3: Điểm Bắt đầu (0.75, 0.6772), Điểm Kết thúc (0.0, 0.6772) * Đường 4: Điểm Bắt đầu (0.0, 0.6772), Điểm Kết thúc (0.0, 0.0). Thu nhỏ bản phác thảo 2D theo hệ số 0.75. Chuyển đổi bản phác thảo 2D đã thu nhỏ thành bản phác thảo 3D sử dụng hệ tọa độ đã định nghĩa. Đùn bản phác thảo 3D theo 0.0316 đơn vị theo hướng Z dương.
</description>
Bước 2:
<valid>Yes</valid>
Bước 3:
<think>
***Bước 1: Suy luận các thành phần sẽ có trong json dựa trên mô tả được cung cấp:

parts: Từ mô tả, chúng ta có một phần duy nhất được gắn nhãn "part_1" mô tả một lăng trụ hình chữ nhật ba chiều. Điều này sẽ được phản ánh trong json như "part_1".

part_1: (Lăng trụ hình chữ nhật)
-coordinate_system:
--Euler Angles: [0.0, 0.0, -90.0] (Suy ra từ mô tả: "Tạo một hệ tọa độ mới với các thuộc tính sau: * Góc Euler: (0.0, 0.0, -90.0)").
--Translation Vector: [0.0, 0.0316, 0.0] (Suy ra từ mô tả: "Vector Dịch chuyển: (0.0, 0.0316, 0.0)").
-sketch:
--face_1:
---loop_1:
----line_1: Start [0.0, 0.0], End [0.75, 0.0] (Suy ra từ mô tả: "Đường 1: Điểm Bắt đầu (0.0, 0.0), Điểm Kết thúc (0.75, 0.0)").
----line_2: Start [0.75, 0.0], End [0.75, 0.6772] (Suy ra từ mô tả: "Đường 2: Điểm Bắt đầu (0.75, 0.0), Điểm Kết thúc (0.75, 0.6772)").
----line_3: Start [0.75, 0.6772], End [0.0, 0.6772] (Suy ra từ mô tả: "Đường 3: Điểm Bắt đầu (0.75, 0.6772), Điểm Kết thúc (0.0, 0.6772)").
----line_4: Start [0.0, 0.6772], End [0.0, 0.0] (Suy ra từ mô tả: "Đường 4: Điểm Bắt đầu (0.0, 0.6772), Điểm Kết thúc (0.0, 0.0)").
-extrusion:
--extrude_depth_towards_normal: 0.0316 (Suy ra từ mô tả: "Đùn bản phác thảo 3D theo 0.0316 đơn vị theo hướng Z dương").
--extrude_depth_opposite_normal: 0.0 (Không được đề cập trong mô tả, nhưng được suy ra là 0 vì không cần đùn theo hướng ngược lại).
--sketch_scale: 0.75 (Suy ra từ mô tả: "Thu nhỏ bản phác thảo 2D theo hệ số 0.75").
--operation: "NewBodyFeatureOperation" (Suy ra từ mô tả: "Tạo mô hình 3D của lăng trụ hình chữ nhật").

***Bước 2: Kiểm tra logic, tính đúng đắn về số học, và thực hiện các sửa đổi (nếu cần thiết) từ Bước 1:

-Kiểm tra các đường (line_1, line_2, line_3, line_4):
line_1 (Start [0.0, 0.0], End [0.75, 0.0]) kết nối chính xác với line_2 (Start [0.75, 0.0], End [0.75, 0.6772]).
line_2 (Start [0.75, 0.0], End [0.75, 0.6772]) kết nối chính xác với line_3 (Start [0.75, 0.6772], End [0.0, 0.6772]).
line_3 (Start [0.75, 0.6772], End [0.0, 0.6772]) kết nối chính xác với line_4 (Start [0.0, 0.6772], End [0.0, 0.0]).
line_4 (Start [0.0, 0.6772], End [0.0, 0.0]) kết nối chính xác với line_1 (Start [0.0, 0.0], End [0.75, 0.0]).

--Kết quả: Vòng lặp phác thảo hợp lệ và kín.

-Kiểm tra đùn:
Đùn bản phác thảo 3D với độ sâu 0.0316 đơn vị theo hướng Z dương, như được chỉ định trong mô tả.
Không cần đùn theo hướng ngược lại, vì vậy extrude_depth_opposite_normal được đặt chính xác là 0.

--Kết quả: Đùn chính xác và khớp với yêu cầu.

-Kiểm tra tính nhất quán về kích thước:
Các giá trị cho Điểm Bắt đầu/Điểm Kết thúc của các đường được biểu diễn chính xác và khớp với bản phác thảo 2D được mô tả.
Độ sâu đùn và hệ số thu nhỏ nhất quán với mô tả.

--Kết quả: Các kích thước chính xác và nhất quán.

-Sửa đổi: Dữ liệu suy luận này chính xác và không cần sửa đổi thêm.
</think>
**
"""

def extract_tags(text, tag_name):
    """Extract content from XML-like tags"""
    pattern = f"<{tag_name}>(.*?)</{tag_name}>"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[0] if matches else None

def process_dataset(start=0, end=500, old_split="range_0_500_vi"):
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
    print(f"Uploading dataset to 'hf_của_tụi_m/cad_reason_stage2' with split '{new_split_name}'...")
    from datasets import Dataset
    
    new_dataset = Dataset.from_list(new_data)
    
    new_dataset.push_to_hub("hf_của_tụi_m/cad_reason_stage2", split=new_split_name)
    print("Dataset uploaded successfully!")

if __name__ == "__main__":
    # Example usage
    process_dataset(start=4500, end=6000, old_split="range_1000_1500_vi") 
