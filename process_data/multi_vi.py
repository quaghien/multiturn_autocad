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

# Prompt template cho single sample - rõ ràng và đầy đủ
single_prompt_template = '''
QUY LUẬT XỬ LÝ CHO MỖI JSON INPUT:

1. Phân tích JSON input để đếm số parts (part_1, part_2, ..., part_n)
2. Tạo multi-turn data tùy chỉnh theo số part: (json_1, prompt_1, think_1 -> ... -> json_n, prompt_n, think_n)

CHI TIẾT QUY LUẬT:
- Tạo json progressive: json_1 (chỉ part_1), json_2 (part_1+part_2), ..., json_n (part_1+part_2+...+part_n)
- Mỗi json đặt trong tag <json_i></json_i>
- Tạo mô tả user input cho từng json trong tag <prompt_i></prompt_i> (mô tả hình dạng, không có số liệu kỹ thuật)
- Tạo suy luận 2 bước trong tag <think_i></think_i>:
  * Bước 1: Suy luận các thành phần sẽ có trong JSON dựa trên mô tả được cung cấp
  * Bước 2: Kiểm tra logic, tính đúng đắn về số học, và thực hiện các sửa đổi (nếu cần thiết) từ Bước 1

FORMAT OUTPUT CHO MỖI SAMPLE:
<sample_n>
<json_1>[json với part_1]</json_1>
<prompt_1>[mô tả user cho json_1]</prompt_1>
<think_1>[suy luận 2 bước cho json_1]</think_1>
<json_2>[json với part_1+part_2]</json_2>
<prompt_2>[mô tả user cho json_2]</prompt_2>
<think_2>[suy luận 2 bước cho json_2]</think_2>
...tiếp tục đến json_n...
</sample_n>

VÍ DỤ CỤ THỂ:
INPUT:
<input_1>
{"parts":{"part_1":{"coordinate_system":{"Euler Angles":[0.0,0.0,0.0],"Translation Vector":[0.0,0.0,0.1607]},"sketch":{"face_1":{"loop_1":{"line_1":{"Start Point":[0.0,0.0],"End Point":[0.3214,0.0]},"line_2":{"Start Point":[0.3214,0.0],"End Point":[0.3214,0.3214]},"line_3":{"Start Point":[0.3214,0.3214],"End Point":[0.0,0.3214]},"line_4":{"Start Point":[0.0,0.3214],"End Point":[0.0,0.0]}}}},"extrusion":{"extrude_depth_towards_normal":0.0804,"extrude_depth_opposite_normal":0.0804,"sketch_scale":0.3214,"operation":"NewBodyFeatureOperation"}},"part_2":{"coordinate_system":{"Euler Angles":[0.0,0.0,0.0],"Translation Vector":[0.3214,0.0,0.1607]},"sketch":{"face_1":{"loop_1":{"line_1":{"Start Point":[0.0,0.0],"End Point":[0.2679,0.0]},"line_2":{"Start Point":[0.2679,0.0],"End Point":[0.2679,0.3214]},"line_3":{"Start Point":[0.2679,0.3214],"End Point":[0.0,0.3214]},"line_4":{"Start Point":[0.0,0.3214],"End Point":[0.0,0.0]}}}},"extrusion":{"extrude_depth_towards_normal":0.0268,"extrude_depth_opposite_normal":0.0268,"sketch_scale":0.3214,"operation":"JoinFeatureOperation"}},"part_3":{"coordinate_system":{"Euler Angles":[0.0,0.0,0.0],"Translation Vector":[0.5893,0.0,0.1607]},"sketch":{"face_1":{"loop_1":{"arc_1":{"Start Point":[0.0,0.0],"Mid Point":[0.1607,0.1607],"End Point":[0.0,0.3214]},"line_1":{"Start Point":[0.0,0.3214],"End Point":[0.0,0.0]}}}},"extrusion":{"extrude_depth_towards_normal":0.0268,"extrude_depth_opposite_normal":0.0268,"sketch_scale":0.3214,"operation":"JoinFeatureOperation"}}}}
</input_1>

OUTPUT:
<sample_1>
<json_1>
{{"parts":{{"part_1":{{"coordinate_system":{{"Euler Angles":[0.0,0.0,0.0],"Translation Vector":[0.0,0.0,0.0]}},"sketch":{{"face_1":{{"loop_1":{{"line_1":{{"Start Point":[0.0,0.0],"End Point":[0.75,0.0]}},"line_2":{{"Start Point":[0.75,0.0],"End Point":[0.75,0.2812]}},"line_3":{{"Start Point":[0.75,0.2812],"End Point":[0.0,0.2812]}},"line_4":{{"Start Point":[0.0,0.2812],"End Point":[0.0,0.0]}}}},"loop_2":{{"circle_1":{{"Center":[0.1716,0.1406],"Radius":0.0769}}}},"loop_3":{{"circle_1":{{"Center":[0.5784,0.1406],"Radius":0.0769}}}}}},"face_2":{{"loop_1":{{"circle_1":{{"Center":[0.1716,0.1406],"Radius":0.0769}}}},"loop_2":{{"circle_1":{{"Center":[0.1716,0.1406],"Radius":0.0675}}}}}},"face_3":{{"loop_1":{{"circle_1":{{"Center":[0.5784,0.1406],"Radius":0.0769}}}},"loop_2":{{"circle_1":{{"Center":[0.5784,0.1406],"Radius":0.0675}}}}}}}},"extrusion":{{"extrude_depth_towards_normal":0.0562,"extrude_depth_opposite_normal":0.0,"sketch_scale":0.75,"operation":"NewBodyFeatureOperation"}}}}}}
</json_1>
<prompt_1>
Tôi muốn tạo một tấm phẳng hình chữ nhật. Trên tấm đó có hai lỗ tròn và ở hai mặt bên của tấm, tại vị trí hai lỗ đó, tôi muốn có hai rãnh tròn đồng tâm.
</prompt_1>
<think_1>
Bước 1: Suy luận các thành phần sẽ có trong JSON dựa trên mô tả được cung cấp
Từ yêu cầu "Tôi muốn tạo một tấm phẳng hình chữ nhật. Trên tấm đó có hai lỗ tròn và ở hai mặt bên của tấm, tại vị trí hai lỗ đó, tôi muốn có hai rãnh tròn đồng tâm", tôi sẽ phân tích và giả định các yếu tố cần thiết, liệt kê các khóa JSON cụ thể.

**parts**: Khóa cấp cao nhất để chứa các phần của mô hình.
**part_1**: Định nghĩa phần đầu tiên của mô hình.
**coordinate_system**: Thành phần để định vị và hướng phần này trong không gian.
  **Euler Angles**: Tôi sẽ giả định không có sự xoay ban đầu. Tôi đặt là `[0.0,0.0,0.0]`.
  **Translation Vector**: Tôi sẽ giả định một vị trí mặc định tại gốc tọa độ. Tôi đặt là `[0.0,0.0,0.0]`.
**sketch**: Thành phần định nghĩa bản phác thảo 2D cơ sở.
  **face_1**: Đại diện cho mặt phẳng chứa bản phác thảo chính của tấm và các lỗ.
    **loop_1**: Đại diện cho hình chữ nhật bên ngoài của tấm.
      **line_1, line_2, line_3, line_4**: Tôi sẽ đặt các điểm `Start Point` và `End Point` để tạo hình chữ nhật. Ví dụ: `line_1:{"Start Point":[0.0,0.0],"End Point":[0.75,0.0]}`, `line_2:{"Start Point":[0.75,0.0],"End Point":[0.75,0.2812]}`, `line_3:{"Start Point":[0.75,0.2812],"End Point":[0.0,0.2812]}`, `line_4:{"Start Point":[0.0,0.2812],"End Point":[0.0,0.0]}`.
    **loop_2, loop_3**: Đại diện cho hai lỗ tròn trên tấm.
      **circle_1**: Là hình dạng lỗ tròn.
        **Center**: Tôi sẽ đặt các vị trí tâm cho hai lỗ, ví dụ: `loop_2:{"circle_1":{"Center":[0.1716,0.1406]}}` và `loop_3:{"circle_1":{"Center":[0.5784,0.1406]}}`.
        **Radius**: Tôi sẽ đặt bán kính cho hai lỗ, ví dụ: `0.0769`.
  **face_2, face_3**: Đại diện cho hai rãnh tròn đồng tâm ở mặt bên. Mỗi rãnh sẽ là một `face` riêng.
    **loop_1**: Đại diện cho vòng ngoài của rãnh.
      **circle_1**: Là hình dạng vòng ngoài.
        **Center**: Tôi sẽ đặt tâm của vòng ngoài trùng với tâm lỗ tương ứng, ví dụ: `face_2:{"loop_1":{"circle_1":{"Center":[0.1716,0.1406]}}}` và `face_3:{"loop_1":{"circle_1":{"Center":[0.5784,0.1406]}}}`.
        **Radius**: Tôi sẽ đặt bán kính của vòng ngoài, ví dụ: `0.0769`.
    **loop_2**: Đại diện cho vòng trong của rãnh.
      **circle_1**: Là hình dạng vòng trong.
        **Center**: Tôi sẽ đặt tâm của vòng trong trùng với tâm vòng ngoài, ví dụ: `face_2:{"loop_2":{"circle_1":{"Center":[0.1716,0.1406]}}}` và `face_3:{"loop_2":{"circle_1":{"Center":[0.5784,0.1406]}}}`.
        **Radius**: Tôi sẽ đặt bán kính của vòng trong, nhỏ hơn vòng ngoài một chút, ví dụ: `0.0675`.
**extrusion**: Thành phần định nghĩa quá trình đùn để tạo hình 3D.
  **extrude_depth_towards_normal**: Tôi sẽ đặt một chiều dày hợp lý cho tấm, ví dụ `0.0562`.
  **extrude_depth_opposite_normal**: Tôi suy luận rằng quá trình đùn chỉ diễn ra một chiều. Tôi đặt là `0.0`.
  **sketch_scale**: Tôi sẽ đặt một tỷ lệ để tấm không bị thon, ví dụ `0.75`.
  **operation**: Tôi suy luận rằng đây là thao tác để tạo một khối 3D mới hoàn chỉnh. Tôi đặt là `"NewBodyFeatureOperation"`.

Bước 2: Kiểm tra logic, tính đúng đắn về số học, và thực hiện các sửa đổi (nếu cần thiết) từ Bước 1
Tôi kiểm tra tính hợp lý của các thông số đã suy luận từ Bước 1.

**coordinate_system**:
  **Euler Angles:[0.0,0.0,0.0]**: Suy luận: Không xoay. Kết luận: Hợp lý.
  **Translation Vector:[0.0,0.0,0.0]**: Suy luận: Tại gốc tọa độ. Kết luận: Hợp lý.
**sketch**:
  **face_1**:
    **loop_1 (hình chữ nhật)**: Các điểm tọa độ tạo hình chữ nhật hợp lý. Kết luận: Hợp lý.
    **loop_2, loop_3 (lỗ tròn)**: Tâm và bán kính của các lỗ được đặt đối xứng và hợp lý trên tấm. Kết luận: Hợp lý.
  **face_2, face_3 (rãnh đồng tâm)**:
    **loop_1, loop_2**: Các vòng tròn đồng tâm với bán kính khác nhau tạo rãnh. Vị trí tâm trùng với các lỗ trên `face_1` là hợp lý cho mô tả "ở hai mặt bên của tấm, tại vị trí hai lỗ đó". Kết luận: Hợp lý.
**extrusion**:
  **extrude_depth_towards_normal:0.0562**: Suy luận: Xác định chiều dày của tấm. Kết luận: Hợp lý.
  **extrude_depth_opposite_normal:0.0**: Suy luận: Chỉ đùn một chiều. Kết luận: Hợp lý.
  **sketch_scale:0.75**: Suy luận: Tỷ lệ phác thảo để tấm không bị thon. Kết luận: Hợp lý.
  **operation:"NewBodyFeatureOperation"**: Suy luận: Tạo khối mới. Kết luận: Hợp lý.
Thực hiện các sửa đổi (nếu cần thiết): Không có thông số nào cần sửa đổi.
</think_1>

<json_2>
{{"parts":{{"part_1":{{"coordinate_system":{{"Euler Angles":[0.0,0.0,0.0],"Translation Vector":[0.0,0.0,0.0]}},"sketch":{{"face_1":{{"loop_1":{{"line_1":{{"Start Point":[0.0,0.0],"End Point":[0.75,0.0]}},"line_2":{{"Start Point":[0.75,0.0],"End Point":[0.75,0.2812]}},"line_3":{{"Start Point":[0.75,0.2812],"End Point":[0.0,0.2812]}},"line_4":{{"Start Point":[0.0,0.2812],"End Point":[0.0,0.0]}}}},"loop_2":{{"circle_1":{{"Center":[0.1716,0.1406],"Radius":0.0769}}}},"loop_3":{{"circle_1":{{"Center":[0.5784,0.1406],"Radius":0.0769}}}}}},"face_2":{{"loop_1":{{"circle_1":{{"Center":[0.1716,0.1406],"Radius":0.0769}}}},"loop_2":{{"circle_1":{{"Center":[0.1716,0.1406],"Radius":0.0675}}}}}},"face_3":{{"loop_1":{{"circle_1":{{"Center":[0.5784,0.1406],"Radius":0.0769}}}},"loop_2":{{"circle_1":{{"Center":[0.5784,0.1406],"Radius":0.0675}}}}}}}},"extrusion":{{"extrude_depth_towards_normal":0.0562,"extrude_depth_opposite_normal":0.0,"sketch_scale":0.75,"operation":"NewBodyFeatureOperation"}}}},"part_2":{{"coordinate_system":{{"Euler Angles":[0.0,0.0,0.0],"Translation Vector":[0.0947,0.0637,0.0562]}},"sketch":{{"face_1":{{"loop_1":{{"circle_1":{{"Center":[0.0769,0.0769],"Radius":0.0769}}}},"loop_2":{{"circle_1":{{"Center":[0.0769,0.0769],"Radius":0.0675}}}}}},"face_2":{{"loop_1":{{"circle_1":{{"Center":[0.0769,0.0769],"Radius":0.0675}}}}}},"face_3":{{"loop_1":{{"circle_1":{{"Center":[0.4837,0.0769],"Radius":0.0769}}}},"loop_2":{{"circle_1":{{"Center":[0.4837,0.0769],"Radius":0.0675}}}}}},"face_4":{{"loop_1":{{"circle_1":{{"Center":[0.4837,0.0769],"Radius":0.0675}}}}}}}},"extrusion":{{"extrude_depth_towards_normal":0.0,"extrude_depth_opposite_normal":0.0187,"sketch_scale":0.5606,"operation":"CutFeatureOperation"}}}}}}}}
</json_2>
<prompt_2>
Tôi muốn thêm hai rãnh tròn nữa, ở hai phía còn lại của tấm hình chữ nhật, nằm gần các lỗ tròn ban đầu. Hai rãnh này cũng là dạng vành khuyên và nông.
</prompt_2>
<think_2>
Bước 1: Suy luận các thành phần sẽ có trong JSON dựa trên mô tả được cung cấp
Từ yêu cầu "Tôi muốn thêm hai rãnh tròn nữa, ở hai phía còn lại của tấm hình chữ nhật, nằm gần các lỗ tròn ban đầu. Hai rãnh này cũng là dạng vành khuyên và nông", tôi sẽ phân tích và giả định các yếu tố cần thiết, liệt kê các khóa JSON cụ thể cho `part_2`.

**part_2**: Định nghĩa hai rãnh tròn mới.
**coordinate_system**: Thành phần để định vị và hướng phần này trong không gian.
  **Euler Angles**: Tôi sẽ giả định không có sự xoay ban đầu. Tôi đặt là `[0.0,0.0,0.0]`.
  **Translation Vector**: Tôi sẽ đặt vị trí của các rãnh. Z-coordinate sẽ bằng chiều dày của tấm (`0.0562`) để chúng nằm trên mặt trên của tấm. X và Y sẽ được dịch một chút để các rãnh nằm gần các lỗ tròn ban đầu nhưng ở phía khác, ví dụ `[0.0947,0.0637,0.0562]`.
**sketch**: Thành phần định nghĩa bản phác thảo 2D hình vành khuyên (hai hình tròn đồng tâm) cho mỗi rãnh.
  **face_1**: Đại diện cho rãnh thứ nhất.
    **loop_1**: Đại diện cho vòng ngoài của rãnh.
      **circle_1**: Là hình dạng vòng ngoài.
        **Center**: Tôi sẽ đặt tâm của vòng ngoài, ví dụ `[0.0769,0.0769]`.
        **Radius**: Tôi sẽ đặt bán kính của vòng ngoài, ví dụ `0.0769`.
    **loop_2**: Đại diện cho vòng trong của rãnh.
      **circle_1**: Là hình dạng vòng trong.
        **Center**: Tôi sẽ đặt tâm của vòng trong trùng với tâm vòng ngoài, ví dụ `[0.0769,0.0769]`.
        **Radius**: Tôi sẽ đặt bán kính của vòng trong, ví dụ `0.0675`.
  **face_2**: Đại diện cho phần cắt của rãnh thứ nhất (nếu có vòng trong). Tôi suy luận rằng đây là một vòng tròn đồng tâm với vòng ngoài để tạo rãnh vành khuyên.
    **loop_1**:
      **circle_1**:
        **Center**: Tôi sẽ đặt tâm của vòng trong trùng với tâm vòng ngoài, ví dụ `[0.0769,0.0769]`.
        **Radius**: Tôi sẽ đặt bán kính của vòng trong, ví dụ `0.0675`.
  **face_3**: Đại diện cho rãnh thứ hai.
    **loop_1**: Đại diện cho vòng ngoài của rãnh.
      **circle_1**: Là hình dạng vòng ngoài.
        **Center**: Tôi sẽ đặt tâm của vòng ngoài, ví dụ `[0.4837,0.0769]`.
        **Radius**: Tôi sẽ đặt bán kính của vòng ngoài, ví dụ `0.0769`.
    **loop_2**: Đại diện cho vòng trong của rãnh.
      **circle_1**: Là hình dạng vòng trong.
        **Center**: Tôi sẽ đặt tâm của vòng trong trùng với tâm vòng ngoài, ví dụ `[0.4837,0.0769]`.
        **Radius**: Tôi sẽ đặt bán kính của vòng trong, ví dụ `0.0675`.
  **face_4**: Đại diện cho phần cắt của rãnh thứ hai (nếu có vòng trong). Tôi suy luận rằng đây là một vòng tròn đồng tâm với vòng ngoài để tạo rãnh vành khuyên.
    **loop_1**:
      **circle_1**:
        **Center**: Tôi sẽ đặt tâm của vòng trong trùng với tâm vòng ngoài, ví dụ `[0.4837,0.0769]`.
        **Radius**: Tôi sẽ đặt bán kính của vòng trong, ví dụ `0.0675`.
**extrusion**: Thành phần định nghĩa quá trình đùn để tạo rãnh (cắt).
  **extrude_depth_towards_normal**: Tôi suy luận rằng quá trình khoét rãnh sẽ đi ngược chiều pháp tuyến của bề mặt. Tôi đặt là `0.0`.
  **extrude_depth_opposite_normal**: Tôi sẽ đặt chiều sâu của rãnh rất nông, ví dụ `0.0187`.
  **sketch_scale**: Tôi sẽ đặt một tỷ lệ để rãnh không bị biến dạng, ví dụ `0.5606`.
  **operation**: Tôi suy luận rằng đây là thao tác để cắt vật thể hiện có. Tôi đặt là `"CutFeatureOperation"`.

Bước 2: Kiểm tra logic, tính đúng đắn về số học, và thực hiện các sửa đổi (nếu cần thiết) từ Bước 1
Tôi kiểm tra tính hợp lý của các thông số đã suy luận từ Bước 1.

**coordinate_system**:
  **Euler Angles:[0.0,0.0,0.0]**: Suy luận: Không xoay. Kết luận: Hợp lý.
  **Translation Vector:[0.0947,0.0637,0.0562]**: Suy luận: Vị trí trên mặt trên của tấm và hơi dịch chuyển. Kết luận: Hợp lý.
**sketch**:
  **face_1, face_3 (vòng ngoài)**:
    **circle_1**: Tâm và bán kính tạo vòng ngoài. Kết luận: Hợp lý.
  **face_1, face_3 (vòng trong)**:
    **circle_1**: Tâm và bán kính tạo vòng trong đồng tâm và nhỏ hơn. Kết luận: Hợp lý.
  **face_2, face_4**: Các rãnh này có vẻ được định nghĩa bằng một vòng tròn duy nhất (chỉ có `loop_1` và `circle_1`), nhưng mô tả là "vành khuyên". Có thể `loop_2` của `face_1` và `face_3` đã tạo ra vòng trong, và `face_2` và `face_4` chỉ là các bản sao của vòng trong đó hoặc là các mặt phẳng phác thảo cho việc cắt. Dựa vào JSON, `face_2` và `face_4` chỉ có một `loop` với một `circle` có bán kính `0.0675`, đây chính là bán kính của vòng trong từ `face_1` và `face_3`. Điều này cho thấy `face_2` và `face_4` được dùng để định nghĩa lỗ cắt cho rãnh vành khuyên. Kết luận: Hợp lý để tạo rãnh vành khuyên.
**extrusion**:
  **extrude_depth_towards_normal:0.0**: Suy luận: Không đùn về phía pháp tuyến. Kết luận: Hợp lý.
  **extrude_depth_opposite_normal:0.0187**: Suy luận: Chiều sâu rãnh rất nông. Kết luận: Hợp lý.
  **sketch_scale:0.5606**: Suy luận: Tỷ lệ phác thảo để rãnh không bị biến dạng. Kết luận: Hợp lý.
  **operation:"CutFeatureOperation"**: Suy luận: Thực hiện thao tác cắt. Kết luận: Hợp lý.
Thực hiện các sửa đổi (nếu cần thiết): Cấu trúc của `face_2` và `face_4` trong JSON có vẻ hơi thừa nếu mục đích là tạo rãnh vành khuyên. Thông thường, rãnh vành khuyên được định nghĩa bởi hai vòng tròn đồng tâm trong cùng một `loop` hoặc `face`. Tuy nhiên, với cấu trúc hiện tại, nó vẫn có thể hoạt động nếu phần mềm CAD hiểu rằng việc cắt dựa trên vòng tròn nhỏ hơn sẽ tạo ra một phần rỗng bên trong vòng tròn lớn hơn. Với mục đích suy luận từ mô tả, tôi sẽ giả định đây là cách người dùng muốn tạo rãnh vành khuyên.
</think_2>
</sample_1>
'''

# Prompt template cho batch processing (dynamic number of samples)
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
Tôi sẽ cung cấp cho bạn {batch_size} sample JSON input và bạn cần xử lý từng sample riêng biệt.

PHÂN BIỆT INPUT:
{chr(10).join(input_sections)}

QUY LUẬT VÀ VÍ DỤ XỬ LÝ:
{single_template}

OUTPUT YÊU CẦU:
Xử lý từng JSON input theo quy luật trên và đặt kết quả vào tag tương ứng:

{chr(10).join(output_sections)}

LƯU Ý QUAN TRỌNG:
- Mỗi <sample_n> chứa đầy đủ: json_n, prompt_n, think_n
- Progressive JSON: json_1 (part_1 only), json_2 (part_1+part_2), json_3 (part_1+part_2+part_3), etc.
- Tất cả tag phải được tạo như ví dụ
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

def survey_dataset(dataset_name, split_name):
    """Survey dataset to find maximum number of parts"""
    print(f"Loading dataset {dataset_name} with split {split_name}...")
    
    dataset = load_dataset(dataset_name, split=split_name)
    print(f"Dataset loaded with {len(dataset)} samples")
    
    max_parts = 0
    part_counts = {}
    valid_samples = 0
    
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
        
    print(f"\nSurvey Results:")
    print(f"Total samples: {len(dataset)}")
    print(f"Valid samples with JSON: {valid_samples}")
    print(f"Maximum number of parts found: {max_parts}")
    print(f"\nPart count distribution:")
    
    for count in sorted(part_counts.keys()):
        print(f"  {count} parts: {part_counts[count]} samples")
    
    return max_parts, part_counts

def extract_tags_from_response(response_text, max_parts):
    """Extract json_n, prompt_n, think_n tags from Gemini response"""
    extracted_data = {}
    
    # Extract tất cả tag có thể có (lên đến max_parts + thêm một ít để đảm bảo)
    for i in range(1, max_parts + 5):  # Thêm buffer
        # Extract json_n
        json_pattern = f'<json_{i}>\s*(.*?)\s*</json_{i}>'
        json_match = re.search(json_pattern, response_text, re.DOTALL)
        if json_match:
            extracted_data[f'json_{i}'] = json_match.group(1).strip()
        
        # Extract prompt_n
        prompt_pattern = f'<prompt_{i}>\s*(.*?)\s*</prompt_{i}>'
        prompt_match = re.search(prompt_pattern, response_text, re.DOTALL)
        if prompt_match:
            extracted_data[f'input_{i}'] = prompt_match.group(1).strip()
        
        # Extract think_n
        think_pattern = f'<think_{i}>\s*(.*?)\s*</think_{i}>'
        think_match = re.search(think_pattern, response_text, re.DOTALL)
        if think_match:
            extracted_data[f'think_{i}'] = think_match.group(1).strip()
    
    return extracted_data

def extract_samples_from_batch_response(response_text, batch_size):
    """Extract individual sample responses from batch response"""
    samples = {}
    
    for i in range(1, batch_size + 1):  # Extract sample_1 to sample_batch_size
        sample_pattern = f'<sample_{i}>\s*(.*?)\s*</sample_{i}>'
        sample_match = re.search(sample_pattern, response_text, re.DOTALL)
        if sample_match:
            sample_content = sample_match.group(1).strip()
            samples[f'sample_{i}'] = {
                'content': sample_content,
                'length': len(sample_content)
            }
    
    return samples

# Single sample processing removed - only using batch processing now

def process_batch_with_gemini(samples_batch, api_key, max_parts, batch_size):
    """Process a batch of samples with Gemini API"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash-lite-preview-06-17')
        
        # Get actual batch size (might be less than max batch_size for last batch)
        actual_batch_size = len(samples_batch)
        
        # Prepare input data for batch prompt
        input_data = {}
        for i, sample in enumerate(samples_batch, 1):
            input_data[f'input_{i}'] = sample['completion']
        
        # Create batch prompt with data embedded directly (no .format() needed)
        prompt = create_batch_prompt_template(actual_batch_size, single_prompt_template, input_data)
        
        response = model.generate_content(prompt)
        response_text = response.text
        
        # Extract individual samples from batch response
        extracted_samples = extract_samples_from_batch_response(response_text, actual_batch_size)
        
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
        print(f"Error processing batch: {str(e)}")
        return [None] * len(samples_batch)

def process_thread(dataset_samples, thread_id, api_key_index, max_parts, samples_per_thread, wait_seconds, start_idx, end_idx, batch_size):
    """Process samples in a thread using batch processing"""
    
    # Load API key
    api_key = os.getenv(f'GEMINI_API_KEY_{api_key_index}')
    if not api_key:
        print(f"Warning: GEMINI_API_KEY_{api_key_index} not found")
        return []
    
    processed_data = []
    
    print(f"Thread {thread_id}: Processing samples {start_idx} to {end_idx} with batch_size={batch_size}")
    
    # Process samples in batches
    dataset_list = list(dataset_samples)
    total_samples = len(dataset_list)
    
    for batch_start in tqdm(range(0, total_samples, batch_size), desc=f"Thread {thread_id}"):
        batch_end = min(batch_start + batch_size, total_samples)
        samples_batch = dataset_list[batch_start:batch_end]
        actual_batch_size = len(samples_batch)
        
        print(f"Thread {thread_id}: Processing batch {batch_start//batch_size + 1} with {actual_batch_size} samples")
        
        # Process batch with Gemini
        batch_results = process_batch_with_gemini(samples_batch, api_key, max_parts, batch_size)
        
        # Process each sample in the batch
        for idx, (sample, result) in enumerate(zip(samples_batch, batch_results)):
            # Create new sample with original data (always include, even if processing failed)
            new_sample = {
                'original_prompt': sample['prompt'],
                'original_completion': sample['completion']
            }
            
            # Add new_length field
            if result:
                new_sample['new_length'] = result.get('new_length', 0)
            else:
                new_sample['new_length'] = 0
            
            # Initialize all possible fields based on max_parts
            for i in range(1, max_parts + 1):
                if result:
                    new_sample[f'input_{i}'] = result.get(f'input_{i}', '')
                    new_sample[f'think_{i}'] = result.get(f'think_{i}', '')
                    new_sample[f'json_{i}'] = result.get(f'json_{i}', '')
                else:
                    # If processing failed, fill with empty strings
                    new_sample[f'input_{i}'] = ''
                    new_sample[f'think_{i}'] = ''
                    new_sample[f'json_{i}'] = ''
            
            processed_data.append(new_sample)
        
        # Wait to avoid API overload (wait after each batch)
        if batch_end < total_samples:  # Don't wait after last batch
            time.sleep(wait_seconds)
    
    return processed_data

def create_multi_turn_dataset(dataset_name, split_name, new_dataset_name, num_threads, samples_per_thread, wait_seconds, start_index=0, batch_size=5):
    """Main function to create multi-turn dataset"""
    
    # Load environment
    load_environment()
    
    # Survey dataset first
    print("Surveying dataset...")
    max_parts, distribution = survey_dataset(dataset_name, split_name)
    
    if not max_parts:
        print("No valid samples found!")
        return
    
    # Calculate total fields: original_prompt + original_completion + new_length + (input_i + think_i + json_i) * max_parts  
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
            api_key_index = (thread_id % 10) + 1  # Use GEMINI_API_KEY_1 to GEMINI_API_KEY_10
            
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
                print(f"Thread {thread_id} completed: {len(thread_results)} valid samples (samples {start_idx}-{end_idx})")
            except Exception as e:
                print(f"Thread {thread_id} failed: {str(e)}")
    
    if not all_processed_data:
        print("No data was processed successfully!")
        return
    
    print(f"Total processed samples: {len(all_processed_data)}")
    
    # Ensure all samples have the same structure before creating dataset
    if all_processed_data:
        # Create a template with all required fields
        field_template = {
            'original_prompt': '',
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
    
    # Push to hub
    end_index = start_index + total_samples
    new_split_name = f"{split_name}_{start_index}_{end_index}"
    print(f"Pushing dataset to {new_dataset_name} with split {new_split_name}")
    
    new_dataset.push_to_hub(new_dataset_name, split=new_split_name)
    print("Dataset pushed successfully!")

# Example usage
if __name__ == "__main__":
    # Parameters
    dataset_name = "wanhin/cad_hqh1"
    split_name = "range_500_1000_en"
    new_dataset_name = "wanhin/new_multi"
    num_threads = 8
    samples_per_thread = 10  # Total samples per thread
    wait_seconds = 55  # Wait time between batches
    start_index = 0  # Thêm tham số bắt đầu từ index nào
    batch_size = 10  # Number of samples per batch/API call
    
    create_multi_turn_dataset(
        dataset_name=dataset_name,
        split_name=split_name,
        new_dataset_name=new_dataset_name,
        num_threads=num_threads,
        samples_per_thread=samples_per_thread,
        wait_seconds=wait_seconds,
        start_index=start_index,
        batch_size=batch_size
    )