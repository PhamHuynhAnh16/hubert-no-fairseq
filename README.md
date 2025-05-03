<div align="center">

# Hubert Base No Fairseq

Suy luận mô hình Hubert Base mà không cần cài đặt Fairseq
</div>

---

Do Fairseq khó cài đặt trên python >= 3.11 và Fairseq có quá nhiều tính năng mà không sử dụng hết nên kho lưu trữ đã được tạo ra để tách các mã suy luận mô hình Hubert Base từ fairseq gốc để suy luận mà không cần phải tải Fairseq.

---

- Mã được thử nghiệm trên phiên bản python 3.11.8, pytorch 2.7.0 và omegaconf 2.3.0

Cài đặt và sử dụng
---
- Cài đặt 
```
python -m pip install torch omegaconf
```
Hoặc
```
python -m pip install -r requirements.txt
```
- Sử dụng
```python
import torch
from fairseq.fairseq import load_model

input_path = "hubert_base.pt" # Đường đẫn tệp Hubert Base
model, _, _ = load_model(input_path)

device = "cpu" # cuda:0
audio = ... # Đầu vào âm thanh để sử dụng Hubert Base

model = model[0]
model.eval().to(device)

with torch.no_grad():
    source = {
        "source": audio, 
        "padding_mask": torch.BoolTensor(audio.shape).fill_(False).to(device), 
        "output_layer": 12 # Số lượng layers đầu ra
    }
    logits = model.extract_features(**source)[0]
```
Bạn có thể sử dụng các contentvec_base từ kho lưu trữ [Contentvec](https://github.com/auspicious3000/contentvec) hoặc kho của [Vietnamese RVC](https://huggingface.co/AnhP/Vietnamese-RVC-Project/tree/main/embedders/fairseq)
Bạn có thể tham khảo tệp example.py để tham khảo thêm

Có thể sử dụng dự án để thay thế fairseq trong [RVC](https://github.com/fumiama/Retrieval-based-Voice-Conversion-WebUI/tree/main)
---
- Sao chép thư mục fairseq vào thư mục chính của RVC
- Mở tệp `web.py` ở dòng 75 - 77 sửa thành
```python
from fairseq import fairseq

fairseq.GradMultiply.forward = forward_dml
```
- Mở tệp `infer\lib\rtrvc.py` ở dòng 6, 39 và 73 sửa các dòng thành
```python
import fairseq -> from fairseq import fairseq

fairseq.modules.grad_multiply.GradMultiply.forward = forward_dml -> fairseq.GradMultiply.forward = forward_dml

models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
    ["assets/hubert/hubert_base.pt"],
    suffix="",
)
->
models, _, _ = fairseq.load_model(
    "assets/hubert/hubert_base.pt",
)
```
- Mở tệp `infer\modules\train\extract_feature_print.py` ở dòng 26, 47 và 91 sửa các dòng thành
```python
import fairseq -> from fairseq import fairseq

fairseq.modules.grad_multiply.GradMultiply.forward = forward_dml -> fairseq.GradMultiply.forward = forward_dml

models, saved_cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
    [model_path],
    suffix="",
)
->
models, saved_cfg, task = fairseq.load_model(
    model_path,
)
```
- Mở tệp `infer\modules\vc\utils.py` ở dòng 3 và 24 sửa các dòng thành
```python
from fairseq import checkpoint_utils -> from fairseq import fairseq

models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
    ["assets/hubert/hubert_base.pt"],
    suffix="",
)
->
models, _, _ = fairseq.load_model(
    "assets/hubert/hubert_base.pt",
)
```
- Mở tệp `rvc\hubert.py` ở dòng 5, 6 và 266 sửa các dòng thành
```python
from fairseq.checkpoint_utils import load_model_ensemble_and_task -> from fairseq.fairseq import load_model, index_put

from fairseq.utils import index_put # Xóa dòng này

models, _, _ = load_model_ensemble_and_task(
    [model_path],
    suffix="",
)
->
models, _, _ = load_model(
    model_path,
)
```

# Miễn trừ trách nhiệm
---
Phần mềm này được cung cấp "nguyên trạng", **không có bất kỳ cam kết hoặc bảo đảm nào**, dù rõ ràng hay ngụ ý, bao gồm nhưng không giới hạn ở các bảo đảm về khả năng thương mại, sự phù hợp cho một mục đích cụ thể hoặc không vi phạm quyền của bên thứ ba.

**Tác giả và các cộng tác viên không chịu trách nhiệm** đối với bất kỳ thiệt hại, mất mát dữ liệu, hoặc các hậu quả pháp lý nào phát sinh từ việc sử dụng hoặc sử dụng sai phần mềm này.

Người dùng hoàn toàn tự chịu trách nhiệm khi sử dụng phần mềm. Vui lòng đảm bảo bạn hiểu rõ các điều khoản cấp phép và tuân thủ các quy định pháp luật hiện hành khi sử dụng phần mềm này.

Nếu phần mềm được sử dụng để xử lý dữ liệu giọng nói hoặc nhận dạng cá nhân, **không sử dụng vào các mục đích xâm phạm quyền riêng tư, mạo danh, theo dõi, hoặc các hành vi vi phạm pháp luật và đạo đức.**
