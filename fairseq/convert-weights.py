import sys
import types
import torch
import traceback

model_path = r"G:\Assets\RVC\embedders\hubert_base.pt"
output_path = "hubert_base_safe.pt"

class Dictionary:
    def __init__(self, *args, **kwargs):
        pass

fairseq = types.ModuleType("fairseq")
fairseq_data = types.ModuleType("fairseq.data")
fairseq_data_dictionary = types.ModuleType("fairseq.data.dictionary")
fairseq_data_dictionary.Dictionary = Dictionary
fairseq.data = fairseq_data
fairseq_data.dictionary = fairseq_data_dictionary
sys.modules["fairseq"] = fairseq
sys.modules["fairseq.data"] = fairseq_data
sys.modules["fairseq.data.dictionary"] = fairseq_data_dictionary

state = torch.load(model_path, map_location="cpu", weights_only=False)
torch.save({key: state[key] for key in ['cfg', 'model']}, output_path)
try:
    torch.load(output_path, map_location="cpu", weights_only=True)
    print("✅ Đã chuyển đổi thành công!")
except:
    print("Đã xảy ra lỗi khi chuyển đổi trọng số!")
    print(traceback.format_exc())