import os
import sys
import torch
import librosa
import logging

logging.getLogger("torch").setLevel(logging.ERROR)
sys.path.append(os.getcwd())

from fairseq.fairseq import load_model

is_half = False
device = "cpu"
output_layers = 12
input_audio = "test.wav"
input_hubert = r"G:\RVC\embedders\fairseq\hubert_base.pt"

model, cfg, task = load_model(input_hubert)

def load_audio(file):
    audio, sr = librosa.load(file, sr=None)
    if len(audio.shape) > 1: audio = librosa.to_mono(audio.T)
    if sr != 16000: audio = librosa.resample(audio, orig_sr=sr, target_sr=16000, res_type="soxr_vhq")
    return audio.flatten()

feats = torch.from_numpy(load_audio(input_audio)).float()

if feats.dim() == 2: feats = feats.mean(-1)
assert feats.dim() == 1, feats.dim()

feats = feats.view(1, -1)

model = model[0]
model.eval()
model = model.to(device).to(torch.float16 if is_half else torch.float32).eval()

if output_layers == 12:
    with torch.no_grad():
        logits = model.extract_features(**{"source": feats, "padding_mask": torch.BoolTensor(feats.shape).fill_(False).to(device), "output_layer": 12})
        feats = logits[0]

    print("shape 12 layers:", feats.shape)
    print(feats)
else:
    with torch.no_grad():
        logits = model.extract_features(**{"source": feats, "padding_mask": torch.BoolTensor(feats.shape).fill_(False).to(device), "output_layer": 9})
        feats = model.final_proj(logits[0])

    print("shape 9 layers:", feats.shape)
    print(feats)