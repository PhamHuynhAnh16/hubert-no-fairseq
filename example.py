import os
import sys
import torch
import librosa
import logging

logging.getLogger("torch").setLevel(logging.ERROR)
sys.path.append(os.getcwd())

from fairseq.fairseq import load_model

is_half = False
output_layers = 12
input_audio = "test.wav"
input_hubert = "hubert_base.pt"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = load_model(input_hubert, unsafe_weight_allow=False)

def load_audio(file):
    audio, sr = librosa.load(file, sr=None)
    if len(audio.shape) > 1: audio = librosa.to_mono(audio.T)
    if sr != 16000: audio = librosa.resample(audio, orig_sr=sr, target_sr=16000, res_type="soxr_vhq")
    return audio.flatten()

feats = torch.from_numpy(load_audio(input_audio)).to(device).float()

if feats.dim() == 2: feats = feats.mean(-1)
assert feats.dim() == 1, feats.dim()

feats = feats.view(1, -1)

model.eval().to(device).to(torch.float16 if is_half else torch.float32).eval()

with torch.no_grad():
    logits = model.extract_features(source=feats, output_layer=output_layers)
    feats = logits[0]

print("Output:", feats.shape)
print(feats)