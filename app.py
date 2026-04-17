import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tempfile
import os
import re
from PIL import Image

# ==========================================
# CONFIG
# ==========================================
st.set_page_config(page_title="Stego Detection", layout="wide")
st.title("🔍 Steganography Detector (No OpenCV)")

# ==========================================
# IMAGE LOADER (REPLACES CV2)
# ==========================================
def load_image(path):
    img = Image.open(path).convert("RGB")
    img_np = np.array(img)

    # Convert to grayscale manually
    gray = np.dot(img_np[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

    return img_np, gray


# ==========================================
# DATA EXTRACTOR (UPDATED)
# ==========================================
class CodeBERTDataExtractor:
    def __init__(self, path):
        self.path = path
        self.filename = os.path.basename(path).lower()

        img = Image.open(path).convert("RGB")
        self.image = np.array(img)

        self.flat_pixels = self.image.flatten()

    def get_raw_text(self, limit=5000):
        header_bits = "".join([str(p & 1) for p in self.flat_pixels[:64]])

        try:
            data_len = int(header_bits, 2)
        except:
            data_len = 0

        if 0 < data_len < (len(self.flat_pixels) // 8):
            start_bit, end_bit = 64, 64 + (data_len * 8)
        else:
            start_bit, end_bit = 0, limit * 8

        bits = "".join([str(p & 1) for p in self.flat_pixels[start_bit:end_bit]])
        byte_arr = [int(bits[i:i+8], 2) for i in range(0, len(bits), 8)]

        return bytes(byte_arr).decode('utf-8', errors='ignore')

    def process_for_classifier(self):
        text = self.get_raw_text()
        return text


# ==========================================
# SRNet (UNCHANGED)
# ==========================================
class SRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1,64,3,1,1), nn.BatchNorm2d(64), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(64,64,3,1,1), nn.BatchNorm2d(64), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Conv2d(64,64,3,1,1), nn.BatchNorm2d(64))
        self.layer4 = nn.Sequential(nn.Conv2d(64,128,3,1,1), nn.BatchNorm2d(128))
        self.layer5 = nn.Sequential(nn.Conv2d(128,128,3,1,1), nn.BatchNorm2d(128))
        self.layer6 = nn.Sequential(nn.Conv2d(128,256,3,1,1), nn.BatchNorm2d(256))
        self.layer7 = nn.Sequential(nn.Conv2d(256,256,3,1,1), nn.BatchNorm2d(256))
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256,2)

    def forward(self,x):
        x = self.layer1(x)
        noise = self.layer2(x)
        x = F.relu(self.layer3(noise))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        x = F.relu(self.layer6(x))
        x = F.relu(self.layer7(x))
        x = self.pool(x)
        return self.fc(x.view(x.size(0),-1)), noise


# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_model(path):
    device = "cpu"
    model = SRNet().to(device)

    checkpoint = torch.load(path, map_location=device)
    sd = checkpoint.get("model_state_dict", checkpoint)

    model.load_state_dict(sd, strict=False)
    model.eval()

    return model, device


# ==========================================
# ANALYSIS
# ==========================================
def analyze(img_path, model, device):
    img, gray = load_image(img_path)

    t = torch.tensor(gray).unsqueeze(0).unsqueeze(0).float()
    t = (t - 127.5) / 127.5

    with torch.no_grad():
        logits, noise = model(t)
        prob = torch.softmax(logits,1)[0,1].item()

    heatmap = torch.mean(torch.abs(noise),1).squeeze().numpy()

    # normalize heatmap
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    lsb = (gray & 1) * 255
    patch = gray[:10,:10] & 1
    diff = np.abs(gray - (gray & ~1))

    return img, heatmap, lsb, patch, diff, prob


# ==========================================
# UI
# ==========================================
model_path = "model/srnet_epoch3_best.pth"

uploaded = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded.read())
        path = tmp.name

    st.image(uploaded)

    model, device = load_model(model_path)

    with st.spinner("Analyzing..."):
        img, heatmap, lsb, patch, diff, prob = analyze(path, model, device)

    st.metric("Stego Probability", f"{prob*100:.2f}%")

    col1,col2,col3 = st.columns(3)
    col1.image(img, caption="Original")
    col2.image(heatmap, caption="Noise Map")
    col3.image(lsb, caption="LSB")

    st.image(patch*255, caption="Magnified Bits")
    st.image(diff, caption="Difference")

    extractor = CodeBERTDataExtractor(path)
    text = extractor.process_for_classifier()

    st.code(text if text else "No hidden data")
