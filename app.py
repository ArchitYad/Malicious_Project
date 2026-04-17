import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import tempfile
import os
import re
from PIL import Image

# ==========================================
# 1. SRNet Model
# ==========================================
class SRNet(nn.Module):
    def __init__(self, num_classes=2):
        super(SRNet, self).__init__()
        self.layer1 = self._make_layer12(1, 64)
        self.layer2 = self._make_layer12(64, 64)
        self.layer3 = self._make_res(64, 64)
        self.layer4 = self._make_res(64, 128)
        self.layer5 = self._make_res(128, 128)
        self.layer6 = self._make_res(128, 256)
        self.layer7 = self._make_res(256, 256)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def _make_layer12(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True)
        )

    def _make_res(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        l1 = self.layer1(x)
        noise_map = self.layer2(l1)
        x = F.relu(self.layer3(noise_map))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        x = F.relu(self.layer6(x))
        x = F.relu(self.layer7(x))
        x = self.avgpool(x)
        return self.fc(torch.flatten(x, 1)), noise_map


# ==========================================
# 2. Data Extractor (same as your file)
# ==========================================
class CodeBERTDataExtractor:
    def __init__(self, path):
        self.path = path
        self.filename = os.path.basename(path).lower()
        self.image = cv2.imread(path)
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

        if "_js_" in self.filename:
            return "".join(c for c in text if c.isprintable())
        elif "_url_" in self.filename:
            match = re.search(r"https?://[^\s]+", text)
            return match.group(0) if match else ""
        elif "_eth_" in self.filename:
            match = re.search(r"0x[a-fA-F0-9]{40}", text)
            return match.group(0) if match else ""

        return text


# ==========================================
# 3. Load Model
# ==========================================
@st.cache_resource
def load_model(model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SRNet().to(device)

    checkpoint = torch.load(model_path, map_location=device)
    sd = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(sd, strict=False)
    model.eval()

    return model, device


# ==========================================
# 4. Analysis Function
# ==========================================
def analyze_image(img_path, model, device):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    t = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).float().to(device)
    t = (t - 127.5) / 127.5

    with torch.no_grad():
        logits, noise_feat = model(t)
        prob = torch.softmax(logits, 1)[0, 1].item()

    # Noise map
    heatmap = torch.mean(torch.abs(noise_feat), 1).squeeze().cpu().numpy()
    heatmap = cv2.applyColorMap(
        cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.COLORMAP_JET
    )

    # LSB
    lsb = (gray & 1) * 255

    # Magnified bits
    patch = gray[0:10, 0:10] & 1

    # Difference image
    reconstructed = gray & ~1
    diff = cv2.absdiff(gray, reconstructed)

    return img, heatmap, lsb, patch, diff, prob


# ==========================================
# 5. Streamlit UI
# ==========================================
st.set_page_config(page_title="Stego Detection App", layout="wide")
st.title("🔍 Steganography + Malicious Payload Detector")

model_path = "model/srnet.pth"  # or downloaded from GitHub

model, device = load_model(model_path)

uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        img_path = tmp.name

    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    img, heatmap, lsb, patch, diff, prob = analyze_image(img_path, model, device)

    st.subheader("📊 Detection Result")
    st.metric("Stego Probability", f"{prob*100:.2f}%")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original")

    with col2:
        st.image(heatmap, caption="SRNet Noise Map")

    with col3:
        st.image(lsb, caption="LSB Bit Plane")

    st.subheader("🔬 Bit Analysis")

    st.image(patch * 255, caption="Magnified Bits (0/1)")

    st.subheader("⚡ Difference Image")
    st.image(diff, caption="Original vs Clean")

    st.subheader("🧠 Extracted Hidden Data")

    extractor = CodeBERTDataExtractor(img_path)
    text = extractor.process_for_classifier()

    st.code(text if text else "No hidden data found")
