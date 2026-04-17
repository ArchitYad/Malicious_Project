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
st.title("🔍 Steganography + Malicious Payload Detector")

# ==========================================
# IMAGE LOADER (NO CV2)
# ==========================================
def load_image(path):
    img = Image.open(path).convert("RGB")
    img_np = np.array(img)

    # Convert to grayscale
    gray = np.dot(img_np[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

    return img_np, gray


# ==========================================
# DATA EXTRACTOR (YOUR LOGIC)
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

        if "_url_" in self.filename:
            url_match = re.search(r"https?://[^\s'\"<>]+", text)
            return url_match.group(0) if url_match else ""

        elif "_html_" in self.filename:
            html_match = re.search(r"(<.*(?:</html>|</script>))\s*$", text, re.DOTALL | re.IGNORECASE)
            return html_match.group(1) if html_match else ""

        elif "_js_" in self.filename:
            cleaned_js = text.replace('?', '').strip()
            return "".join(c for c in cleaned_js if c.isprintable() or c in "\n\r\t")

        elif "_eth_" in self.filename:
            eth_match = re.search(r"0x[a-fA-F0-9]{40}", text)
            return eth_match.group(0) if eth_match else ""

        elif "_ps_" in self.filename:
            return "".join(c for c in text if c.isprintable() or c in "\n\r\t").strip()

        return text


# ==========================================
# SRNet (UNCHANGED)
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
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_model(model_path):
    device = "cpu"  # stable for deployment
    model = SRNet().to(device)

    if not os.path.exists(model_path):
        st.error(f"❌ Model not found: {model_path}")
        return None, device

    checkpoint = torch.load(model_path, map_location=device)
    sd = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

    model.load_state_dict(sd, strict=False)
    model.eval()

    return model, device


# ==========================================
# ANALYSIS FUNCTION
# ==========================================
def analyze(img_path, model, device):
    img, gray = load_image(img_path)

    t = torch.tensor(gray).unsqueeze(0).unsqueeze(0).float().to(device)
    t = (t - 127.5) / 127.5

    with torch.no_grad():
        logits, noise = model(t)
        prob = torch.softmax(logits, 1)[0, 1].item()

    # Heatmap
    heatmap = torch.mean(torch.abs(noise), 1).squeeze().cpu().numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # LSB
    lsb = (gray & 1) * 255

    # Magnified patch
    patch = gray[:10, :10] & 1

    # ✅ FIXED DIFFERENCE IMAGE (NO OVERFLOW)
    diff = np.abs(gray.astype(np.int16) - (gray & 254).astype(np.int16)).astype(np.uint8)

    return img, heatmap, lsb, patch, diff, prob


# ==========================================
# UI
# ==========================================
model_path = "model/srnet_epoch3_best.pth"

uploaded = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded.read())
        path = tmp.name

    st.image(uploaded, caption="Uploaded Image")

    # Load model
    with st.spinner("🔄 Loading model..."):
        model, device = load_model(model_path)

    if model is None:
        st.stop()

    # Analyze
    with st.spinner("🔍 Analyzing image..."):
        img, heatmap, lsb, patch, diff, prob = analyze(path, model, device)

    # Output
    st.subheader("📊 Stego Probability")
    st.metric("Probability", f"{prob*100:.2f}%")

    col1, col2, col3 = st.columns(3)
    col1.image(img, caption="Original Image")
    col2.image(heatmap, caption="SRNet Noise Map")
    col3.image(lsb, caption="LSB Plane")

    st.subheader("🔬 Magnified Bits")
    st.image(patch * 255, caption="Bit Pattern (0/1)")

    st.subheader("⚡ Difference Image")
    st.image(diff, caption="Original vs Clean")

    # Extract hidden data
    st.subheader("🧠 Extracted Hidden Data")
    extractor = CodeBERTDataExtractor(path)
    text = extractor.process_for_classifier()

    st.code(text if text else "[No Valid Pattern Found]")
