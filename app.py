import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tempfile
import re
import joblib
import cv2
from PIL import Image

# ==========================================
# CONFIG
# ==========================================
st.set_page_config(page_title="Stego Detector", layout="wide")
st.title("🔍 Stego Detection + Raw Payload Classification")

# ==========================================
# LABEL MAP
# ==========================================
label_map = {"js": 0, "html": 1, "ps": 2, "eth": 3, "url": 4}
reverse_label_map = {v: k for k, v in label_map.items()}

# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    return joblib.load("model/final_fast_model.pkl")

clf = load_model()

# ==========================================
# IMAGE LOADER
# ==========================================
def load_image(path):
    img = Image.open(path).convert("RGB")
    img_np = np.array(img)
    gray = np.dot(img_np[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    return img_np, gray

# ==========================================
# RAW LSB EXTRACTOR (EXACT SAME AS TRAINING)
# ==========================================
class RawExtractor:
    def __init__(self, path):
        img = cv2.imread(path)   # ✅ SAME AS TRAINING
        if img is None:
            raise Exception("Image not found")
        self.flat_pixels = img.flatten()

    def extract(self, limit=5000):
        bits = "".join([str(p & 1) for p in self.flat_pixels[:limit * 8]])
        byte_arr = [int(bits[i:i+8], 2) for i in range(0, len(bits), 8)]
        return bytes(byte_arr).decode('utf-8', errors='ignore')

# ==========================================
# CLEAN TEXT (ONLY BASIC)
# ==========================================
def clean_text(text):
    return re.sub(r"[^\x20-\x7E\n\r\t]", "", text)

# ==========================================
# ENTROPY
# ==========================================
def entropy(text):
    if len(text) == 0:
        return 0
    probs = [text.count(c)/len(text) for c in set(text)]
    return -sum(p*np.log2(p) for p in probs if p > 0)

# ==========================================
# FEATURE EXTRACTION (MUST MATCH TRAINING)
# ==========================================
def extract_features(text):
    if len(text) == 0:
        return [0]*25

    length = len(text)

    return [
        length,
        entropy(text),

        sum(c.isalpha() for c in text)/length,
        sum(c.isdigit() for c in text)/length,
        sum(c.isspace() for c in text)/length,

        text.count(";")/length,
        text.count("{")/length,
        text.count("}")/length,
        text.count("<")/length,
        text.count(">")/length,
        text.count("=")/length,
        text.count("(")/length,
        text.count(")")/length,

        int("http" in text),
        int("https" in text),
        int("0x" in text),
        int("<html" in text.lower()),
        int("function" in text),
        int("var" in text),
        int("let" in text),
        int("const" in text),
        int("powershell" in text.lower()),
        int("invoke" in text.lower()),

        text.count("?")/length,
        text.count("@")/length,
    ]

# ==========================================
# CLASSIFICATION
# ==========================================
def predict_class(text):
    text = clean_text(text)
    feat = np.array([extract_features(text)])

    probs = clf.predict_proba(feat)[0]
    pred = np.argmax(probs)

    return reverse_label_map[pred], probs

# ==========================================
# SRNet MODEL (FOR VISUALIZATION)
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
# LOAD SRNET
# ==========================================
@st.cache_resource
def load_srnet():
    model = SRNet()
    checkpoint = torch.load("model/srnet_epoch3_best.pth", map_location="cpu")
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)
    model.eval()
    return model

srnet = load_srnet()

# ==========================================
# ANALYSIS
# ==========================================
def analyze(path):
    img, gray = load_image(path)

    t = torch.tensor(gray).unsqueeze(0).unsqueeze(0).float()
    t = (t - 127.5) / 127.5

    with torch.no_grad():
        logits, noise = srnet(t)
        prob = torch.softmax(logits,1)[0,1].item()

    heatmap = torch.mean(torch.abs(noise),1).squeeze().numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    lsb = (gray & 1) * 255

    return img, heatmap, lsb, prob

# ==========================================
# UI
# ==========================================
uploaded = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded.read())
        path = tmp.name

    st.image(uploaded, caption="Uploaded Image")

    img, heatmap, lsb, prob = analyze(path)

    st.metric("Stego Probability", f"{prob*100:.2f}%")

    col1, col2, col3 = st.columns(3)
    col1.image(img, caption="Original")
    col2.image(heatmap, caption="SRNet Noise Map")
    col3.image(lsb, caption="LSB Bit Plane")

    # ===== RAW TEXT =====
    extractor = RawExtractor(path)
    raw_text = extractor.extract()

    st.subheader("📄 Raw Extracted Text")
    st.code(raw_text[:1000] if raw_text else "[No Data Found]")

    # ===== CLASSIFICATION =====
    label, probs = predict_class(raw_text)

    st.subheader("🧠 Detected Class")
    st.write(label.upper())

    st.subheader("📊 Probabilities")
    st.write(probs)
