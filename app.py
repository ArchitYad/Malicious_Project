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
from lime.lime_text import LimeTextExplainer
import numpy as np
from groq import Groq
import json

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
# SRNet MODEL
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

@st.cache_resource
def load_srnet():
    model = SRNet()
    checkpoint = torch.load("model/srnet_epoch3_best.pth", map_location="cpu")
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)
    model.eval()
    return model

srnet = load_srnet()

# ==========================================
# IMAGE LOADER
# ==========================================
def load_image(path):
    img = Image.open(path).convert("RGB")
    img_np = np.array(img)
    gray = np.dot(img_np[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    return img_np, gray

# ==========================================
# RAW LSB EXTRACTOR (MATCH TRAINING)
# ==========================================
class RawExtractor:
    def __init__(self, path):
        img = cv2.imread(path)
        if img is None:
            raise Exception("Image not found")
        self.flat_pixels = img.flatten()

    def extract(self, limit=5000):
        bits = "".join([str(p & 1) for p in self.flat_pixels[:limit * 8]])
        byte_arr = [int(bits[i:i+8], 2) for i in range(0, len(bits), 8)]
        return bytes(byte_arr).decode('utf-8', errors='ignore')

# ==========================================
# CLEAN TEXT (BASIC)
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
# FEATURE EXTRACTION (MATCH TRAINING)
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

@st.cache_resource
def load_classifier(label):
    from transformers import pipeline

    model_map = {
        "url": "Arch11yad/url_malicious_detect",
        "js": "Arch11yad/js_malicious_detect",
        "html": "Arch11yad/HTML_Malicious_detect_y",
        "ps": "Arch11yad/powershell_final",
    }

    return pipeline(
        "text-classification",
        model=model_map[label],
        token=st.secrets["HF_TOKEN"]
    )

def get_lime_explanation(text, classifier):

    def predict_proba(texts):
        results = classifier(texts)

        probs = []
        for r in results:
            if isinstance(r, list):
                r = r[0]

            score = r["score"]

            if "malicious" in r["label"].lower():
                probs.append([1 - score, score])
            else:
                probs.append([score, 1 - score])

        return np.array(probs)

    explainer = LimeTextExplainer(class_names=["benign", "malicious"])

    exp = explainer.explain_instance(
        text[:300],        # 🔥 shorter text
        predict_proba,
        num_features=6,    # 🔥 fewer words
        num_samples=50     # 🔥 CRITICAL FIX (no freeze)
    )

    return [w for w, _ in exp.as_list()]

@st.cache_resource
def load_groq():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

groq_client = load_groq()

def build_prompt(cleaned, lime_words, prediction, confidence):

    snippet = cleaned[:800]   # 🔥 keep it short

    return f"""
You are a cybersecurity malware analyst.

Analyze the extracted content from a steganographic image.

--- CLEANED TEXT (partial) ---
{snippet}

--- IMPORTANT WORDS (LIME) ---
{lime_words}

--- MODEL OUTPUT ---
Prediction: {prediction}
Confidence: {confidence}

TASK:
1. Identify attack type
2. Explain what the attack does
3. Give severity score (1–10)
4. Suggest precautions

Return ONLY JSON:

{{
  "attack_type": "...",
  "severity_score": "...",
  "what_it_does": "...",
  "precautions": "..."
}}
"""

def get_ai_report(cleaned, lime_words, prediction, confidence):

    prompt = build_prompt(cleaned, lime_words, prediction, confidence)

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)
# ==========================================
# TABS
# ==========================================
tab1, tab2, tab3 = st.tabs(["🔍 Image Analysis", "🧾 Clean Output", "🤖 AI Report"])
# ==========================================
# TAB 1
# ==========================================
with tab1:

    uploaded = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])

    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded.read())
            path = tmp.name

        st.image(uploaded)

        img, heatmap, lsb, prob = analyze(path)

        st.metric("Stego Probability", f"{prob*100:.2f}%")

        col1, col2, col3 = st.columns(3)
        col1.image(img, caption="Original")
        col2.image(heatmap, caption="SRNet Noise Map")
        col3.image(lsb, caption="LSB Bit Plane")

        # RAW TEXT
        extractor = RawExtractor(path)
        raw_text = extractor.extract()

        st.session_state["raw_text"] = raw_text

        st.subheader("📄 Raw Extracted Text")
        st.code(raw_text[:1000] if raw_text else "[No Data Found]")

        # CLASSIFICATION
        label, probs = predict_class(raw_text)

        st.session_state["label"] = label

        st.subheader("🧠 Detected Class")
        st.write(label.upper())

        st.subheader("📊 Probabilities")
        st.write(probs)

with tab2:

    if "raw_text" not in st.session_state or "label" not in st.session_state:
        st.warning("⚠️ Please upload image in Tab 1 first.")
    else:
        raw_text = st.session_state["raw_text"]
        label = st.session_state["label"]

        st.subheader("Detected Type")
        st.write(label.upper())

        # ================= CLEANING =================
        def clean_text_by_label(text, label):

            if label == "url":
                match = re.search(r"https?://[^\s'\"<>]+", text)
                return match.group(0) if match else ""

            elif label == "html":
                match = re.search(r"(<.*(?:</html>|</script>))", text, re.DOTALL | re.IGNORECASE)
                return match.group(1) if match else ""

            elif label == "js":
                cleaned = text.replace("?", "")
                cleaned = "".join(c for c in cleaned if c.isprintable() or c in "\n\r\t")
                return cleaned.strip()

            elif label == "eth":
                match = re.search(r"0x[a-fA-F0-9]{40}", text)
                return match.group(0) if match else ""

            elif label == "ps":
                cleaned = "".join(c for c in text if c.isprintable() or c in "\n\r\t")
                return cleaned.strip()

            return text

        cleaned = clean_text_by_label(raw_text, label)

        st.subheader("🧾 Cleaned Output")
        st.code(cleaned if cleaned else "[No valid pattern found]")

        # ================= MALICIOUS DETECTION =================
        st.subheader("🚨 Malicious Analysis")

        if not cleaned or len(cleaned) < 5:
            st.warning("Text too small for analysis")
        else:
            try:
                if label in ["url", "js", "html", "ps"]:

                        classifier = load_classifier(label)
                    
                        result = classifier(cleaned[:512])[0]
                        label_out = result["label"]
                        if label_out.lower() in ["label_1", "malicious"]:
                            label_out = "Malicious"
                        else:
                            label_out = "Benign"
                        st.success(f"Prediction: {label_out}")
                        st.write("Confidence:", round(result["score"], 3))
                    
                        # ===== LIME (FIXED) =====
                        with st.spinner("Generating explanation..."):
                            try:
                                important_words = get_lime_explanation(cleaned, classifier)
                            except Exception as e:
                                st.warning("LIME failed, fallback used")
                                important_words = []
                    
                        st.subheader("🧠 Important Words (LIME)")
                        st.write(important_words if important_words else "[No strong signals]")
                    
                        st.session_state["lime_words"] = important_words
                        st.session_state["cleaned_text"] = cleaned
                        st.session_state["malicious_label"] = label_out
                        st.session_state["malicious_score"] = result["score"]

                elif label == "eth":

                    # -------- ETH FTTransformer --------
                    class FTTransformer(nn.Module):
                        def __init__(self, input_dim=1, d_model=64, n_heads=4, n_layers=2):
                            super().__init__()
                            self.embedding = nn.Linear(input_dim, d_model)

                            encoder_layer = nn.TransformerEncoderLayer(
                                d_model=d_model, nhead=n_heads, batch_first=True
                            )

                            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

                            self.fc = nn.Sequential(
                                nn.Linear(d_model, 64),
                                nn.ReLU(),
                                nn.Linear(64, 2)
                            )

                        def forward(self, x):
                            x = self.embedding(x)
                            x = x.unsqueeze(1)
                            x = self.transformer(x)
                            x = x.mean(dim=1)
                            return self.fc(x)

                    eth_model = FTTransformer()
                    eth_model.load_state_dict(torch.load("model/ethaddress_model.pth", map_location="cpu"))
                    eth_model.eval()

                    x = torch.tensor([[len(cleaned)]], dtype=torch.float32)

                    with torch.no_grad():
                        logits = eth_model(x)
                        probs = torch.softmax(logits, dim=1)[0]

                    pred = "Malicious" if probs[1] > 0.5 else "Safe"

                    st.success(f"Prediction: {pred}")
                    st.write("Confidence:", float(probs[1]))

                    st.info("ETH detected → skipping LIME")
                    st.session_state["lime_words"] = []
                    st.session_state["cleaned_text"] = cleaned
                    st.session_state["malicious_label"] = pred
                    st.session_state["malicious_score"] = float(probs[1])
                else:
                    st.info("No model available for this type")

            except Exception as e:
                st.error(f"Detection failed: {str(e)}")

with tab3:

    if "cleaned_text" not in st.session_state:
        st.warning("⚠️ Run Tab 2 first")
    else:

        cleaned = st.session_state.get("cleaned_text", "")
        lime_words = st.session_state.get("lime_words", [])
        prediction = st.session_state.get("malicious_label", "unknown")
        confidence = st.session_state.get("malicious_score", 0.0)

        # truncate
        cleaned = cleaned[:800]
        lime_words = lime_words[:10]

        # ================= HEADER =================
        st.markdown("## 🤖 AI Forensic Report Dashboard")

        # ================= SUMMARY CARDS =================
        col1, col2 = st.columns(2)

        # Risk color logic
        if prediction.lower() in ["malicious", "label_1"]:
            risk_color = "🔴 High Risk"
        else:
            risk_color = "🟢 Low Risk"

        col1.metric("Prediction", prediction)
        col2.metric("Confidence", f"{confidence:.2f}")

        st.markdown(f"### Risk Level: {risk_color}")

        # ================= LIME WORDS =================
        st.markdown("### 🧠 Important Indicators")
        if lime_words:
            st.write(" | ".join([f"`{w}`" for w in lime_words]))
        else:
            st.info("No strong indicators found")

        # ================= CLEAN TEXT PREVIEW =================
        with st.expander("📄 View Extracted Text (Partial)"):
            st.code(cleaned if cleaned else "[No data]")

        # ================= AI REPORT =================
        st.markdown("### 🔍 AI Analysis")

        if not cleaned:
            st.warning("No cleaned text available")
        else:
            with st.spinner("Analyzing threat..."):
                try:
                    report = get_ai_report(
                        cleaned,
                        lime_words,
                        prediction,
                        confidence
                    )

                    st.success("Analysis Complete")

                    # ===== BEAUTIFUL OUTPUT =====
                    st.markdown("### 🛡️ Threat Details")

                    colA, colB = st.columns(2)

                    colA.markdown(f"**Attack Type:**  \n{report.get('attack_type','-')}")
                    colB.markdown(f"**Severity Score:**  \n{report.get('severity_score','-')} / 10")

                    st.markdown("### ⚙️ What the Attack Does")
                    st.info(report.get("what_it_does", "-"))

                    st.markdown("### 🧯 Recommended Precautions")
                    st.success(report.get("precautions", "-"))

                except Exception as e:
                    st.error(f"Failed to generate report: {str(e)}")
