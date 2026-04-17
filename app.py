import streamlit as st
import torch
import os

st.set_page_config(page_title="Model Debug", layout="centered")
st.title("🧪 SRNet Model Debugger")

model_path = "model/srnet_epoch3_best.pth"

st.write("📂 Model Path:", model_path)

# ==========================================
# 1. Check if file exists
# ==========================================
if os.path.exists(model_path):
    st.success("✅ Model file FOUND")

    file_size = os.path.getsize(model_path) / (1024 * 1024)
    st.write(f"📦 File Size: {file_size:.2f} MB")
else:
    st.error("❌ Model file NOT FOUND")
    st.stop()


# ==========================================
# 2. Try loading model file
# ==========================================
st.subheader("🔄 Loading checkpoint...")

try:
    checkpoint = torch.load(model_path, map_location="cpu")
    st.success("✅ Checkpoint LOADED successfully")
except Exception as e:
    st.error(f"❌ torch.load FAILED: {e}")
    st.stop()


# ==========================================
# 3. Inspect checkpoint structure
# ==========================================
st.subheader("📊 Checkpoint Info")

if isinstance(checkpoint, dict):
    st.write("Keys in checkpoint:")
    st.write(list(checkpoint.keys()))

    if "model_state_dict" in checkpoint:
        sd = checkpoint["model_state_dict"]
        st.success("✅ Found 'model_state_dict'")
    else:
        sd = checkpoint
        st.warning("⚠️ Using checkpoint directly as state_dict")
else:
    st.error("❌ Unexpected checkpoint format")
    st.stop()


# ==========================================
# 4. Try loading into dummy model
# ==========================================
import torch.nn as nn
import torch.nn.functional as F

class SRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 64, 3, 1, 1)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = self.conv(x)
        x = F.adaptive_avg_pool2d(x, (1,1))
        return self.fc(x.view(x.size(0), -1))


st.subheader("🧠 Loading into Model")

model = SRNet()

try:
    missing, unexpected = model.load_state_dict(sd, strict=False)

    st.success("✅ Model LOADED into architecture")

    st.write("🔍 Missing keys:", missing)
    st.write("🔍 Unexpected keys:", unexpected)

except Exception as e:
    st.error(f"❌ Model loading FAILED: {e}")
    st.stop()


# ==========================================
# 5. Final Test (forward pass)
# ==========================================
st.subheader("⚡ Forward Pass Test")

try:
    dummy = torch.randn(1, 1, 256, 256)
    out = model(dummy)
    st.success("✅ Forward pass SUCCESS")
    st.write("Output shape:", out.shape)
except Exception as e:
    st.error(f"❌ Forward pass FAILED: {e}")
