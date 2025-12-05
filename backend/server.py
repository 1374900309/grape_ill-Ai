import io
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv  # ✅ 新增导入

# ========== 加载 .env 环境变量 ==========
load_dotenv()  # 会自动读取 backend 目录下的 .env 文件

# 让 Python 能找到 grape_cls.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grape_cls import build_model, load_model

# ========== Disease Mapping ==========
disease_advice_map = {
    "Tomato_healthy": {"name": "Healthy", "advice": "Leaves are healthy, no treatment required."},
    "Tomato_Early_blight": {"name": "Early blight", "advice": "Remove diseased leaves and apply fungicides."},
    "Tomato_Late_blight": {"name": "Late blight", "advice": "Apply targeted fungicides and control humidity."},
    "Tomato_Leaf_Mold": {"name": "Leaf Mold", "advice": "Maintain ventilation and spray fungicides if needed."},
    "Tomato_Septoria_leaf_spot": {"name": "Septoria leaf spot", "advice": "Remove diseased leaves and apply protective agents."},
    "Tomato_Spider_mites_Two_spotted_spider_mite": {"name": "Two-spotted spider mite", "advice": "Control mite infestation using proper pesticides."},
    "Tomato__Target_Spot": {"name": "Target Spot", "advice": "Avoid dense planting and improve ventilation."},
    "Tomato__Tomato_mosaic_virus": {"name": "Tomato mosaic virus", "advice": "Remove infected plants and ensure seed health."},
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {"name": "Tomato yellow leaf curl virus", "advice": "Control insect vectors and remove infected plants."},
    "Tomato_Bacterial_spot": {"name": "Bacterial spot", "advice": "Avoid excessive humidity and use copper-based fungicides."}
}

# ========== Load Model ==========
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../runs/tomato_demo/best.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, meta = load_model(MODEL_PATH, device)
idx_to_class = {v: k for k, v in meta["class_to_idx"].items()}
img_size = meta["img_size"]
mean, std = meta["mean"], meta["std"]

# 图像预处理
tf = A.Compose([
    A.LongestMaxSize(max_size=img_size),
    A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0)),
    A.CenterCrop(img_size, img_size),
    A.Normalize(mean=mean, std=std),
    ToTensorV2()
])

# ========== FastAPI App ==========
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化 OpenAI 客户端（从 .env 中读取）
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = np.array(img)

    x = tf(image=img)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    top_idx = probs.argmax()
    class_name = idx_to_class[top_idx]
    confidence = float(probs[top_idx])

    # 匹配疾病信息
    mapped = disease_advice_map.get(class_name, {"name": class_name, "advice": "No advice available."})
    disease = mapped["name"]
    base_advice = mapped["advice"]

    # ========== 调用 ChatGPT 生成自然语言建议 ==========
    try:
        prompt = (
            f"You are an expert in plant pathology. "
            f"The AI model detected **{disease}** in a tomato leaf with confidence {confidence:.2%}. "
            f"Base advice: {base_advice}\n\n"
            f"Please explain this diagnosis in clear, natural English, "
            f"and provide 2–3 specific, practical recommendations for tomato farmers."
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",  # 可换成 "gpt-4o" 以获得更高质量回答
            messages=[
                {"role": "system", "content": "You are a helpful agricultural assistant specialized in tomato disease diagnosis."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=250,
        )

        llm_reply = response.choices[0].message.content.strip()
    except Exception as e:
        llm_reply = f"(⚠️ ChatGPT integration failed: {e})"

    return {
        "disease": disease,
        "confidence": confidence,
        "advice": base_advice,
        "llm_response": llm_reply
    }
