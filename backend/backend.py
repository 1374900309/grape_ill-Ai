import io, json
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from grape_cls import predict_image

app = FastAPI()

# 模型与建议库路径
CKPT_PATH = r"F:\grape_ill\runs\tomato_demo\best.pt"
ADVICE_PATH = r"F:\grape_ill\advice.json"

with open(ADVICE_PATH, "r", encoding="utf-8") as f:
    ADVICE_DB = json.load(f)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 保存临时文件
    img_bytes = await file.read()
    with open("tmp.jpg", "wb") as f:
        f.write(img_bytes)

    # 预测
    res = predict_image(CKPT_PATH, "tmp.jpg", topk=1)
    label, prob = res[0]
    advice = ADVICE_DB.get(label, "暂无建议，请咨询专家。")

    return JSONResponse({
        "label": label,
        "confidence": prob,
        "advice": advice
    })
