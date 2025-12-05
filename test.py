import json
from grape_cls import predict_image

ckpt = r"F:\grape_ill\runs\tomato_demo\best.pt"
test_img = r"F:\grape_ill\data\Tomato_Late_blight\00ce4c63-9913-4b16-898c-29f99acf0dc3___RS_Late.B 4982.jpg"

# 预测
res = predict_image(ckpt, test_img, topk=1)
label, prob = res[0]

# 读取建议
with open("advice.json", "r", encoding="utf-8") as f:
    advice_db = json.load(f)

print(f"预测类别: {label} (置信度 {prob:.2f})")
print("建议:", advice_db.get(label, "暂无建议，请咨询专家。"))
