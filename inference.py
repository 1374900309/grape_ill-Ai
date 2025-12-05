import sys
from grape_cls import predict_image

if __name__ == "__main__":
    ckpt = r"F:\grape_ill\runs\tomato_demo\best.pt"
    test_img = r"F:\grape_ill\data\Tomato_Late_blight\00ce4c63-9913-4b16-898c-29f99acf0dc3___RS_Late.B 4982.jpg"

    res = predict_image(ckpt, test_img, topk=3)

    print("\n🔎 预测结果：")
    for label, prob in res:
        print(f"{label} -> {prob:.2f}")
