🌱 Plant Disease Detection System (Grape Ill) – Documentation
📘 1. Project Overview

This project is an AI-powered plant disease detection system.
Users can upload or take a photo of a leaf, and the system will automatically:

Classify the plant disease category

Output confidence scores

Give a short diagnosis description

Provide suggestions for treatment (optional)

The project contains:

A backend (Python / FastAPI / PyTorch) for model inference

A frontend (React or HTML/JS) for user interaction

A trained model (best.pt) for grape leaf disease classification

🌿 2. Main Features

✔ Upload image to detect plant diseases
✔ High-accuracy deep learning model (EfficientNet / ConvNeXt / YOLO-based classifier)
✔ API endpoint for image classification
✔ Real-time prediction
✔ Support for future expansion (multi-crop datasets)

⚙️ 3. Installation Guide
🔧 3.1. Clone Repository
git clone https://github.com/1374900309/grape_ill-Ai.git
cd grape_ill-Ai

🖥️ 4. Backend Setup (FastAPI + PyTorch)
📌 Step 1: Create Environment
cd backend
pip install -r requirements.txt
fastapi
uvicorn
torch
pillow
opencv-python
numpy
pip install fastapi uvicorn torch pillow opencv-python numpy

📌 Step 2: Start Backend Server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

http://localhost:8000/docs
🔍 5. API Usage
🔹 POST /predict
Request

Type: multipart/form-data

Parameter: file (image)

Example (Python)
import requests

url = "http://localhost:8000/predict"
files = {"file": open("test_leaf.jpg", "rb")}

res = requests.post(url, files=files)
print(res.json())

Response Example
{
  "class": "Leaf Blight",
  "confidence": 0.984,
  "description": "Symptoms include yellow-brown lesions expanding along the leaf veins."
}

🌐 6. Frontend Usage
Step 1: Install dependencies
cd frontend
npm install

Step 2: Start frontend
npm start
http://localhost:3000

📝 7. How the Model Works

Your best.pt model is trained on a dataset of plant leaf images. It uses:

Pretrained CNN / Transformer

Image augmentation

Fine-tuning to classify disease types

When user uploads an image：

The backend loads model → preprocesses the image

Model outputs predicted class + probabilities

Backend returns JSON response

Frontend displays diagnosis results
