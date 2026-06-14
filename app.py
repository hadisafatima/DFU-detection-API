# from fastapi import FastAPI, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# from PIL import Image
# import numpy as np
# import tensorflow as tf
# import io

# app = FastAPI()

# # =========================
# # CORS (for Flutter / mobile)
# # =========================
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # =========================
# # LOAD MODEL
# # =========================
# model = tf.keras.models.load_model("dfu_model.h5")

# IMG_SIZE = 224

# # ⚠️ IMPORTANT: FIX LABEL ORDER HERE based on training
# # Check: train_gen.class_indices
# class_names = ["Ulcer", "Normal"]   # CHANGE THIS if needed

# # =========================
# # HOME ROUTE
# # =========================
# @app.get("/")
# def home():
#     return {"message": "DFU CNN API is running 🚀"}

# # =========================
# # PREDICTION ROUTE
# # =========================
# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):

#     # Read image
#     contents = await file.read()
#     image = Image.open(io.BytesIO(contents)).convert("RGB")

#     # Resize (MUST match training)
#     image = image.resize((IMG_SIZE, IMG_SIZE))

#     # Convert to array
#     img_array = np.array(image).astype(np.float32)

#     # Normalize (MOST COMMON SETUP)
#     img_array = img_array / 255.0

#     # Expand dimensions for batch
#     img_array = np.expand_dims(img_array, axis=0)

#     # Predict
#     prediction = model.predict(img_array)[0]

#     # =========================
#     # CASE 1: Sigmoid output (binary)
#     # =========================
#     if len(prediction) == 1:
#         prob = float(prediction[0])

#         # safer threshold logic
#         predicted_class = 1 if prob > 0.5 else 0

#         return {
#             "prediction": class_names[predicted_class],
#             "confidence": 1- prob,
#             "normal_probability": prob,
#             "ulcer_probability": 1-  prob
#         }

#     # =========================
#     # CASE 2: Softmax output (2 classes)
#     # =========================
#     else:
#         probs = prediction.tolist()
#         predicted_class = int(np.argmax(probs))

#         return {
#             "prediction": class_names[predicted_class],
#             "confidence": float(max(probs)),
#             "probabilities": {
#                 class_names[i]: float(probs[i]) for i in range(len(probs))
#             }
#         }





from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import os
import gdown

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# DOWNLOAD MODEL FROM DRIVE
# =========================
MODEL_PATH = "dfu_model.h5"
GDRIVE_FILE_ID = "185gd4jyOOH4kaUlD_2FfHSD9pDqew-hk"

if not os.path.exists(MODEL_PATH):
    print("Downloading latest model from Google Drive...")
    gdown.download(
        f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}",
        MODEL_PATH,
        quiet=False
    )

model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

IMG_SIZE = 224
class_names = ["Ulcer", "Normal"]

# =========================
# ROUTES
# =========================
@app.get("/")
def home():
    return {"message": "DFU CNN API is running 🚀"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    img_array = np.array(image).astype(np.float32)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]

    if len(prediction) == 1:
        prob = float(prediction[0])
        predicted_class = 1 if prob > 0.5 else 0
        return {
            "prediction": class_names[predicted_class],
            "confidence": 1 - prob,
            "ulcer_probability": 1 - prob,
            "normal_probability": prob
        }
    else:
        probs = prediction.tolist()
        predicted_class = int(np.argmax(probs))
        return {
            "prediction": class_names[predicted_class],
            "confidence": float(max(probs)),
            "probabilities": {
                class_names[i]: float(probs[i]) for i in range(len(probs))
            }
        }