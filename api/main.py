import utils.transport
import utils.session
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import io, os
import google.generativeai as genai
from dotenv import load_dotenv

# ==============================
# Load env (Gemini API key)
# ==============================
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ==============================
# Load model
# ==============================
MODEL_DIR = os.path.join("models", "mobilenetv2_kaggle")
model = AutoModelForImageClassification.from_pretrained(MODEL_DIR)
processor = AutoImageProcessor.from_pretrained(MODEL_DIR)

# ==============================
# FastAPI setup
# ==============================
app = FastAPI(title="🌱 Crop Disease Detection API + Gemini")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# Helper: Ask Gemini
# ==============================
def get_solution_from_gemini(crop: str, disease: str, confidence: str):
    prompt = f"""
    A farmer has uploaded a picture of a {crop} leaf.
    The AI model detected: {disease} (confidence {confidence}).

    Please explain in simple farmer-friendly terms:
    1. What this disease is.
    2. Why it occurs.
    3. Immediate treatment steps (chemical and organic).
    4. Preventive measures.
    5. Translate the advice also into Hindi.

    Make it short, clear, and actionable.
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text

# ==============================
# Routes
# ==============================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            preds = outputs.logits.softmax(dim=-1)

        pred_class = preds.argmax(-1).item()
        confidence = preds[0][pred_class].item()
        label = model.config.id2label[pred_class]

        # Extract crop name (simple split)
        crop = label.split("___")[0] if "___" in label else "Unknown crop"

        # ✅ Ask Gemini for full solution
        solution = get_solution_from_gemini(crop, label, f"{confidence:.2%}")

        return JSONResponse({
            "filename": file.filename,
            "prediction": label,
            "confidence": f"{confidence:.2%}",
            "gemini_solution": solution
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/")
async def root():
    return {"message": "🌱 Crop Disease Detection + Gemini AI is running!"}
