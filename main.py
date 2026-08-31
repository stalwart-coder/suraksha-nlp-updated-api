from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

app = FastAPI(title="SurakshaNLP API")

print("Loading model...")
MODEL_ID = "Bipin-Pal/suraksha-nlp-muril"
HF_TOKEN = os.environ.get("HF_TOKEN", None)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, token=HF_TOKEN)
model.eval()
print("✅ Model loaded!")

class ScanRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(req: ScanRequest):
    inputs = tokenizer(
        req.text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.softmax(outputs.logits, dim=1)
    is_scam = probs[0][1].item() > 0.5
    score = round(probs[0][1].item() * 100, 2)
    
    return {
        "risk_level": "HIGH" if (is_scam and score > 80) else "SUSPICIOUS" if (is_scam and score > 50) else "SAFE",
        "confidence_score": score,
        "scam_category": "SCAM_DETECTED" if is_scam else "SAFE",
        "manipulation_tactics": ["Urgency", "Fear"] if is_scam else [],
        "model_version": "SurakshaNLP-MuRIL-v1.0"
    }

@app.get("/health")
def health():
    return {"status": "running", "model": MODEL_ID}

@app.get("/")
def root():
    return {"message": "SurakshaNLP API is live!"}

@app.get("/")
def root():
    return {"message": "SurakshaNLP API is live!"}
