from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
from PIL import Image
import io

# Import your architecture
from src.model import Model 
from src.dataset import val_transforms

app = FastAPI()

# VERY IMPORTANT: This allows your frontend to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows requests from any frontend URL
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model on startup
print("Loading model...")
device = torch.device('cpu') # Hugging Face free tier uses CPU
model = Model(use_ensemble=True)
model.load_state_dict(torch.load('best.pt', map_location=device))
model.to(device)
model.eval()
print("Model loaded successfully!")

@app.get("/")
def read_root():
    return {"status": "API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and convert the image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Preprocess using your existing transforms
        input_tensor = val_transforms(image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.sigmoid(output).item()
            
        # Format the result
        label = "Fake" if prediction > 0.5 else "Real"
        confidence = prediction if label == "Fake" else 1 - prediction
        
        return {
            "label": label, 
            "confidence": round(confidence * 100, 2),
            "success": True
        }
    except Exception as e:
        return {"success": False, "error": str(e)}