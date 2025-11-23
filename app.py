import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import io
import os
import random
import torch.nn.functional as F
from fastapi.middleware.cors import CORSMiddleware

# Import model architectures and utility functions
from classifier_model import DeepConvClassifier 
from vaegan_model import Decoder, Encoder 
from transform_utils import preprocess_image_opencv, postprocess_generated_image

# --- Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4
LATENT_DIM = 64
ARTIFACTS_DIR = "artifacts"

VAE_WEIGHT_MAP = {
    0: os.path.join(ARTIFACTS_DIR, 'chirp_spectrogram_vaegan.pth'),
    1: os.path.join(ARTIFACTS_DIR, 'helix_spectrogram_vaegan.pth'),
    2: os.path.join(ARTIFACTS_DIR, 'light_mod_vaegan.pth'), 
    3: os.path.join(ARTIFACTS_DIR, 'wl_vaegan.pth'),
}

CLASS_NAMES = {
    0: "Chirp",
    1: "Helix",
    2: "Light Modulation",
    3: "Wandering Line",
}

# --- Global Model and Cache Initialization ---
CLASSIFIER_MODEL = None
VAE_CACHE = {} 

def load_models():
    """Loads the Classifier model (with metadata extraction)."""
    global CLASSIFIER_MODEL
    
    print(f"Initializing models on device: {device}")
    
    try:
        classifier_path = os.path.join(ARTIFACTS_DIR, 'final_cnn_glitch_classifier.pth')
        checkpoint = torch.load(classifier_path, map_location=device)
        
        # Checkpoint extraction logic
        state_dict = checkpoint.get('classifier_state_dict', checkpoint)
        
        CLASSIFIER_MODEL = DeepConvClassifier(num_classes=NUM_CLASSES).to(device)
        CLASSIFIER_MODEL.load_state_dict(state_dict)
        CLASSIFIER_MODEL.eval()
        print(f"✅ Classifier loaded successfully.")
    except Exception as e:
        print(f"❌ ERROR: Failed to load Classifier. Error: {e}")
        CLASSIFIER_MODEL = None 

def load_vaegan_for_class(class_index: int):
    """Loads both Encoder and Decoder for a specific class from one file."""
    if class_index in VAE_CACHE:
        return VAE_CACHE[class_index]
    
    path = VAE_WEIGHT_MAP.get(class_index)
    if not path or not os.path.exists(path):
        # NOTE: This error is crucial if the index doesn't exist in the VAE_WEIGHT_MAP
        raise FileNotFoundError(f"VAE weights for index {class_index} not mapped or file not found at {path}")
    
    checkpoint = torch.load(path, map_location=device)
    
    # --- 1. Load Encoder ---
    encoder = Encoder().to(device)
    encoder_state_dict = checkpoint.get('encoder') or checkpoint.get('encoder_state_dict')
    if encoder_state_dict:
        encoder.load_state_dict(encoder_state_dict)
    encoder.eval()

    # --- 2. Load Decoder ---
    decoder = Decoder().to(device)
    decoder_state_dict = checkpoint.get('decoder') or checkpoint.get('decoder_state_dict') or checkpoint.get('generator') or checkpoint
    decoder.load_state_dict(decoder_state_dict)
    decoder.eval()
    
    VAE_CACHE[class_index] = (encoder, decoder)
    print(f"✅ VAE-GAN (Encoder+Decoder) for class {class_index} loaded.")
    return encoder, decoder

# --- FastAPI Setup ---
app = FastAPI(title="Glitch Generator")

# --- CORS CONFIGURATION (Mandatory for Frontend) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)
# ---------------------------------------------------

load_models()

class GenerationResponse(BaseModel):
    predicted_class_index: int
    predicted_class_name: str
    generated_image_base64: str

@app.get("/")
def read_root():
    """Health check endpoint."""
    return {"status": "ok", "service": "Glitch Generator API", "models_loaded": CLASSIFIER_MODEL is not None}

@app.post("/generate", response_model=GenerationResponse)
async def classify_and_generate(file: UploadFile = File(...)):
    if CLASSIFIER_MODEL is None:
        raise HTTPException(status_code=500, detail="Classifier failed to initialize.")
    
    try:
        contents = await file.read()
        input_tensor = preprocess_image_opencv(contents).unsqueeze(0).to(device)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing error: {e}")

    # 1. Classification
    with torch.no_grad():
        output = CLASSIFIER_MODEL(input_tensor)
        probabilities = F.softmax(output, dim=1) 
        _, pred_idx_tensor = torch.max(output, 1)
        pred_idx = pred_idx_tensor.item()
        
        # NOTE: This logging is now crucial for final index confirmation
        print(f"DEBUG: Predicted Index: {pred_idx}")
        print(f"DEBUG: Mapping to file: {VAE_WEIGHT_MAP.get(pred_idx, 'Not Found')}")



    # 2. Load VAE-GAN (Encoder & Decoder)
    try:
        encoder, decoder = load_vaegan_for_class(pred_idx)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model loading error: {e}")

    # 3. Guided Generation
    with torch.no_grad():
        # TEMPORARY LOGGING IN app.py
        # Inside the with torch.no_grad() block for generation:
        mu, logvar = encoder(input_tensor) 
        print(f"DEBUG: Wandering Line MU Mean: {mu.mean().item():.5f}, Std: {mu.std().item():.5f}")

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) * 0.5
        z = mu + eps * std
        print(f"DEBUG: Wandering Line Z Mean: {z.mean().item():.5f}, Std: {z.std().item():.5f}")
        # If these values are all near zero, the VAE-GAN encoder is dead.

        generated_tensor = decoder(z)
        print(f"DEBUG: Decoder RAW Output Min: {generated_tensor.min().item():.5f}, Max: {generated_tensor.max().item():.5f}")

    # 4. Post-process
    base64_img = postprocess_generated_image(generated_tensor)
    
    return GenerationResponse(
        predicted_class_index=pred_idx,
        predicted_class_name=CLASS_NAMES.get(pred_idx, f"Class {pred_idx}"),
        generated_image_base64=base64_img
    )