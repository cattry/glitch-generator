import cv2
import numpy as np
import torch
import io
import base64
from PIL import Image # Only for the final output conversion

# --- Constants ---
IMG_SIZE = 256
MEAN = [0.5]
STD = [0.5]

# --- 1. Pre-processing (MATCHING newnormvs.py) ---

def preprocess_image_opencv(image_bytes):
    """
    Replicates the exact preprocessing from newnormvs.py using OpenCV.
    """
    # 1. Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    
    # 2. Decode image using OpenCV (Grayscale mode)
    # cv2.IMREAD_REDUCED_GRAYSCALE_2 was used in your script, but for general inputs
    # we should use standard grayscale loading to be safe, or resize manually.
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError("Could not decode image")

    # 3. Resize using INTER_AREA (The critical match!)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    
    # 4. Normalize to [0, 1] range (as done in newnormvs.py)
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # 5. OPTIONAL: Mimic the 'save as uint8' artifacting from training
    # Your training script saved the float image as a uint8 image file, 
    # which introduces quantization noise. We mimic that here.
    img_quantized = (img_normalized * 255).astype(np.uint8)
    img_final = img_quantized.astype(np.float32) / 255.0
    
    # 6. Convert to PyTorch Tensor
    # OpenCV is [H, W], PyTorch needs [C, H, W]
    tensor = torch.from_numpy(img_final).unsqueeze(0) 
    
    # 7. Apply Final Training Normalization ([0, 1] -> [-1, 1])
    # (X - 0.5) / 0.5
    tensor = (tensor - MEAN[0]) / STD[0]
    
    return tensor

# --- 2. Post-processing for Decoder Output ---

def postprocess_generated_image(tensor):
    """
    Robust post-processing using NumPy.
    """
    # 1. Denormalize
    image_tensor = (tensor.squeeze(0) * torch.tensor(STD, device=tensor.device).view(1, 1, 1)) + torch.tensor(MEAN, device=tensor.device).view(1, 1, 1)

    # 2. Convert to NumPy
    np_img = image_tensor.cpu().detach().numpy().squeeze()
    
    # 3. Sanitize and Clamp
    np_img = np.nan_to_num(np_img, nan=0.0, posinf=1.0, neginf=0.0)
    np_img = np.clip(np_img, 0.0, 1.0)
    
    # 4. Convert to Uint8
    np_img_uint8 = (np_img * 255.0).astype(np.uint8)

    # 5. Convert to PIL for saving (Pillow is fine for saving, just not loading/resizing)
    image_pil = Image.fromarray(np_img_uint8, mode='L')
    
    byte_arr = io.BytesIO()
    image_pil.save(byte_arr, format='PNG')
    base64_img = base64.b64encode(byte_arr.getvalue()).decode('utf-8')
    
    return f"data:image/png;base64,{base64_img}"