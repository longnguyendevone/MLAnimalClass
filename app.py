import gradio as gr
import torch
import torchvision.transforms as T
from PIL import Image
import json
import os

# Import our factory
from src.models import get_model_architecture

# 1. Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load Class Names
try:
    with open('checkpoints/classes.json', 'r') as f:
        CLASS_NAMES = json.load(f)
except FileNotFoundError:
    print("Warning: classes.json not found. Run main.py first.")
    CLASS_NAMES = [f"Class_{i}" for i in range(10)]

# 2. Helper to find available trained models
def get_available_models():
    # Scans checkpoints folder for .pth files
    if not os.path.exists("checkpoints"): return []
    files = [f.replace(".pth", "") for f in os.listdir("checkpoints") if f.endswith(".pth")]
    return files

model_cache = {}

def predict_animal(image, model_name):
    if image is None: return None
    
    # 3. Model Loading Logic
    if model_name not in model_cache:
        # Rebuild architecture
        # We assume num_classes matches the loaded JSON
        model = get_model_architecture(model_name, len(CLASS_NAMES), pretrained=False)
        
        # Load Weights
        weight_path = f"checkpoints/{model_name}.pth"
        try:
            model.load_state_dict(torch.load(weight_path, map_location=device))
        except Exception as e:
            return {f"Error loading {model_name}": str(e)}
            
        model.to(device)
        model.eval()
        model_cache[model_name] = model
    
    # 4. Inference
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_t = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model_cache[model_name](img_t)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
    
    return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

# 5. Launch UI
available_models = get_available_models()
if not available_models:
    available_models = ["No models found - Train first!"]

ui = gr.Interface(
    fn=predict_animal,
    inputs=[
        gr.Image(type="pil", label="Upload Animal Image"),
        gr.Dropdown(choices=available_models, label="Select Model", value=available_models[0])
    ],
    outputs=gr.Label(num_top_classes=3),
    title="Animals-10 Classifier",
    description=f"Running on {device.upper()}. Supports Custom CNNs, Transformers, and EfficientNets."
)

if __name__ == "__main__":
    # Clear proxies to ensure localhost is accessible
    for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
        if k in os.environ: del os.environ[k]

    ui.launch(server_name="127.0.0.1", share=False)