import os
import io
import time
import base64
import numpy as np
import cv2

# Fix for the "Matplotlib GUI" thread warning
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from PIL import Image
from flask import Flask, request, jsonify, render_template

# PyTorch & Timm
import torch
import torch.nn.functional as F
from torchvision import transforms
import timm

# XAI Libraries
from lime import lime_image
from skimage.segmentation import mark_boundaries

app = Flask(__name__)

# 1. Device and Model Initialization
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'swin_transformer.pth'
NUM_CLASSES = 8

print(f"Loading Swin Transformer on {DEVICE}...")
model = timm.create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=NUM_CLASSES)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print("Swin Transformer loaded successfully.")
except Exception as e:
    print(f"Error loading weights: {e}")

CLASS_NAMES = sorted([
    'dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis', 
    'normal-cecum', 'normal-pylorus', 'normal-z-line', 
    'polyps', 'ulcerative-colitis'
])

def get_health_status(class_name):
    if class_name in ['normal-cecum', 'normal-pylorus', 'normal-z-line']:
        return 'healthy'
    return 'abnormal'

# PyTorch Normalization
pytorch_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image_bytes):
    img_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    tensor_img = pytorch_transform(img_pil).unsqueeze(0).to(DEVICE)
    raw_img = np.array(img_pil.resize((224, 224))) / 255.0
    return tensor_img, raw_img

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1, dpi=300)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return encoded

# Wrapper specifically for LIME evaluation
def predict_fn_numpy(numpy_images):
    tensors = []
    for img in numpy_images:
        # Guarantee values are in 0-1 range to avoid Matplotlib clipping errors
        if img.max() > 1.5:
            img = img / 255.0
        img_t = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        img_t = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_t)
        tensors.append(img_t)
        
    batch = torch.stack(tensors).to(DEVICE)
    with torch.no_grad():
        outputs = model(batch)
        probs = F.softmax(outputs, dim=1)
    return probs.cpu().numpy()

# =========================================================
# 1. Grad-CAM++ (Exact Swin Implementation)
# =========================================================
class GradCAMPP:
    def __init__(self, model, layer):
        self.model = model
        self.layer = layer
        self.grad = None
        self.act = None

        self.layer.register_forward_hook(self.save_act)
        try:
            self.layer.register_full_backward_hook(self.save_grad)
        except AttributeError:
            self.layer.register_backward_hook(self.save_grad)

    def save_act(self, m, i, o):
        self.act = o

    def save_grad(self, m, gi, go):
        self.grad = go[0]

    def generate(self, x, cls):
        out = self.model(x)
        self.model.zero_grad()
        out[0, cls].backward(retain_graph=True)

        g = self.grad
        a = self.act

        if g.dim() == 4:
            weights = g.mean(dim=(2, 3), keepdim=True)
            cam = (weights * a).sum(dim=1)
        elif g.dim() == 3: # Swin Transformer specific token mapping
            weights = g.mean(dim=1, keepdim=True)
            cam = (weights * a).sum(dim=2)
            side = int(np.sqrt(cam.shape[-1]))
            cam = cam.view(1, 1, side, side)
        else:
            cam = g.abs().mean(dim=1, keepdim=True)

        cam = torch.relu(cam).squeeze()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = cv2.resize(cam.detach().cpu().numpy(), (224, 224))
        return cam

def generate_swin_gradcam(tensor_img, raw_img, pred_class):
    try:
        # Targets norm2 of the final block perfectly
        target_layer = model.layers[-1].blocks[-1].norm2
        cam_gen = GradCAMPP(model, target_layer)
        gradcam_map = cam_gen.generate(tensor_img, pred_class)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(raw_img)
        
        # Uses standard Notebook 'jet' colormap with alpha
        im1 = ax.imshow(gradcam_map, cmap="jet", alpha=0.5)
        ax.axis('off')
        
        cbar = plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Activation Strength (Grad-CAM++)', rotation=270, labelpad=20)
        
        return fig_to_base64(fig)
    except Exception as e:
        print(f"❌ Grad-CAM++ error: {e}")
        return None

# =========================================================
# 2. LIME (Original Working Implementation)
# =========================================================
def generate_lime(raw_img):
    try:
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            raw_img.astype('double'), 
            predict_fn_numpy, 
            top_labels=1, 
            hide_color=0, 
            num_samples=300
        )
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0], 
            positive_only=True, 
            num_features=5, 
            hide_rest=False
        )
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(mark_boundaries(temp, mask))
        ax.axis('off')
        return fig_to_base64(fig)
    except Exception as e:
        print(f"❌ LIME error: {e}")
        return None

# =========================================================
# 3. SHAP / Integrated Gradients (From Notebook Cell 5)
# =========================================================
def generate_notebook_shap(tensor_img, raw_img, pred_class):
    try:
        steps = 25
        baseline = torch.zeros_like(tensor_img).to(DEVICE)
        grads = []

        for i in range(steps + 1):
            alpha = i / steps
            x = baseline + alpha * (tensor_img - baseline)
            x.requires_grad_(True)

            out = model(x)
            score = out[0, pred_class]

            model.zero_grad()
            score.backward()

            grads.append(x.grad.detach())

        avg_grads = torch.mean(torch.stack(grads), dim=0)
        ig = (tensor_img - baseline) * avg_grads
        
        # Exact Notebook Math: absolute mean across channels, squeezed to 2D numpy array
        ig_map = ig.abs().mean(dim=1).squeeze().cpu().numpy()
        
        # Exact Notebook Normalization
        ig_map = (ig_map - ig_map.min()) / (ig_map.max() - ig_map.min() + 1e-8)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(raw_img)
        
        # Applies the precise Notebook "hot" mapping 
        im = ax.imshow(ig_map, cmap="hot", alpha=0.55, vmin=0, vmax=1)
        ax.axis('off')
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Attribution Strength (SHAP/IG)', rotation=270, labelpad=20)

        b64_img = fig_to_base64(fig)
        plt.close('all') 
        return b64_img
    except Exception as e:
        print(f"❌ SHAP error: {e}")
        plt.close('all')
        return None

# ---------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    try:
        start_time = time.time()
        
        image_bytes = file.read()
        tensor_img, raw_img = preprocess_image(image_bytes)
        
        with torch.no_grad():
            outputs = model(tensor_img)
            predictions = F.softmax(outputs, dim=1)[0].cpu().numpy()
            pred_class = int(np.argmax(predictions)) 
            
        end_time = time.time()
        inference_time = f"{round((end_time - start_time) * 1000)} ms"
        
        top_class_raw = CLASS_NAMES[pred_class]
        top_confidence = float(predictions[pred_class])
        top_class_display = top_class_raw.replace('-', ' ').title()
        
        all_predictions = [
            {'class_name': CLASS_NAMES[i].replace('-', ' ').title(), 'confidence': round(float(conf) * 100, 2)}
            for i, conf in enumerate(predictions)
        ]
        all_predictions = sorted(all_predictions, key=lambda x: x['confidence'], reverse=True)

        print("Generating Grad-CAM++...")
        gradcam_b64 = generate_swin_gradcam(tensor_img, raw_img, pred_class)
        
        print("Generating LIME...")
        lime_b64 = generate_lime(raw_img) 
        
        print("Generating SHAP (Integrated Gradients)...")
        shap_b64 = generate_notebook_shap(tensor_img, raw_img, pred_class)
        
        print("Analysis Complete! Sending back to frontend.")
        
        response = {
            'top_prediction': top_class_display,
            'top_confidence': f"{round(top_confidence * 100, 2)}%",
            'top_prediction_raw': top_class_raw,
            'health_status': get_health_status(top_class_raw),
            'all_predictions': all_predictions,
            'model_used': "Swin Transformer",
            'inference_time': inference_time,
            'gradcam': gradcam_b64,
            'lime': lime_b64,
            'shap': shap_b64
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Server Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5050)