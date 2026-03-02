import os
import io
import time
import base64
import numpy as np
import cv2

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from PIL import Image
from flask import Flask, request, jsonify, render_template

import tensorflow as tf
import torch
import torch.nn.functional as F
from torchvision import transforms
import timm
import shap
from lime import lime_image
from skimage.segmentation import mark_boundaries

# Stop TensorFlow from hoarding all GPU memory so PyTorch can breathe
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

app = Flask(__name__)

CLASS_NAMES = sorted([
    'dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis', 
    'normal-cecum', 'normal-pylorus', 'normal-z-line', 
    'polyps', 'ulcerative-colitis'
])

def get_health_status(class_name):
    if class_name in ['normal-cecum', 'normal-pylorus', 'normal-z-line']: return 'healthy'
    return 'abnormal'

def fig_to_base64(fig, dpi=300):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1, dpi=dpi)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return encoded

def preprocess_image_tf(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((224, 224))
    raw_img = np.array(img) / 255.0
    return np.expand_dims(raw_img, axis=0), raw_img

pytorch_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
def preprocess_image_pt(image_bytes):
    img_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor_img = pytorch_transform(img_pil).unsqueeze(0).to(DEVICE)
    raw_img = np.array(img_pil.resize((224, 224))) / 255.0
    return tensor_img, raw_img

# =========================================================
# [ PASTE SECTION 1: CUSTOM CNN HERE ]
# =========================================================
try:
    cnn_model = tf.keras.models.load_model('cnn_model.h5')
    print("✅ Custom CNN loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load CNN: {e}")

def generate_cnn_gradcam(tensor_img, raw_img, pred_index):
    try:
        target_layer = [l for l in reversed(cnn_model.layers) if isinstance(l, tf.keras.layers.Conv2D)][0]
        grad_model = tf.keras.models.Model(inputs=cnn_model.inputs, outputs=[target_layer.output, cnn_model.output])
        tensor_img_tf = tf.convert_to_tensor(tensor_img, dtype=tf.float32)
        with tf.GradientTape() as tape:
            outputs = grad_model(tensor_img_tf)
            conv_output, preds = outputs[0], outputs[1]
            if isinstance(preds, list): preds = preds[0]
            if isinstance(conv_output, list): conv_output = conv_output[0]
            class_channel = preds[:, pred_index]
        grads = tape.gradient(class_channel, conv_output)
        weights = tf.reduce_mean(grads, axis=(1, 2))
        cam = tf.reduce_sum(weights[:, None, None, :] * conv_output, axis=-1)
        heatmap = cv2.resize(np.maximum(cam[0].numpy(), 0) / (np.max(cam[0].numpy()) + 1e-8), (224, 224))
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(raw_img)
        im1 = ax.imshow(heatmap, cmap="jet", alpha=0.4) 
        ax.axis('off')
        cbar = plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Activation Strength (Grad-CAM)', rotation=270, labelpad=20)
        return fig_to_base64(fig)
    except Exception as e:
        print(f"❌ CNN Grad-CAM error: {e}")
        return None

def generate_cnn_shap(tensor_img, raw_img, pred_class):
    """
    CNN SHAP using Integrated Gradients (Keras 3 safe).
    Falls back to Gradient x Input if needed.
    Keeps your 3-panel layout unchanged.
    """
    try:
        # --------------------------------------------------
        # 1️⃣  Baseline (black image)
        # --------------------------------------------------
        baseline = np.zeros_like(tensor_img[0]).astype(np.float32)
        image = tensor_img[0].astype(np.float32)

        # --------------------------------------------------
        # 2️⃣  Compute gradient function (fresh tape each call)
        # --------------------------------------------------
        def compute_gradient(img_batch, class_idx):
            x = tf.Variable(img_batch, dtype=tf.float32)
            with tf.GradientTape() as tape:
                # Targeted to cnn_model specifically
                preds = cnn_model(x, training=False)
                target = preds[:, class_idx]
            grad = tape.gradient(target, x)
            if grad is None:
                return None
            return grad.numpy()

        # --------------------------------------------------
        # 3️⃣  Integrated Gradients
        # --------------------------------------------------
        n_steps = 40
        alphas = np.linspace(0, 1, n_steps).astype(np.float32)
        accumulated_grad = np.zeros_like(image, dtype=np.float32)

        for alpha in alphas:
            interpolated = baseline + alpha * (image - baseline)
            interpolated = interpolated[np.newaxis, ...]
            grad = compute_gradient(interpolated, pred_class)
            if grad is not None:
                accumulated_grad += grad[0]

        avg_grad = accumulated_grad / n_steps
        ig = (image - baseline) * avg_grad

        # --------------------------------------------------
        # 4️⃣  If IG failed → Gradient × Input fallback
        # --------------------------------------------------
        if np.mean(np.abs(ig)) < 1e-10:
            grad = compute_gradient(tensor_img, pred_class)
            if grad is not None:
                ig = grad[0] * image
            else:
                ig = np.zeros_like(image)

        # --------------------------------------------------
        # 5️⃣  Convert to spatial map
        # --------------------------------------------------
        shap_map = np.sum(ig, axis=-1)

        # Smooth slightly
        shap_map = cv2.GaussianBlur(shap_map, (9, 9), 0)

        # Normalize for visualization
        vmax = np.percentile(np.abs(shap_map), 98)
        if vmax < 1e-8:
            vmax = np.max(np.abs(shap_map)) + 1e-8
        vmin = -vmax

        # --------------------------------------------------
        # 6️⃣  Plot (same layout you already use)
        # --------------------------------------------------
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Panel 1: Original
        axes[0].imshow(raw_img)
        axes[0].axis('off')
        axes[0].set_title("Original Image", fontsize=16)

        # Panel 2: Overlay
        axes[1].imshow(raw_img)
        axes[1].imshow(shap_map, cmap="RdBu_r", vmin=vmin, vmax=vmax, alpha=0.5)
        axes[1].axis('off')
        axes[1].set_title("SHAP Overlay (Integrated Gradients)", fontsize=16)

        # Panel 3: Contribution Map
        im = axes[2].imshow(shap_map, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[2].axis('off')
        axes[2].set_title("SHAP Contribution Map\n(Red = Supports | Blue = Opposes)", fontsize=16)

        plt.tight_layout()
        return fig_to_base64(fig, dpi=400)

    except Exception as e:
        print(f"❌ Real SHAP error: {e}")
        plt.close('all')
        return None

# =========================================================
# [ PASTE SECTION 2: SWIN TRANSFORMER HERE ]
# =========================================================
# =========================================================
# SWIN TRANSFORMER (PYTORCH)
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict) or not state_dict: return state_dict
    return {k[len('module.'):]: v for k, v in state_dict.items()} if all(isinstance(k, str) and k.startswith('module.') for k in state_dict.keys()) else state_dict

try:
    swin_model = timm.create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=8)
    checkpoint = torch.load('swin_transformer.pth', map_location=DEVICE, weights_only=False)
    state_dict = checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    swin_model.load_state_dict(_strip_module_prefix(state_dict))
    swin_model.to(DEVICE)
    swin_model.eval()
    print(f"✅ Swin Transformer loaded successfully on {DEVICE}!")
except Exception as e:
    print(f"❌ Error loading Swin Transformer weights: {e}")

class GradCAMPP:
    def __init__(self, model, layer):
        self.model = model; self.layer = layer; self.grad = None; self.act = None
        self.layer.register_forward_hook(self.save_act)
        try: self.layer.register_full_backward_hook(self.save_grad)
        except AttributeError: self.layer.register_backward_hook(self.save_grad)
    def save_act(self, m, i, o): self.act = o
    def save_grad(self, m, gi, go): self.grad = go[0]
    def generate(self, x, cls):
        out = self.model(x)
        self.model.zero_grad()
        out[0, cls].backward(retain_graph=True)
        g, a = self.grad, self.act
        if g.dim() == 4: cam = (g.mean(dim=(2, 3), keepdim=True) * a).sum(dim=1)
        elif g.dim() == 3: 
            cam = (g.mean(dim=1, keepdim=True) * a).sum(dim=2)
            cam = cam.view(1, 1, int(np.sqrt(cam.shape[-1])), int(np.sqrt(cam.shape[-1])))
        else: cam = g.abs().mean(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze()
        return cv2.resize(((cam - cam.min()) / (cam.max() - cam.min() + 1e-8)).detach().cpu().numpy(), (224, 224))

def generate_swin_gradcam(tensor_img, raw_img, pred_class):
    try:
        gradcam_map = GradCAMPP(swin_model, swin_model.layers[-1].blocks[-1].norm2).generate(tensor_img, pred_class)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(raw_img)
        im1 = ax.imshow(gradcam_map, cmap="jet", alpha=0.5)
        ax.axis('off')
        cbar = plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Activation Strength (Grad-CAM++)', rotation=270, labelpad=20)
        return fig_to_base64(fig)
    except Exception as e:
        print(f"❌ Swin Grad-CAM error: {e}")
        return None

def generate_swin_shap(tensor_img, raw_img, pred_class): 
    try:
        input_img = tensor_img.detach().clone()
        _mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
        _std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)
        baseline = (torch.zeros(1, 3, 224, 224, device=DEVICE) - _mean) / _std
        grads = []
        for i in range(51):
            x = (baseline + (i / 50) * (input_img - baseline)).detach().requires_grad_(True)
            out = swin_model(x)
            swin_model.zero_grad()
            out[0, pred_class].backward()
            grads.append(x.grad.detach().clone())
        ig_map = cv2.GaussianBlur(((input_img - baseline) * torch.mean(torch.stack(grads), dim=0)).sum(dim=1).squeeze().cpu().numpy(), (11, 11), sigmaX=3)
        if np.max(np.abs(ig_map)) > 1e-8: ig_map = ig_map / np.max(np.abs(ig_map))
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(raw_img)
        im = ax.imshow(ig_map, cmap="seismic", alpha=0.7, vmin=-1, vmax=1)
        ax.axis('off')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Attribution (Negative \u2192 Positive)', rotation=270, labelpad=20)
        return fig_to_base64(fig)
    except Exception as e:
        print(f"❌ Swin SHAP error: {e}")
        return None
# =========================================================
# [ PASTE SECTION 3: RESNET50 HERE ]
# =========================================================

# =========================================================
# RESNET50 (TENSORFLOW)
# =========================================================
try:
    resnet_model = tf.keras.models.load_model('resnet_model.h5')
    print("✅ ResNet50 loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load ResNet: {e}")

def generate_resnet_gradcam(tensor_img, raw_img, pred_index):
    try:
        target_layer = [l for l in reversed(resnet_model.layers) if isinstance(l, tf.keras.layers.Conv2D)][0]
        grad_model = tf.keras.models.Model(inputs=resnet_model.inputs, outputs=[target_layer.output, resnet_model.output])
        tensor_img_tf = tf.convert_to_tensor(tensor_img, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            outputs = grad_model(tensor_img_tf)
            conv_output, preds = outputs[0], outputs[1]
            if isinstance(preds, list): preds = preds[0]
            if isinstance(conv_output, list): conv_output = conv_output[0]
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, conv_output)
        weights = tf.reduce_mean(grads, axis=(1, 2))
        cam = tf.reduce_sum(weights[:, None, None, :] * conv_output, axis=-1)

        heatmap = cv2.resize(np.maximum(cam[0].numpy(), 0) / (np.max(cam[0].numpy()) + 1e-8), (224, 224))
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(raw_img)
        im1 = ax.imshow(heatmap, cmap="jet", alpha=0.4)
        ax.axis('off')
        cbar = plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Activation Strength (Grad-CAM)', rotation=270, labelpad=20)
        return fig_to_base64(fig)
    except Exception as e:
        print(f"❌ ResNet Grad-CAM error: {e}")
        return None

def _gradient_x_input_fallback(tensor_img, raw_img, pred_class):
    try:
        inp = tf.Variable(tf.convert_to_tensor(tensor_img, dtype=tf.float32))
        with tf.GradientTape() as tape:
            tape.watch(inp)
            score = resnet_model(inp, training=False)[:, pred_class]
        grads = tape.gradient(score, inp).numpy()[0] 
        saliency = cv2.GaussianBlur(np.sum(grads * tensor_img[0], axis=-1), (25, 25), 0)
        
        vmax = np.percentile(np.abs(saliency), 99)
        if vmax < 1e-8: vmax = np.max(np.abs(saliency)) + 1e-8

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(raw_img); axes[0].axis('off'); axes[0].set_title("Original Image", fontsize=16)
        axes[1].imshow(raw_img); axes[1].imshow(saliency, cmap="RdBu_r", vmin=-vmax, vmax=vmax, alpha=0.45); axes[1].axis('off'); axes[1].set_title("Gradient * Input (Fallback)", fontsize=16)
        axes[2].imshow(saliency, cmap="RdBu_r", vmin=-vmax, vmax=vmax); axes[2].axis('off'); axes[2].set_title("Attribution Map", fontsize=16)
        plt.tight_layout()
        return fig_to_base64(fig, dpi=400)
    except Exception as e:
        print(f"❌ ResNet Fallback error: {e}")
        return None

def generate_resnet_shap(tensor_img, raw_img, pred_class):
    try:
        background = np.clip(raw_img + np.random.normal(0, 0.015, (20, 224, 224, 3)), 0, 1).astype(np.float32)
        try:
            single_output = tf.keras.models.Model(inputs=resnet_model.inputs, outputs=resnet_model.output[:, pred_class])
        except Exception:
            def single_output(x): return resnet_model.predict(x)[:, pred_class]

        explainer = shap.GradientExplainer(single_output, background)
        shap_values = explainer.shap_values(tensor_img)
        sv = shap_values[0] if isinstance(shap_values, list) else shap_values

        if sv.ndim == 4: class_shap = sv[0]
        elif sv.ndim == 3: class_shap = sv
        else: return _gradient_x_input_fallback(tensor_img, raw_img, pred_class)

        shap_map = cv2.GaussianBlur(np.sum(class_shap, axis=-1), (25, 25), 0)
        vmax = np.percentile(np.abs(shap_map), 99)
        if vmax < 1e-8: vmax = np.max(np.abs(shap_map)) + 1e-8

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(raw_img); axes[0].axis('off'); axes[0].set_title("Original Image", fontsize=16)
        axes[1].imshow(raw_img); axes[1].imshow(shap_map, cmap="RdBu_r", vmin=-vmax, vmax=vmax, alpha=0.45); axes[1].axis('off'); axes[1].set_title("ResNet SHAP Overlay", fontsize=16)
        axes[2].imshow(shap_map, cmap="RdBu_r", vmin=-vmax, vmax=vmax); axes[2].axis('off'); axes[2].set_title("SHAP Attribution Map", fontsize=16)
        plt.tight_layout()
        return fig_to_base64(fig, dpi=400)
    except Exception as e:
        print(f"❌ ResNet SHAP error: {e}")
        return _gradient_x_input_fallback(tensor_img, raw_img, pred_class)
# =========================================================
# UNIVERSAL LIME 
# =========================================================
def generate_lime(raw_img, pred_class, model_type):
    try:
        if model_type == 'swin' and 'swin_model' in globals():
            def predict_fn(numpy_images):
                DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                tensors = []
                for img in numpy_images:
                    if img.max() > 1.5: img = img / 255.0
                    img_t = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(torch.tensor(img, dtype=torch.float32).permute(2, 0, 1))
                    tensors.append(img_t)
                with torch.no_grad():
                    return F.softmax(swin_model(torch.stack(tensors).to(DEVICE)), dim=1).cpu().numpy()
        elif model_type == 'resnet' and 'resnet_model' in globals():
            def predict_fn(images): return resnet_model.predict(images, verbose=0)
        elif 'cnn_model' in globals():
            def predict_fn(images): return cnn_model.predict(images, verbose=0)
        else:
            return None

        explainer_lime = lime_image.LimeImageExplainer()
        explanation = explainer_lime.explain_instance(
            raw_img.astype('double'), predict_fn, top_labels=3, hide_color=0, num_samples=300 if model_type == 'swin' else 1000 
        )
        temp, mask = explanation.get_image_and_mask(pred_class, positive_only=True, num_features=5 if model_type == 'swin' else 8, hide_rest=False)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(mark_boundaries(temp, mask))
        ax.axis('off')
        return fig_to_base64(fig)
    except Exception as e:
        print(f"❌ LIME error: {e}")
        return None

# =========================================================
# FLASK ROUTES
# =========================================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/viewer_3d.html')
def viewer_3d():
    return render_template('viewer_3d.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files: return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    model_type = request.form.get('model_type', 'cnn').lower()
    
    try:
        start_time = time.time()
        image_bytes = file.read()

        if model_type == 'swin':
            if 'swin_model' not in globals(): return jsonify({'error': 'Swin Transformer code not added yet!'}), 500
            tensor_img, raw_img = preprocess_image_pt(image_bytes)
            with torch.no_grad():
                predictions = F.softmax(swin_model(tensor_img), dim=1)[0].cpu().numpy()
            pred_class = int(np.argmax(predictions)) 
            gradcam_b64 = generate_swin_gradcam(tensor_img, raw_img, pred_class)
            shap_b64 = generate_swin_shap(tensor_img, raw_img, pred_class)
            model_display_name = "Swin Transformer"

        elif model_type == 'resnet':
            if 'resnet_model' not in globals(): return jsonify({'error': 'ResNet code not added yet!'}), 500
            tensor_img, raw_img = preprocess_image_tf(image_bytes)
            predictions = resnet_model.predict(tensor_img, verbose=0)[0]
            pred_class = int(np.argmax(predictions))
            gradcam_b64 = generate_resnet_gradcam(tensor_img, raw_img, pred_class)
            shap_b64 = generate_resnet_shap(tensor_img, raw_img, pred_class)
            model_display_name = "ResNet50"

        else: # CNN Default
            if 'cnn_model' not in globals(): return jsonify({'error': 'CNN code not added yet!'}), 500
            tensor_img, raw_img = preprocess_image_tf(image_bytes)
            predictions = cnn_model.predict(tensor_img, verbose=0)[0]
            pred_class = int(np.argmax(predictions))
            gradcam_b64 = generate_cnn_gradcam(tensor_img, raw_img, pred_class)
            shap_b64 = generate_cnn_shap(tensor_img, raw_img, pred_class)
            model_display_name = "Custom CNN"

        lime_b64 = generate_lime(raw_img, pred_class, model_type)
        inference_time = f"{round((time.time() - start_time) * 1000)} ms"
        top_class_raw = CLASS_NAMES[pred_class]

        all_predictions = sorted([
            {'class_name': CLASS_NAMES[i].replace('-', ' ').title(), 'confidence': round(float(conf) * 100, 2)}
            for i, conf in enumerate(predictions)
        ], key=lambda x: x['confidence'], reverse=True)

        return jsonify({
            'top_prediction': top_class_raw.replace('-', ' ').title(),
            'top_confidence': f"{round(float(predictions[pred_class]) * 100, 2)}%",
            'top_prediction_raw': top_class_raw,
            'health_status': get_health_status(top_class_raw),
            'all_predictions': all_predictions,
            'model_used': model_display_name,
            'inference_time': inference_time,
            'gradcam': gradcam_b64,
            'lime': lime_b64,
            'shap': shap_b64
        })

    except Exception as e:
        print(f"Server Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5050)