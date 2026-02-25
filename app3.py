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

# TensorFlow
import tensorflow as tf

# XAI Libraries
import shap
from lime import lime_image
from skimage.segmentation import mark_boundaries

app = Flask(__name__)

# =========================================================
# 1. STRICT MODEL INITIALIZATION
# =========================================================
MODEL_PATH = 'resnet_model.h5' 

print(f"Loading ResNet on TensorFlow {tf.__version__}...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"\n❌ CRITICAL ERROR: Could not find '{MODEL_PATH}'!\n"
                            f"Please ensure your model is named exactly '{MODEL_PATH}' and is in this folder: {os.getcwd()}")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ ResNet loaded successfully!")
except Exception as e:
    raise RuntimeError(f"\n❌ CRITICAL ERROR: Failed to load the ResNet model. Details: {e}")

CLASS_NAMES = sorted([
    'dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis', 
    'normal-cecum', 'normal-pylorus', 'normal-z-line', 
    'polyps', 'ulcerative-colitis'
])

def get_health_status(class_name):
    if class_name in ['normal-cecum', 'normal-pylorus', 'normal-z-line']:
        return 'healthy'
    return 'abnormal'

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    
    raw_img = np.array(img) / 255.0
    tensor_img = np.expand_dims(raw_img, axis=0)
    
    return tensor_img, raw_img

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1, dpi=300)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return encoded


# =========================================================
# ResNet Grad-CAM 
# =========================================================
def generate_resnet_gradcam(tensor_img, raw_img, pred_index):
    try:
        target_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                target_layer = layer
                break
        
        if target_layer is None:
            raise ValueError("Could not find a Conv2D layer in the model!")

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[target_layer.output, model.output]
        )
        
        tensor_img_tf = tf.convert_to_tensor(tensor_img, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            outputs = grad_model(tensor_img_tf)
            
            conv_output = outputs[0]
            preds = outputs[1]
            
            if isinstance(preds, list): preds = preds[0]
            if isinstance(conv_output, list): conv_output = conv_output[0]
                
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, conv_output)
        if isinstance(grads, list): grads = grads[0]
            
        weights = tf.reduce_mean(grads, axis=(1, 2))
        cam = tf.reduce_sum(weights[:, None, None, :] * conv_output, axis=-1)

        heatmap = cam[0].numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= (np.max(heatmap) + 1e-8)
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(raw_img)
        im1 = ax.imshow(heatmap_resized, cmap="jet", alpha=0.4) 
        ax.axis('off')
        
        cbar = plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Activation Strength (Grad-CAM)', rotation=270, labelpad=20)
        
        return fig_to_base64(fig)
    except Exception as e:
        print(f"❌ ResNet Grad-CAM error: {e}")
        return None

# =========================================================
# LIME
# =========================================================
def predict_fn_lime_shap(images):
    return model.predict(images, verbose=0)

def generate_lime(raw_img, pred_class):
    try:
        explainer_lime = lime_image.LimeImageExplainer()
        explanation = explainer_lime.explain_instance(
            raw_img.astype('double'), 
            predict_fn_lime_shap, 
            top_labels=3, 
            hide_color=0, 
            num_samples=1000 
        )
        
        temp, mask = explanation.get_image_and_mask(
            pred_class, 
            positive_only=True, 
            num_features=8, 
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
# THE OFFICIAL SHAP IMPLEMENTATION (GitHub Standard)


# =========================================================
# THE OFFICIAL SHAP IMPLEMENTATION (Ultra-High Quality Edition)
# =========================================================
def generate_real_shap(tensor_img, raw_img, pred_class):
    try:
        # 1. Use official SHAP Image Masker
        masker = shap.maskers.Image("inpaint_telea", raw_img.shape)
        
        # 2. Initialize the Explainer
        explainer = shap.Explainer(predict_fn_lime_shap, masker)
        
        # 3. Boosted max_evals to 500 for higher mathematical precision
        shap_values_obj = explainer(tensor_img, max_evals=500, batch_size=32)
        
        # 4. Extract raw numpy arrays
        class_shap_values = shap_values_obj.values[..., pred_class] 
        
        # 5. Plot using official SHAP library
        plt.figure()
        shap.image_plot([class_shap_values], tensor_img, show=False)
        
        fig = plt.gcf()
        
        # --- 🚀 THE QUALITY BOOST 🚀 ---
        # Double the physical dimensions of the SHAP figure
        current_size = fig.get_size_inches()
        fig.set_size_inches(current_size[0] * 2, current_size[1] * 2)
        
        # Bypass the standard fig_to_base64 to enforce Ultra-High DPI (500) and zero padding
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=500)
        buf.seek(0)
        b64_img = base64.b64encode(buf.read()).decode('utf-8')
        
        plt.close('all')
        return b64_img
        
    except Exception as e:
        print(f"❌ Real SHAP error: {e}")
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
        
        predictions = model.predict(tensor_img, verbose=0)[0]
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

        print("Generating ResNet Grad-CAM...")
        gradcam_b64 = generate_resnet_gradcam(tensor_img, raw_img, pred_class)
        
        print("Generating ResNet LIME...")
        lime_b64 = generate_lime(raw_img, pred_class) 
        
        print("Generating ResNet Real SHAP...")
        occlusion_b64 = generate_real_shap(tensor_img, raw_img, pred_class)
        
        print("Analysis Complete! Sending back to frontend.")
        
        response = {
            'top_prediction': top_class_display,
            'top_confidence': f"{round(top_confidence * 100, 2)}%",
            'top_prediction_raw': top_class_raw,
            'health_status': get_health_status(top_class_raw),
            'all_predictions': all_predictions,
            'model_used': "ResNet50",
            'inference_time': inference_time,
            'gradcam': gradcam_b64,
            'lime': lime_b64,
            'shap': occlusion_b64 
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Server Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5051)