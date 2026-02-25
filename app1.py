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

# TensorFlow - Using the base import to prevent IDE underline bugs
import tensorflow as tf

# XAI Libraries
from lime import lime_image
from skimage.segmentation import mark_boundaries

app = Flask(__name__)

# =========================================================
# 1. STRICT MODEL INITIALIZATION
# =========================================================
MODEL_PATH = 'cnn_model.h5' 

print(f"Loading CNN on TensorFlow {tf.__version__}...")

# Check if the file actually exists before trying to load it!
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"\n❌ CRITICAL ERROR: Could not find '{MODEL_PATH}'!\n"
                            f"Please download 'kvasir_enhanced_model.h5' from your Google Drive, "
                            f"rename it to '{MODEL_PATH}', and place it in this exact folder: {os.getcwd()}")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ CNN loaded successfully!")
except Exception as e:
    raise RuntimeError(f"\n❌ CRITICAL ERROR: Failed to load the CNN model. Details: {e}")

CLASS_NAMES = sorted([
    'dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis', 
    'normal-cecum', 'normal-pylorus', 'normal-z-line', 
    'polyps', 'ulcerative-colitis'
])

def get_health_status(class_name):
    if class_name in ['normal-cecum', 'normal-pylorus', 'normal-z-line']:
        return 'healthy'
    return 'abnormal'

# EXACT Preprocessing from your Notebook (rescale=1./255)
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

def predict_fn_lime(images):
    return model.predict(images, verbose=0)

# =========================================================
# EXACT NOTEBOOK CODE: Improved Grad-CAM
# =========================================================
# =========================================================
# EXACT NOTEBOOK CODE: Improved Grad-CAM (Keras 3 Safe)
# =========================================================
def generate_notebook_gradcam(tensor_img, raw_img, pred_index):
    try:
        layer_name = "last_conv_layer"
        try:
            model.get_layer(layer_name)
        except ValueError:
            for layer in reversed(model.layers):
                if len(layer.output_shape) == 4:
                    layer_name = layer.name
                    break

        grad_model = tf.keras.models.Model(model.inputs, [model.get_layer(layer_name).output, model.output])
        
        # Keras 3 strict tensor conversion to avoid input structure warnings
        tensor_img_tf = tf.convert_to_tensor(tensor_img, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            outputs = grad_model(tensor_img_tf)
            
            # Keras 3 safely unwrap outputs if they are in a list
            last_conv_layer_output = outputs[0]
            preds = outputs[1]
            
            if isinstance(preds, list):
                preds = preds[0]
            if isinstance(last_conv_layer_output, list):
                last_conv_layer_output = last_conv_layer_output[0]
                
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        if isinstance(grads, list):
            grads = grads[0]
            
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        last_conv_output = last_conv_layer_output[0]
        heatmap = last_conv_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap_np = heatmap.numpy()

        heatmap_resized = cv2.resize(heatmap_np, (224, 224))
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(raw_img)
        im1 = ax.imshow(heatmap_resized, cmap="jet", alpha=0.4) 
        ax.axis('off')
        
        cbar = plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Activation Strength (Grad-CAM)', rotation=270, labelpad=20)
        
        return fig_to_base64(fig)
    except Exception as e:
        print(f"❌ Notebook Grad-CAM error: {e}")
        return None

# =========================================================
# EXACT NOTEBOOK CODE: LIME
# =========================================================
def generate_lime(raw_img, pred_class):
    try:
        explainer_lime = lime_image.LimeImageExplainer()
        
        explanation = explainer_lime.explain_instance(
            raw_img.astype('double'), 
            predict_fn_lime, 
            top_labels=3, 
            hide_color=0, 
            num_samples=500 
        )
        
        temp, mask = explanation.get_image_and_mask(
            pred_class, 
            positive_only=True, 
            num_features=10, 
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
# EXACT NOTEBOOK CODE: Occlusion Sensitivity
# =========================================================
def compute_occlusion_sensitivity(tensor_img, raw_img, pred_class):
    try:
        patch_size = 16
        stride = 8
        
        img_height, img_width = 224, 224
        sensitivity_map = np.zeros((img_height, img_width))

        original_pred = model.predict(tensor_img, verbose=0)
        original_prob = np.max(original_pred)

        for y in range(0, img_height - patch_size + 1, stride):
            for x in range(0, img_width - patch_size + 1, stride):
                occluded = raw_img.copy()
                occluded[y:y+patch_size, x:x+patch_size] = 0 
                
                occ_tensor = np.expand_dims(occluded, axis=0)
                occluded_pred = model.predict(occ_tensor, verbose=0)
                occluded_prob = occluded_pred[0][pred_class] 

                importance = original_prob - occluded_prob
                sensitivity_map[y:y+patch_size, x:x+patch_size] = np.maximum(
                    sensitivity_map[y:y+patch_size, x:x+patch_size], importance
                )

        sensitivity_map = np.maximum(sensitivity_map, 0)
        if sensitivity_map.max() > 0:
            sensitivity_map = sensitivity_map / sensitivity_map.max()

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(raw_img)
        
        im = ax.imshow(sensitivity_map, cmap="hot", alpha=0.5)
        ax.axis('off')
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Importance (Occlusion Sensitivity)', rotation=270, labelpad=20)

        b64_img = fig_to_base64(fig)
        plt.close('all') 
        return b64_img
    except Exception as e:
        print(f"❌ Occlusion Sensitivity error: {e}")
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

        print("Generating Notebook's Grad-CAM...")
        gradcam_b64 = generate_notebook_gradcam(tensor_img, raw_img, pred_class)
        
        print("Generating Notebook's LIME...")
        lime_b64 = generate_lime(raw_img, pred_class) 
        
        print("Generating Notebook's Occlusion Sensitivity...")
        occlusion_b64 = compute_occlusion_sensitivity(tensor_img, raw_img, pred_class)
        
        print("Analysis Complete! Sending back to frontend.")
        
        response = {
            'top_prediction': top_class_display,
            'top_confidence': f"{round(top_confidence * 100, 2)}%",
            'top_prediction_raw': top_class_raw,
            'health_status': get_health_status(top_class_raw),
            'all_predictions': all_predictions,
            'model_used': "Convolutional Neural Network (CNN)",
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
    app.run(debug=True, host='0.0.0.0', port=5050)