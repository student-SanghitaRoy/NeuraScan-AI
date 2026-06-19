"""
predict.py
Loads VGG16, ResNet50, EfficientNet models and runs ensemble prediction.
"""

import os
import numpy as np
from PIL import Image
import tensorflow as tf

IMAGE_SIZE   = 96
CLASS_LABELS = ['glioma', 'meningioma', 'notumor', 'pituitary']
NUM_CLASSES  = 4

RISK_MAP = {
    'glioma':     {'level': 'High',   'color': '#e74c3c', 'icon': '🔴'},
    'meningioma': {'level': 'Medium', 'color': '#f39c12', 'icon': '🟡'},
    'notumor':    {'level': 'Low',    'color': '#27ae60', 'icon': '🟢'},
    'pituitary':  {'level': 'Medium', 'color': '#f39c12', 'icon': '🟡'},
}

CLASS_DISPLAY = {
    'glioma':     'Glioma',
    'meningioma': 'Meningioma',
    'notumor':    'No Tumour',
    'pituitary':  'Pituitary Tumour',
}

CLASS_DESC = {
    'glioma':     'Arises from glial cells. Most common malignant brain tumour. Requires urgent specialist review.',
    'meningioma': 'Arises from the meninges. Usually benign and slow-growing. Often monitored without surgery.',
    'notumor':    'No abnormal mass detected in this MRI scan. Appears within normal limits.',
    'pituitary':  'Tumour in the pituitary gland. Often treatable with medication or minimally invasive surgery.',
}

_models = {}


def _force_float32_policy(model):
    """
    Models trained in the notebook used tf.keras.mixed_precision.set_global_policy
    ('mixed_float16'). That per-layer dtype policy gets serialized into the saved
    .keras file, so EVERY layer (including the nested VGG16/ResNet50/EfficientNet
    base) keeps computing in float16 forever, even if this Flask process never
    sets any mixed precision policy itself.

    On CPU (no AVX-512 float16 support -> TensorFlow falls back to a slower,
    less precise Eigen path) this float16 compute path causes gradients computed
    via tf.GradientTape (as used for Grad-CAM) to underflow to ~0, which is why
    the resulting heatmap collapses to a flat, uniform image and tumour area
    rounds to 0.0%. Predictions still "work" because predict() only needs the
    final softmax probabilities, which tolerate the precision loss far better
    than raw gradients do.

    Forcing every layer's dtype policy to float32 after loading fixes this for
    both prediction and Grad-CAM, with no change to model weights or accuracy.
    """
    for layer in model.layers:
        try:
            layer.dtype_policy = tf.keras.mixed_precision.Policy('float32')
        except Exception:
            pass
        # Recurse into nested models (e.g. the VGG16/ResNet50/EfficientNet base)
        if hasattr(layer, 'layers'):
            _force_float32_policy(layer)


def load_models(model_dir):
    global _models
    if _models:
        return _models

    # NOTE: EfficientNet is intentionally NOT loaded. In testing it scored
    # only ~44.6% accuracy (barely above the 25% random baseline for 4
    # classes), and including it in the ensemble average pulled overall
    # accuracy DOWN from 90.4% (VGG16 alone) to 89.3% (3-way ensemble).
    # Until EfficientNet's training is fixed and re-validated, the ensemble
    # is more accurate without it.
    to_load = {
        'vgg16':   'vgg16_final.keras',
        'resnet50':'resnet50_final.keras',
    }

    loaded = {}
    for name, fname in to_load.items():
        path = os.path.join(model_dir, fname)
        if os.path.exists(path):
            print(f'  Loading {name} ...', flush=True)
            m = tf.keras.models.load_model(path, compile=False)
            if name == 'vgg16':
                # Only VGG16 needs float32: gradcam.py computes Grad-CAM
                # gradients against it specifically, and float16 gradients
                # underflow on CPU (see _force_float32_policy docstring).
                # ResNet50 is only used for forward-pass prediction, which
                # works fine at its native lower-memory precision, so we
                # leave it alone to save RAM on memory-constrained hosts.
                _force_float32_policy(m)
            loaded[name] = m
            print(f'  OK {name}', flush=True)
        else:
            print(f'  SKIP {name} — not found at {path}', flush=True)

    if not loaded:
        raise FileNotFoundError(f'No .keras files found in {model_dir}')

    _models = loaded
    return _models


def predict(img_path):
    if not _models:
        raise RuntimeError('Models not loaded.')

    img = Image.open(img_path).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    batch = np.expand_dims(arr, axis=0)

    probs = np.zeros(NUM_CLASSES, dtype=np.float64)
    for model in _models.values():
        p = model.predict(batch, verbose=0)[0]
        probs += p.astype(np.float64)
    probs /= len(_models)

    idx        = int(np.argmax(probs))
    pred_class = CLASS_LABELS[idx]
    confidence = round(float(probs[idx]) * 100.0, 2)
    risk       = RISK_MAP[pred_class]

    return {
        'pred_class':   pred_class,
        'display_name': CLASS_DISPLAY[pred_class],
        'confidence':   confidence,
        'risk_level':   risk['level'],
        'risk_color':   risk['color'],
        'risk_icon':    risk['icon'],
        'description':  CLASS_DESC[pred_class],
        'area_pct':     0.0,
        'probs':        {CLASS_LABELS[i]: round(float(probs[i]), 4) for i in range(NUM_CLASSES)},
        'models_used':  list(_models.keys()),
    }
