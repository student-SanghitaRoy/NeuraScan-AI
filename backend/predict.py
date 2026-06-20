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

# ── NON-MRI IMAGE REJECTION THRESHOLDS ───────────────────────────────────────
# Brain MRI scans are essentially grayscale (R≈G≈B per pixel). A colour photo
# of an everyday object will have much higher per-pixel channel deviation.
# This threshold was tuned against synthetic colour-vs-grayscale test images;
# real JPEG compression can introduce a few units of noise even in genuinely
# grayscale scans, so this leaves headroom rather than using 0.
GRAYSCALE_DEVIATION_THRESHOLD = 10.0

# The model always outputs a full softmax distribution, even for images it
# has never seen anything like. On genuinely out-of-distribution input it
# tends to produce a low, close-to-uniform spread across all 4 classes
# (no real learned signal to latch onto), rather than a confident, decisive
# call. Confidence below this on the winning class is treated as "the model
# isn't sure this is even a brain scan", not just "uncertain about which
# tumour type". Real brain MRI test scans consistently scored 90-100%
# confidence, so this threshold still leaves real scans comfortable margin
# while catching uncertain/wrong-domain input much more aggressively.
MIN_CONFIDENCE_THRESHOLD = 65.0

# Shannon entropy of the 4-class probability distribution, in bits.
# Max possible entropy for 4 classes (perfectly uniform 25/25/25/25) is 2.0
# bits. A confident, decisive prediction has LOW entropy even if it isn't
# the only signal — e.g. [55%,30%,10%,5%] passes a 65% confidence bar by
# being just under it but is still a fairly spread-out, uncertain-looking
# distribution; entropy catches that shape directly rather than relying on
# the single winning-class number alone.
MAX_ENTROPY_THRESHOLD = 1.0

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


class NotBrainMRIError(Exception):
    """Raised when the uploaded image doesn't look like a brain MRI scan."""
    pass


def _grayscale_deviation(arr):
    """
    arr: float32 HxWx3 array, values 0-255 range.
    Returns the average per-pixel deviation of each channel from the
    pixel's mean — near 0 for true grayscale images, much higher for
    colour photos.
    """
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    mean = (r + g + b) / 3.0
    deviation = (np.abs(r - mean) + np.abs(g - mean) + np.abs(b - mean)) / 3.0
    return float(np.mean(deviation))


def _entropy_bits(probs):
    """
    Shannon entropy of a probability distribution, in bits.
    0.0 = fully confident (one class at 100%). Higher = more spread out/
    uncertain across classes. Max for 4 classes (uniform 25% each) is 2.0.
    """
    p = np.clip(probs, 1e-12, 1.0)  # avoid log(0)
    return float(-np.sum(p * np.log2(p)))


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
    arr_raw = np.array(img, dtype=np.float32)  # 0-255 range, for the grayscale check

    deviation = _grayscale_deviation(arr_raw)
    if deviation > GRAYSCALE_DEVIATION_THRESHOLD:
        raise NotBrainMRIError(
            'This image doesn\'t look like a grayscale brain MRI scan '
            '(too much colour variation). Please upload a brain MRI image.'
        )

    arr = arr_raw / 255.0
    batch = np.expand_dims(arr, axis=0)

    probs = np.zeros(NUM_CLASSES, dtype=np.float64)
    for model in _models.values():
        p = model.predict(batch, verbose=0)[0]
        probs += p.astype(np.float64)
    probs /= len(_models)

    idx        = int(np.argmax(probs))
    pred_class = CLASS_LABELS[idx]
    confidence = round(float(probs[idx]) * 100.0, 2)
    entropy    = _entropy_bits(probs)

    if confidence < MIN_CONFIDENCE_THRESHOLD or entropy > MAX_ENTROPY_THRESHOLD:
        raise NotBrainMRIError(
            'This image doesn\'t look like a brain MRI scan the model '
            'recognises (confidence too low / distribution too uncertain '
            'across categories). Please upload a clear brain MRI image.'
        )

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
