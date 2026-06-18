"""
gradcam.py
Generates Grad-CAM heatmap overlay for an MRI image using VGG16.
"""

import os
import uuid
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

import predict as P


def _get_model():
    
    if 'vgg16' not in P._models:
        raise RuntimeError(
            "Grad-CAM requires the VGG16 model, but it is not loaded "
            f"(loaded models: {list(P._models.keys())}). Check that "
            "vgg16_final.keras exists in the model directory."
        )
    return P._models['vgg16']


def _gradcam(model, batch):
    
    try:
        base_vgg = model.get_layer("vgg16")
    except Exception as e:
        print("GradCAM Error (no nested vgg16 base layer found):", e)
        return np.zeros((6, 6), dtype=np.float32)

    x = tf.convert_to_tensor(batch, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv1 = base_vgg.get_layer("block1_conv1")(x)
        conv2 = base_vgg.get_layer("block1_conv2")(conv1)
        p1 = base_vgg.get_layer("block1_pool")(conv2)

        c21 = base_vgg.get_layer("block2_conv1")(p1)
        c22 = base_vgg.get_layer("block2_conv2")(c21)
        p2 = base_vgg.get_layer("block2_pool")(c22)

        c31 = base_vgg.get_layer("block3_conv1")(p2)
        c32 = base_vgg.get_layer("block3_conv2")(c31)
        c33 = base_vgg.get_layer("block3_conv3")(c32)
        p3 = base_vgg.get_layer("block3_pool")(c33)

        c41 = base_vgg.get_layer("block4_conv1")(p3)
        c42 = base_vgg.get_layer("block4_conv2")(c41)
        c43 = base_vgg.get_layer("block4_conv3")(c42)
        p4 = base_vgg.get_layer("block4_pool")(c43)

        c51 = base_vgg.get_layer("block5_conv1")(p4)
        c52 = base_vgg.get_layer("block5_conv2")(c51)

        conv_out = base_vgg.get_layer("block5_conv3")(c52)

        p5 = base_vgg.get_layer("block5_pool")(conv_out)

        y = model.layers[2](p5)     # Flatten
        y = model.layers[3](y)      # Dropout
        y = model.layers[4](y)      # Dense(128)
        y = model.layers[5](y)      # Dropout
        preds = model.layers[6](y)  # Softmax

        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_out)

    if grads is None:
        print("GradCAM Error: gradient is None (conv_out not connected to loss)")
        return np.zeros((6, 6), dtype=np.float32)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_sum(conv_out[0] * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)

    if tf.reduce_max(heatmap) > 0:
        heatmap /= tf.reduce_max(heatmap)

    return heatmap.numpy()


def generate_gradcam_overlay(img_path, save_dir, alpha=0.45):
    os.makedirs(save_dir, exist_ok=True)

    img  = Image.open(img_path).convert('RGB').resize((P.IMAGE_SIZE, P.IMAGE_SIZE))
    arr  = np.array(img, dtype=np.float32) / 255.0
    batch = np.expand_dims(arr, 0)

    raw  = _gradcam(_get_model(), batch)
    hm   = cv2.resize(
        raw.astype(np.float32),
        (P.IMAGE_SIZE, P.IMAGE_SIZE),
        interpolation=cv2.INTER_CUBIC
    )
    hm   = np.clip(hm, 0, 1)

    THRESHOLD = 0.10
    mask     = (hm > THRESHOLD)
    area_pct = round(float(np.sum(mask)) / (P.IMAGE_SIZE * P.IMAGE_SIZE) * 100, 2)

    jet       = plt.cm.jet(hm)[:, :, :3]
    overlay   = np.clip((1 - alpha) * arr + alpha * jet, 0, 1)

    uid = uuid.uuid4().hex[:10]

    hm_file = f'{uid}_heatmap.jpg'
    ov_file = f'{uid}_overlay.jpg'

    cv2.imwrite(
        os.path.join(save_dir, hm_file),
        cv2.cvtColor((jet * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    )
    cv2.imwrite(
        os.path.join(save_dir, ov_file),
        cv2.cvtColor((overlay * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    )

    return {
        'heatmap_filename': hm_file,
        'overlay_filename': ov_file,
        'area_pct':         area_pct,
    }
