---
title: NeuraScan AI
emoji: 🧠
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# NeuraScan AI

AI-powered brain tumour detection web app using a VGG16 deep learning model with Grad-CAM explainability and automated PDF report generation.

## Overview

Upload a brain MRI scan and get:
- A classification across four categories: glioma, meningioma, pituitary tumour, or no tumour
- A confidence score and risk level
- A Grad-CAM attention heatmap showing which regions of the scan most influenced the model's prediction
- A downloadable PDF report

## Tech stack

- **Backend:** Flask, TensorFlow/Keras (VGG16)
- **Explainability:** Grad-CAM
- **Reporting:** fpdf2
- **Frontend:** HTML/CSS/JS (no framework)

## Disclaimer

This system is for research and educational purposes only. It is **not** a certified medical device and does not provide medical diagnoses. All results must be reviewed by a qualified neurologist or radiologist before any medical decision is made.
