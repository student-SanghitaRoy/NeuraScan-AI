# 🧠 NeuraScan AI — Brain Tumour Detection using Deep Learning

## 🔗 Live Deployment

🚀 **Deployed Application:**
👉 https://huggingface.co/spaces/HFRsanghita/neurascan-ai

---

## 📌 Project Overview

This project implements an **AI-powered Brain Tumour Detection system** that classifies brain MRI scans into four categories — glioma, meningioma, pituitary tumour, or no tumour — using a deep convolutional neural network.

The system uses **transfer learning (VGG16)** for classification, **Grad-CAM** for visual explainability, and is deployed as a full **Flask web application** with automated PDF report generation and an AI chatbot assistant.

This project demonstrates a complete, end-to-end deep learning pipeline — from dataset preparation and model training, through explainability and evaluation, to a production-grade cloud deployment.

---

## 🧠 Methodology

### 1️⃣ Data Preprocessing

- Loaded the dataset (organised as `Training/` and `Testing/` folders, 4 classes, 7,200 images total)
- Created a stratified 80/20 validation split from the Training data in code (4,480 train / 1,120 validation), keeping the Testing set (1,600 images) fully separate for final evaluation
- Resized all images to 96×96×3 and normalised pixel values
- Applied data augmentation on the training split only

### 2️⃣ Model Training (Transfer Learning)

- Fine-tuned three ImageNet-pretrained CNN backbones: **VGG16**, **ResNet50**, **EfficientNetB0**
- Trained under a mixed-precision (float16) policy for faster convergence
- Compared all three models and a soft-voting ensemble on held-out test data

### 3️⃣ Explainability (Grad-CAM)

- Implemented Gradient-weighted Class Activation Mapping (Grad-CAM) on VGG16
- Generated attention heatmaps highlighting the regions influencing each prediction
- Critically evaluated heatmap quality, including known limitations on ambiguous scans

### 4️⃣ Deployment Pipeline

- Packaged the trained model into a Flask web application
- Added Grad-CAM visualisation, automated PDF reporting, and input validation
- Containerised with Docker and deployed to a public cloud host

---

## 🛠 Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pillow
- Matplotlib
- Flask
- HTML, CSS & JavaScript
- fpdf2 (PDF report generation)
- Sarvam AI API (chatbot assistant)
- Gunicorn
- Docker
- Hugging Face Spaces (Cloud Deployment)

---

## ✨ Features

- Brain MRI classification across 4 tumour categories
- Grad-CAM attention heatmap for every prediction
- Confidence score and colour-coded risk level (Low / Medium / High)
- Automated, downloadable PDF diagnostic report
- AI chatbot for plain-language explanation of results
- Input validation to reject non-MRI image uploads
- Clean, responsive, dark-themed user interface
- Cloud-deployed, publicly accessible application

---

## 📂 Project Structure

```
BRAINTUMOURDETECTOR/
│
├── backend/
│   ├── app.py
│   ├── predict.py
│   ├── gradcam.py
│   ├── report.py
│   └── model/
│       └── vgg16_final.keras
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── static/
│   ├── favicon.svg
│   ├── uploads/
│   ├── gradcam/
│   └── reports/
│
├── Brain MRI Dataset
│
├── Dockerfile
├── Procfile
├── requirements.txt
└── README.md
```

---

## 📊 System Workflow

User Uploads MRI Scan → Image Validation → VGG16 Inference →
Grad-CAM Heatmap Generation → Risk Level Assignment →
PDF Report Generation → Results Displayed in Web Interface

---

## 🔮 Future Improvements

- Re-tune EfficientNetB0 to make it a viable ensemble contributor
- Run full 5-fold cross-validation for a more robust accuracy estimate
- Train a dedicated gatekeeper model to more reliably reject non-MRI uploads
- Targeted data augmentation to reduce glioma/meningioma misclassification
- Multi-language support for the chatbot assistant

---

## 📚 Learning Outcomes

- Practical implementation of CNN transfer learning for medical imaging
- Model evaluation using accuracy, F1-score, confusion matrices, and ROC-AUC
- Explainable AI using Grad-CAM and critical interpretation of its limitations
- Flask backend integration with file upload and PDF generation
- Production deployment troubleshooting (CPU/memory constraints, dependency conflicts)
- Docker containerisation and cloud deployment of a deep learning application

---

## 👩‍💻 Project Information

- **Project Title:** NeuraScan AI — Brain Tumour Detector
- **Author Name:** Sanghita Roy
- **Roll Number:** 23035010421
- **Program:** B.Sc. (Hons.) Data Science & Artificial Intelligence
- **Institute:** IIT Guwahati

---

## ⭐ Academic Purpose

This project was developed as part of academic learning to demonstrate understanding of:

- Convolutional neural networks and transfer learning
- Medical image classification
- Explainable AI (Grad-CAM)
- End-to-end deployment of deep learning applications

> ⚠️ **Disclaimer:** This system is for research and educational purposes only. It is **not** a certified medical device and does not provide medical diagnoses. All results must be reviewed by a qualified neurologist or radiologist.
