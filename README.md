# 🌾 CropCare AI: Intelligent Crop Disease Diagnosis

<div align="center">

**Designed as a comprehensive computer science and engineering senior project, this repository showcases a complete deep learning and computer vision pipeline for agricultural health.**

*CropCare AI is an end-to-end platform designed to empower farmers with instant, accurate crop disease diagnosis and actionable remedies.*

</div>

---

## 📌 Overview

Developed using **Convolutional Neural Networks**, this tool processes images of sick leaves to identify issues across **51 unique categories** covering **9 different crops**. The project demonstrates a complete machine learning lifecycle, from data preprocessing to high-concurrency API integration.

---

## ✨ Core Features

| Feature | Description |
|---------|-------------|
| 🔍 **Real-Time AI Diagnosis** | The system utilizes a fine-tuned `MobileNetV3Large` model to analyze uploaded leaf images at a `224x224` resolution. It predicts diseases with high statistical confidence. |
| 🌱 **Comprehensive Crop Support** | The platform analyzes grain crops (Corn, Rice, Wheat), cash crops (Cotton, Sugarcane), and vegetable/fruit crops (Tomato, Potato, Pepper, Grape). |
| 💊 **Actionable Treatment Plans** | The application generates immediate, disease-specific remedy protocols. It breaks down solutions into Preventative Measures, Chemical Controls, and Organic/Bio Controls. |
| 📚 **Knowledge Integration** | The diagnosis results are automatically mapped to relevant Wikipedia documentation for extended agricultural research. |

---

## 🧠 Model Optimization & Training Pipeline

### 🔄 Data Pipeline
> The dataset containing over **70,000 images** is extracted locally. It is processed using highly optimized TensorFlow data pipelines with automatic prefetching to maximize GPU throughput.

### 🎯 Transfer Learning Strategy
> The base `MobileNetV3Large` architecture was initialized with **ImageNet weights**. It was trained with a custom classification head using a sparse categorical cross-entropy loss function.

### ⚙️ Advanced Fine-Tuning
> The top **30 layers** of the base model were unfrozen and fine-tuned using a micro learning rate of `1e-5` to preserve pre-trained weights. This phase utilized balanced class weights to handle dataset imbalances across the 51 categories.

### 🛡️ Smart Callbacks
> The training loop incorporated `ReduceLROnPlateau` and `EarlyStopping` callbacks to prevent overfitting. This strategy ultimately achieved a final **validation accuracy of 93.44%**.

---

## 🏗️ Technical Architecture & Setup

### 🚀 High-Concurrency API
The backend is built with **FastAPI** and **Uvicorn** to ensure rapid, non-blocking image processing and inference routing.

### 🎨 Responsive Frontend
The interface is built with **HTML5, CSS3, JavaScript, and Bootstrap**. It features asynchronous fetching and dynamic DOM updates.

---

## 📦 Local Installation

| Step | Description |
|:----:|-------------|
| **1** | Clone the repository and install dependencies using `pip install -r requirements.txt'` |
| **2** | This environment requires `fastapi`, `uvicorn`, `python-multipart`, `tensorflow`, `Pillow`, and `jinja2`. |
| **3** | Place the fine-tuned model (`CropCare_MobileNetV3_FineTuned.keras`) in the `model/` directory. |
| **4** | Launch the server using `uvicorn main:app --reload` and navigate to `http://localhost:8000`. |

---

<div align="center">

**🌿 Empowering Farmers with Intelligent Crop Care 🌿**

</div>
