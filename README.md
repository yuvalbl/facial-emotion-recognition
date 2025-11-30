# 🧠 Facial Expression Recognition (FER) Model  
*A deep-learning computer vision system for real-time emotion recognition*

## 📌 Overview
This repository contains our custom **Facial Expression Recognition (FER)** model developed by our team.  
The goal of the project is to accurately classify facial expressions into predefined emotional categories using modern deep-learning techniques.

---

## 🚀 Features
- explored few pre-trained models (VGG, FaceNet, ResNet and Swin vision transformers)
- fine-tune models to get better resualt for face emotions recognition
- Trained on **FER-2013**, **RAFDB**, and verified on generated syntatic data
- Supports **7 emotion classes**:  
  `angry`, `disgust`, `fear`, `happy`, `sad`, `surprise`, `neutral`
- Real-time inference (webcam / video stream / image input)
- On-CPU and GPU support

---

## 📂 Project Structure
```
├── notebooks/           # Jupyter notebooks of different models and EDA
├── data/
├── results/             # plots
└── README.md
```

---

## 🔧 Installation

### Clone the repo
```bash
git clone https://github.com/yuvalbl/facial-emotion-recognition.git
cd facial-emotion-recognition
```

### Install dependencies
```bash
python -m venv .venv
source ./.venv/bin/activate  # on unix and mac
pip install -r requirements.txt
```

### Getting data
to download dataset from Kaggle run the data_prep notebook, this should create a data folder with `fer_dataset` in it.

---

### Run the inference notebook

To run `notebooks/inference.ipynb`, first download our trained model weights from the Hugging Face Hub:

```bash
pip install -U huggingface_hub
```

---

## 📈 Results (example — replace with yours)

| Metric/Score | VGG16 | FaceNet | ResNet50 | Swin Vision Transformer
|--------|--------|--------|--------|--------|
| Accuracy | 68% | 54% | 36% | 78% |
| Macro F1 | 68% | 54% | 32% | 78% |


Confusion matrix and plots are in `/results/`.

---

## 🤝 Team Members

- **Nati Shchiglik**
- **Yaniv Kempler**
- **Yuval Bar Levi**
- **Wisam Salameh**

---

## 🙌 Acknowledgments
- FER-2013 dataset
