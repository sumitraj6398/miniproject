# 🧠 Brain Tumor Classification Using EfficientNet-B0

Live app: <add your streamlit.app URL here>

A Streamlit web app that classifies brain MRI scans into four categories using a fine-tuned EfficientNet-B0, with Grad-CAM heatmaps showing which regions influenced the prediction.

> **Disclaimer:** for educational purposes only. Not a medical diagnostic tool.

## Features

- Brain MRI classification (Glioma, Meningioma, Pituitary, No Tumor)
- Transfer learning with EfficientNet-B0 (PyTorch)
- Grad-CAM explainability
- Confidence score, per-class probabilities, low-confidence warning

## Model

- Backbone: EfficientNet-B0, custom classification head
- Input: 224 × 224, ImageNet normalization
- Loss: CrossEntropyLoss, Optimizer: Adam
- Class order: glioma, meningioma, notumor, pituitary

| Metric              | Value |
| ------------------- | ----- |
| Training Accuracy   | 97.6% |
| Validation Accuracy | 96%   |
| Test Accuracy       | 96.4% |

## Project Structure

```text
brain-tumor-classification/
├── app.py
├── model.py
├── gradcam.py
├── requirements.txt
├── .gitignore
├── models/
│   └── best_brain_tumor_model.pth
└── README.md
```

## Run Locally

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501

## Deploy (Streamlit Community Cloud)

1. Push this folder to a GitHub repo (including `models/best_brain_tumor_model.pth`).
2. Go to https://share.streamlit.io → Create app.
3. Pick the repo, branch `main`, main file `app.py`.
4. Advanced settings → Python 3.11 → Deploy.

## Author

Aditya Singh
