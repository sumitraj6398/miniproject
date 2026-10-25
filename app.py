import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from gradcam import GradCAM
from model import load_model

MODEL_PATH = "best_brain_tumor_model.pth"
# Alphabetical order = torchvision ImageFolder default. Must match training.
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary",
}

LOW_CONFIDENCE = 60.0  # percent

st.set_page_config(page_title="Brain Tumor Classification", page_icon="🧠", layout="wide")

st.markdown(
    """
<style>
.prediction-box {
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #444;
    text-align: center;
}
</style>
""",
    unsafe_allow_html=True,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_brain_model():
    model = load_model(num_classes=len(CLASSES))
    state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

with st.sidebar:
    st.title("🧠 Brain Tumor AI")
    st.markdown("---")
    st.markdown(
        """
### Model Information

- EfficientNet-B0
- Transfer Learning
- PyTorch
- Streamlit
- Grad-CAM

### Classes

- Glioma
- Meningioma
- No Tumor
- Pituitary
"""
    )
    st.markdown("---")
    st.info("Upload a brain MRI image to get a prediction and Grad-CAM visualization.")
    st.warning(
        "For educational purposes only. This is not a medical diagnostic tool."
    )

try:
    model = load_brain_model()
except FileNotFoundError:
    st.error(f"Model file not found at `{MODEL_PATH}`. Add the trained .pth file to the repo.")
    st.stop()

cam = GradCAM(model, model.features[-1])

st.title("🧠 Brain Tumor Classification System")
st.write("Upload an MRI image and get a prediction with Grad-CAM explainability.")
st.markdown("---")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded MRI")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Prediction")

        if st.button("Predict"):
            image_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                probabilities = torch.softmax(model(image_tensor), dim=1)
                conf, pred = torch.max(probabilities, dim=1)

            pred_idx = pred.item()
            confidence = conf.item() * 100
            predicted_class = CLASSES[pred_idx]

            st.markdown(
                f"""
                <div class="prediction-box">
                    <h2>{DISPLAY_NAMES[predicted_class].upper()}</h2>
                    <h3>Confidence: {confidence:.2f}%</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if confidence < LOW_CONFIDENCE:
                st.warning(
                    "Low confidence. The image may not be a brain MRI or may be out of "
                    "the model's training distribution."
                )

            st.markdown("### Class Probabilities")
            for idx, cls in enumerate(CLASSES):
                prob = probabilities[0][idx].item() * 100
                st.write(f"{DISPLAY_NAMES[cls]}: {prob:.2f}%")
                st.progress(min(max(prob / 100, 0.0), 1.0))

            # Grad-CAM
            heatmap = cam.generate(image_tensor, pred_idx)
            heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

            img_np = np.array(image.resize((224, 224)))
            overlay = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)

            st.markdown("---")
            st.subheader("🔥 Grad-CAM Visualization")
            st.image(overlay, use_container_width=True)

st.markdown("---")
st.markdown("Built with ❤️ using PyTorch, Streamlit, EfficientNet-B0 and Grad-CAM")
