import cv2
import numpy as np
import torch


class GradCAM:
    """Grad-CAM that registers its hooks only for the duration of generate(),
    so nothing accumulates on the cached model across Streamlit reruns."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

    def generate(self, input_tensor, class_idx, size=(224, 224)):
        store = {}

        def forward_hook(module, inputs, output):
            store["activations"] = output
            output.register_hook(lambda grad: store.__setitem__("gradients", grad))

        handle = self.target_layer.register_forward_hook(forward_hook)

        try:
            with torch.enable_grad():
                self.model.zero_grad()
                output = self.model(input_tensor)
                output[0, class_idx].backward()
        finally:
            handle.remove()

        activations = store["activations"].detach()
        gradients = store["gradients"].detach()

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activations).sum(dim=1)).squeeze()
        cam = cam.cpu().numpy()

        cam = cv2.resize(cam, size)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam
