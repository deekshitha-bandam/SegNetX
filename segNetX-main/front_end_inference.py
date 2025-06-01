import gradio as gr
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from arch import SegFormerUNet
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SegFormerUNet().to(device)
checkpoint_path = "model/segformer_unet_focal_loss_97_63.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint)
model.eval()
print("Model weights loaded successfully!")

# Image Transformation
transform = Compose([
    Resize(256, 256),
    Normalize(mean=[0.5], std=[0.5]),
    ToTensorV2()
])

def process_image(image):
    """Process uploaded image, perform segmentation, and compute energy output."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transformed = transform(image=image)['image'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(transformed)
        pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
        pred_mask = (pred_mask > 0.5).astype(np.uint8)
    
    area_m2 = np.sum(pred_mask) * (0.125 ** 2)
    energy_kwh = area_m2 * 0.19 * 1676.2 * 0.935 / 1000
    
    return pred_mask * 255, f"Estimated Solar Panel Area: {area_m2:.2f} m²", f"Estimated Energy Output: {energy_kwh:.2f} MWh per year"

demo = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="numpy"),
    outputs=[gr.Image(type="numpy"), gr.Text(), gr.Text()],
    title="Solar Panel Segmentation",
    description="Upload an image to detect solar panels and estimate energy output.",
)

demo.launch()