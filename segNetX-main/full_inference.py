from arch import SegFormerUNet
import torch
import torch.nn as nn
import numpy as np 
import cv2
import matplotlib.pyplot as plt
import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Define Transformations
# transform = A.Compose([
#     A.Resize(256, 256),  # Resize to SegFormer input size
#     A.HorizontalFlip(p=0.5),
#     A.RandomBrightnessContrast(p=0.2),
#     A.Normalize(mean=[0.5], std=[0.5]),
#     ToTensorV2()
# ])
transform = A.Compose([
    A.Resize(256, 256),  # Resize to SegFormer input size
    A.HorizontalFlip(p=0.5),  # Randomly flip horizontally
    A.VerticalFlip(p=0.2),  # Randomly flip vertically
    A.RandomBrightnessContrast(p=0.2),  # Adjust brightness and contrast
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=20, p=0.5),  # Small shifts, scaling, rotation
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),  # Slight blurring for robustness
    # A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),  # Add slight Gaussian noise
    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),  # Slight grid distortion
    A.Normalize(mean=[0.5], std=[0.5]),  # Normalize
    ToTensorV2()  # Convert to tensor
])

# Custom Dataset Class
class SolarPanelDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace(".bmp", "_label.bmp"))

        # Load Image & Mask
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype("uint8")  # Convert to binary mask
        # Apply Transformations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask.unsqueeze(0)  # Add channel dimension

# Load Dataset
val_dataset = SolarPanelDataset("dataset/val/images", "dataset/val/labels", transform=transform)

def compute_solar_area(mask, PTM=0.125, OPTA=34):
    """
    Compute solar panel area from a binary segmentation mask.
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().detach().numpy()  # Convert to NumPy if Tensor
    if mask.ndim == 3:
        mask = mask.squeeze(0)  # Remove extra channel

    mask = (mask > 0.5).astype(np.float32)  # Ensure binary mask
    panel_pixels = mask.sum()  # Count solar panel pixels
    area_m2 = (panel_pixels * (PTM ** 2)) / np.cos(np.radians(OPTA))  # Convert to m²
    return area_m2

def compute_accuracy_metrics(segmented_mask, actual_mask, PTM=0.125):
    """
    Compute accuracy of segmented area vs. actual area using MAPE and IoU.
    """
    # Compute solar panel areas
    segmented_area = compute_solar_area(segmented_mask, PTM)
    actual_area = compute_solar_area(actual_mask, PTM)

    # Compute Mean Absolute Percentage Error (MAPE)
    mape_error = np.abs((segmented_area - actual_area) / actual_area) * 100 if actual_area != 0 else 0

    # Compute Intersection over Union (IoU)
    intersection = ((segmented_mask > 0.5) & (actual_mask > 0.5)).sum()
    union = ((segmented_mask > 0.5) | (actual_mask > 0.5)).sum()
    iou_score = intersection / union if union != 0 else 0

    return {
        "Segmented Area (m²)": segmented_area,
        "Actual Area (m²)": actual_area,
        "MAPE (%)": mape_error,
        "IoU Score": iou_score
    }


def compute_energy_output(area_m2, efficiency=0.19, GTI=1676.2, PR=0.935):
    """
    Compute estimated solar energy output.
    """
    return area_m2 * efficiency * GTI * PR

def Calculate_solar_energy(val_dataset, model, idx=0):
    model.eval()

    # Load image and mask from validation set
    image, mask = val_dataset[idx]
    orig_image = np.moveaxis(image.numpy(), 0, -1)  # Convert from (C, H, W) to (H, W, C)

    # Move image to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)  # Get raw logits
        pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()  # Convert logits to probabilities
        pred_mask = (pred_mask > 0.5).astype(np.uint8)  # Convert to binary mask

    # Resize ground truth mask for plotting
    mask = mask.squeeze().numpy()

    difference = np.sum(mask != pred_mask)
    print(f"Number of different pixels: {difference}")

    print("-"*20)
    # Compute area in m²
    area_m2 = compute_solar_area(mask)
    print("ORIGINAL MASK ENERGY OUTPUT")
    print(f"Estimated Solar Panel Area: {area_m2:.2f} m²")

    # Compute energy output in kWh
    energy_kwh = compute_energy_output(area_m2)
    print(f"Estimated Energy Output: {(energy_kwh/1000):.2f} MWh per year")
    print("-"*20)

    print("-"*20)
    # Compute area in m²
    area_m2 = compute_solar_area(pred_mask)
    print("PREDICTED MASK ENERGY OUTPUT")
    print(f"Estimated Solar Panel Area: {area_m2:.2f} m²")

    # Compute energy output in kWh
    energy_kwh = compute_energy_output(area_m2)
    print(f"Estimated Energy Output: {(energy_kwh/1000):.2f} MWh per year")
    print("-"*20)

    # Plot the results
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(orig_image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Ground Truth Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask, cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# sample_input = torch.randn(1, 3, 512, 512).to(device)
model = SegFormerUNet().to(device)
model.eval()
checkpoint_path = "model/segformer_unet_focal_loss_97_63.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)

# Load model state dict
model.load_state_dict(checkpoint)

print("Model weights loaded successfully!")

# with torch.no_grad():
#     output = model(sample_input)

# Run visualization for a random validation sample
Calculate_solar_energy(val_dataset, model, idx=21)