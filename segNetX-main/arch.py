import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation

class SegFormerUNet(nn.Module):
    def __init__(self, model_name="nvidia/segformer-b2-finetuned-ade-512-512", num_classes=1):
        super(SegFormerUNet, self).__init__()

        # Load Pretrained SegFormer
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(model_name)

        # Extract Encoder
        self.encoder = self.segformer.segformer.encoder  # Correct way to get encoder

        # U-Net Style Decoder (Upsampling to match input size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),  # 16x16 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # 32x32 -> 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),   # 64x64 -> 128x128
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),    # 128x128 -> 256x256
            nn.ReLU(),
            nn.ConvTranspose2d(32, num_classes, kernel_size=2, stride=2)  # 256x256 -> 512x512
        )

    def forward(self, x):
        retained_input = x  # Keep input image

        # Encoder processing
        encoder_output = self.encoder(x)  # Extract encoder features
        encoder_output = encoder_output.last_hidden_state.permute(0, 1, 2, 3)  # (B, C, H, W)
        # print("Encoder Output Shape:", encoder_output.shape)  # Should be (B, 512, 16, 16)

        # Decoder (Upsample back to input size)
        output = self.decoder(encoder_output)  # (B, num_classes, 512, 512)

        return output  # return segmentation mask
