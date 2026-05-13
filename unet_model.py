"""
unet_model.py
=============
Lightweight U-Net for CAPTCHA image denoising/segmentation.

Predicts per-pixel probability: 0 = background, 1 = text character.
This enables "pixel-level probability prediction" for CAPTCHA solving.

Architecture:
    Encoder: 4 blocks, channels 32 -> 64 -> 128 -> 256
    Bottleneck: 512 channels
    Decoder: 4 blocks with skip connections
    Output: 1-channel sigmoid probability map (128x128)

Total params: ~2M (small, fast training on RTX 3060)

Usage:
    model = CaptchaUNet()
    mask = model(image_tensor)  # (B, 1, 128, 128), values 0-1
"""

import torch
import torch.nn as nn
from pathlib import Path


class ConvBlock(nn.Module):
    """Double convolution block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU.

    Standard building block of U-Net. Two 3x3 convolutions with batch
    normalization and ReLU activation.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class CaptchaUNet(nn.Module):
    """Lightweight U-Net for CAPTCHA text segmentation.

    Predicts a per-pixel probability map indicating which pixels belong
    to text characters vs background noise.

    Input:  (B, 3, 128, 128) - RGB CAPTCHA image
    Output: (B, 1, 128, 128) - Probability map (sigmoid applied)

    Architecture overview:
        Encoder path (downsampling):
            128x128x3  -> 128x128x32  -> 64x64x64  -> 32x32x128 -> 16x16x256
        Bottleneck:
            8x8x512
        Decoder path (upsampling with skip connections):
            16x16x256 -> 32x32x128 -> 64x64x64 -> 128x128x32
        Output:
            128x128x1 (sigmoid)
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1) -> None:
        super().__init__()

        # ── Encoder path ──────────────────────────────────────────────────────
        self.enc1 = ConvBlock(in_channels, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.enc4 = ConvBlock(128, 256)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ── Bottleneck ────────────────────────────────────────────────────────
        self.bottleneck = ConvBlock(256, 512)

        # ── Decoder path ──────────────────────────────────────────────────────
        # Upsample doubles spatial dimensions, then concat with skip connection
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(512, 256)  # 256 (up) + 256 (skip) = 512 -> 256

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(256, 128)  # 128 + 128 = 256 -> 128

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128, 64)   # 64 + 64 = 128 -> 64

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(64, 32)    # 32 + 32 = 64 -> 32

        # ── Output layer ──────────────────────────────────────────────────────
        # 1x1 conv to map features to single-channel prediction
        self.output_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, 3, 128, 128).

        Returns:
            Probability map (B, 1, 128, 128), values in [0, 1].
        """
        # ── Encoder ───────────────────────────────────────────────────────────
        e1 = self.enc1(x)           # (B, 32, 128, 128)
        e2 = self.enc2(self.pool(e1))  # (B, 64, 64, 64)
        e3 = self.enc3(self.pool(e2))  # (B, 128, 32, 32)
        e4 = self.enc4(self.pool(e3))  # (B, 256, 16, 16)

        # ── Bottleneck ────────────────────────────────────────────────────────
        b = self.bottleneck(self.pool(e4))  # (B, 512, 8, 8)

        # ── Decoder with skip connections ─────────────────────────────────────
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))   # (B, 256, 16, 16)
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))  # (B, 128, 32, 32)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))  # (B, 64, 64, 64)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # (B, 32, 128, 128)

        # ── Output ────────────────────────────────────────────────────────────
        logits = self.output_conv(d1)  # (B, 1, 128, 128)
        return torch.sigmoid(logits)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load_unet(model_path: str | Path, device: str = "cpu") -> CaptchaUNet:
    """Load a trained U-Net model from disk.

    Args:
        model_path: Path to the .pth file.
        device: Device to load model onto ("cpu" or "cuda").

    Returns:
        CaptchaUNet model in eval mode.
    """
    model = CaptchaUNet()
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    # Quick test: verify architecture and parameter count
    model = CaptchaUNet()
    print(f"CaptchaUNet Parameters: {model.count_parameters():,}")

    # Test forward pass
    dummy_input = torch.randn(1, 3, 128, 128)
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
"""
"""
