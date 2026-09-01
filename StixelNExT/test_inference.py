#!/usr/bin/env python3
"""
Simple inference script to test StixelNExT model on a single image.
"""
import torch
import cv2
import numpy as np
import os
import sys
from models.ConvNeXt import ConvNeXt

def preprocess_image(img_path, target_size=(1248, 376)):
    """Load and preprocess image for model input."""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")

    original_h, original_w = img.shape[:2]

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, target_size)

    # Convert to tensor: [H, W, C] -> [C, H, W]
    # Keep values in [0, 255] range (same as training data)
    img_tensor = torch.from_numpy(img_resized).float()
    img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW

    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor, (original_h, original_w)

def postprocess_prediction(prediction, original_size, target_size=(1248, 376)):
    """Convert model output to binary mask."""
    prediction = prediction.squeeze(0)  # Remove batch dim

    # Get channel 0 (ground occupancy - higher = more likely ground)
    # Shape: [2, 94, 312] where 94=376/4, 312=1248/4
    ground_map = prediction[0].detach().cpu().numpy()

    # Apply sigmoid to get probabilities (model outputs raw logits)
    ground_prob = 1 / (1 + np.exp(-ground_map))

    # Resize to target image size
    ground_prob = cv2.resize(ground_prob, target_size)

    # Create binary mask (threshold at 0.5)
    binary_mask = (ground_prob > 0.5).astype(np.uint8) * 255

    # Resize to original image size
    original_h, original_w = original_size
    binary_mask = cv2.resize(binary_mask, (original_w, original_h))

    return binary_mask, ground_prob

def create_overlay(img, mask):
    """Create overlay image with green ground mask."""
    overlay = img.copy()

    # Create green overlay where mask is 255
    green_mask = np.zeros_like(img)
    green_mask[mask > 127] = [0, 255, 0]  # BGR green

    # Blend
    result = cv2.addWeighted(img, 0.6, green_mask, 0.4, 0)

    return result

def main():
    # Configuration
    model_path = "best_model_weights/StixelNExT_ground_best.pth"
    img_path = "test_image.jpg"
    output_mask_path = "test_image_mask.png"
    output_overlay_path = "test_image_overlay.jpg"

    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Error: Model not found: {model_path}")
        sys.exit(1)

    if not os.path.exists(img_path):
        print(f"Error: Image not found: {img_path}")
        sys.exit(1)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    model = ConvNeXt(
        stem_features=64,
        depths=[6, 3],
        widths=[96, 192, 384, 768],
        drop_p=0.1,
        target_height=94,   # 376 / 4
        target_width=312,   # 1248 / 4
        out_channels=2
    ).to(device)

    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()
    print(f"Model loaded from: {model_path}")

    # Load and preprocess image
    print(f"Processing image: {img_path}")
    img_tensor, original_size = preprocess_image(img_path)
    img_tensor = img_tensor.to(device)

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        prediction = model(img_tensor)

    # Postprocess
    binary_mask, ground_map = postprocess_prediction(prediction, original_size)

    # Save binary mask
    cv2.imwrite(output_mask_path, binary_mask)
    print(f"Binary mask saved: {output_mask_path}")

    # Create and save overlay
    original_img = cv2.imread(img_path)
    overlay_img = create_overlay(original_img, binary_mask)
    cv2.imwrite(output_overlay_path, overlay_img)
    print(f"Overlay image saved: {output_overlay_path}")

    # Print statistics
    ground_ratio = np.sum(binary_mask > 127) / binary_mask.size
    print(f"\nGround pixel ratio: {ground_ratio:.1%}")
    print(f"Ground probability range: [{ground_map.min():.3f}, {ground_map.max():.3f}]")
    print(f"Ground probability mean: {ground_map.mean():.3f}")

    print("\nDone!")

if __name__ == "__main__":
    main()
