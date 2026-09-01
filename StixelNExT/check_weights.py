import torch
import numpy as np
from models.ConvNeXt import ConvNeXt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt_path = "/home/orin-001/sda/agibotnav/src/FAST_LIO/scripts/model_checkpoints/StixelNExT_ground_best.pth"

model = ConvNeXt(in_channels=3, out_channels=2).to(device)
state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(state_dict)

# Check if any parameters are non-zero
print("Checking parameters:")
has_nonzero = False
for name, param in model.named_parameters():
    if param.data.abs().sum() > 0:
        has_nonzero = True
        break
print(f"Any non-zero parameters: {has_nonzero}")

# Check first layer
print(f"\nFirst conv weight: {model.encoder.stem[0].weight.abs().sum():.6f}")
print(f"First conv bias: {model.encoder.stem[0].bias.abs().sum():.6f}")

# Check output layer
print(f"\nHead decoder weight: {model.decoder.decoder.weight.abs().sum():.6f}")
