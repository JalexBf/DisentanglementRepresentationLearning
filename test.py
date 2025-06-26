import torch

ckpt = torch.load("pretrained_latents.pt", map_location='cpu')

print("=== Classifier keys in saved file ===")
for key in ckpt['classifier'].keys():
    print(key)
