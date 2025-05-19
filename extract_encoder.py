import torch
from unsupervised import UNet, LatentEncoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Rebuild model and encoder
model = UNet(in_channels=1, base_channels=32, time_emb_dim=128, cond_emb_dim=128).to(device)
encoder = LatentEncoder().to(device)

# Dummy optimizer (only needed to load optimizer state from checkpoint)
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(encoder.parameters()),
    lr=3e-5
)

# Load checkpoint
ckpt_path = './unsupervised_checkpoints/ddpm_epoch_20.pt'
print(f"Loading checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
optimizer.load_state_dict(ckpt['optimizer_state_dict'])

# Save encoder
encoder_save_path = './unsupervised_checkpoints/encoder_epoch_20.pt'
torch.save(encoder.state_dict(), encoder_save_path)
print(f"✓ Saved encoder to {encoder_save_path}")
