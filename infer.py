import torch
from unsupervised import LatentTraversalSampler, LatentEncoder, UNet, DDPM

# Load your encoder and DDPM model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = LatentEncoder(z_dim=10).to(device)
encoder.load_state_dict(torch.load('unsupervised_checkpoints/encoder_epoch_20.pt', map_location=device))
encoder.eval()

model = UNet().to(device)
model.load_state_dict(torch.load('unsupervised_checkpoints/ddpm_epoch_20.pt', map_location=device)['model_state_dict'])
model.eval()

# Wrap into DDPM
ddpm = DDPM(model, encoder)

# Use the sampler
sampler = LatentTraversalSampler(ddpm, img_size=64, device=device)
sampler.sample_latent_traversals(
    epoch=20,
    save_dir="./traversals",  # save images here
    num_steps=7,              # number of values per latent dim
    z_range=(-3, 3)           # range of traversal values
)
