import torch
from torchvision.utils import save_image
import os
from unsupervised import UNet, LatentEncoder, DDPM


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
z_dim = 10
img_size = 64
num_steps = 7
z_range = (-3, 3)
save_dir = './generated_samples'
os.makedirs(save_dir, exist_ok=True)

# Load model + encoder
model = UNet().to(device)
encoder = LatentEncoder().to(device)
ddpm = DDPM(model, encoder)
checkpoint = torch.load('./unsupervised_checkpoints/ddpm_epoch_20.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# For each latent dim
for i in range(z_dim):
    with torch.no_grad():
        base_z = torch.zeros(num_steps, z_dim).to(device)
        base_z[:, i] = torch.linspace(*z_range, steps=num_steps).to(device)

        images = ddpm.sample_from_z(img_size=img_size, z=base_z)
        images_vis = (images + 1) * 0.5  # [-1,1] → [0,1]

        # Save image grid
        save_image(images_vis, f'{save_dir}/z{i}_traversal.png', nrow=num_steps)
        print(f'Saved traversal for z[{i}] to {save_dir}/z{i}_traversal.png')

        # Classifier readout
        preds = ddpm.classifier(images)
        decoded = [p.argmax(1).cpu().numpy() for p in preds]
        print(f'z[{i}] affects:')
        print('  Shape:', decoded[0])
        print('  Scale:', decoded[1])
        print('  Orientation:', decoded[2])
        print('  posX:', decoded[3])
        print('  posY:', decoded[4])
