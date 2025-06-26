import torch
from Simple import UNet, DDPM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Function to strip `_orig_mod.` from keys ===
def clean_state_dict(state_dict):
    return {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

# Load model and checkpoint
model = UNet(base_channels=64).to(device)

ckpt = torch.load('./simple_checkpoints/ddpm_epoch_45.pt', map_location=device)
cleaned_state = clean_state_dict(ckpt['ema_state_dict'])  
model.load_state_dict(cleaned_state)  

model.eval()  # no need to compile for inference

ddpm = DDPM(model)

# === Sampling Parameters ===
total = 10000  
batch_size = 256
img_size = 64
all_imgs = []

while len(all_imgs) * batch_size < total:
    print(f"Generating batch {len(all_imgs) + 1}")
    batch = ddpm.sample(img_size=img_size, batch_size=batch_size)  # (B, 1, 64, 64)
    all_imgs.append(batch.cpu())

samples = torch.cat(all_imgs, dim=0)[:total]  # (10000, 1, 64, 64)
binarized = (samples > 0).float()  # Optional binarization

torch.save(binarized, "ddpm_generated_dsprites.pt")
print("Saved 10k generated images.")

# === For evaluation (if needed) ===
from torch.utils.data import TensorDataset, DataLoader

data = torch.load("ddpm_generated_dsprites.pt")
labels = torch.zeros(len(data)).long()  # Dummy labels

dataset = TensorDataset(data, labels)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

# loader is now ready for classifier
