import matplotlib
matplotlib.use('Agg')  # no GUI needed
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math

# Minimal necessary classes
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

class SimpleUNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, time_emb_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.time_proj = nn.Linear(time_emb_dim, base_channels * 2)
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(base_channels*2, base_channels*2, 3, padding=1),
            nn.ReLU()
        )
        self.bot = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels*4, base_channels*2, 3, padding=1),
            nn.ReLU()
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(base_channels*2, base_channels, 4, 2, 1),
            nn.ReLU()
        )
        self.final = nn.Conv2d(base_channels, in_channels, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        t_emb = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        x = self.down1(x)
        x = self.down2(x)
        x = self.bot(x)
        x = x + t_emb
        x = self.up(x)
        return self.final(x)

class DDPM:
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.model = model
        self.device = device
        self.T = timesteps
        self.beta = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alpha = (1. - self.beta).to(device)
        self.alpha_hat = torch.cumprod(self.alpha, dim=0).to(device)

@torch.no_grad()
def sample(ddpm, model, shape=(16, 1, 64, 64), device='cpu'):
    model.eval()
    x = torch.randn(shape, device=device)
    print(f"Starting sample loop: {ddpm.T} timesteps")
    for t in reversed(range(ddpm.T)):
        if t % 100 == 0:
            print(f"Sampling t={t}")
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
        pred_noise = model(x, t_batch)
        alpha = ddpm.alpha[t]
        alpha_hat = ddpm.alpha_hat[t]
        beta = ddpm.beta[t]
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        x = (1 / alpha.sqrt()) * (x - (1 - alpha) / (1 - alpha_hat).sqrt() * pred_noise) + beta.sqrt() * noise
    return x


# --- Run it
device = torch.device('cpu')

model = SimpleUNet().to(device)
model.load_state_dict(torch.load('./ddpm_epoch_5.pt', map_location=device))
model.eval()

ddpm = DDPM(model, device=device)

print("Sampling started...")
samples = sample(ddpm, model, shape=(16, 1, 64, 64), device=device)
print("Sampling finished!")
samples = (samples + 1) / 2
samples = samples.cpu().numpy().squeeze(1)

fig, axes = plt.subplots(4, 4, figsize=(8,8))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(samples[i], cmap='gray')
    ax.axis('off')
plt.tight_layout()
plt.savefig('samples.png')
print("✅ Done! Saved to samples.png")
