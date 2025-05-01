import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os

# ------------------------------
# Dataset
# ------------------------------
class DSpritesDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True, mmap_mode='r')
        self.imgs = data['imgs']
        self.labels = data['latents_classes']  # shape, scale, orient, posX, posY

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        img = torch.tensor(self.imgs[idx], dtype=torch.float32).unsqueeze(0) * 2.0 - 1.0
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label

# ------------------------------
# Model
# ------------------------------
class DSpritesFeatureClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),  # 64 → 32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # 32 → 16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),  # 16 → 8
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),  # 8 → 4
            nn.ReLU(),
            nn.Flatten(),
        )
        self.hidden = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU()
        )

        self.shape = nn.Linear(512, 3)
        self.scale = nn.Linear(512, 6)
        self.orient = nn.Linear(512, 40)
        self.posX = nn.Linear(512, 32)
        self.posY = nn.Linear(512, 32)

    def forward(self, x):
        x = self.backbone(x)
        x = self.hidden(x)
        return [
            self.shape(x),
            self.scale(x),
            self.orient(x),
            self.posX(x),
            self.posY(x)
        ]

# ------------------------------
# Training Loop
# ------------------------------
def train_classifier(npz_path, save_path='dsprites_classifier.pt', epochs=5, batch_size=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = DSpritesDataset(npz_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = DSpritesFeatureClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            preds = model(imgs)
            losses = [
                F.cross_entropy(preds[0], labels[:, 1]),  # shape
                F.cross_entropy(preds[1], labels[:, 2]),  # scale
                F.cross_entropy(preds[2], labels[:, 3]),  # orient
                F.cross_entropy(preds[3], labels[:, 4]),  # posX
                F.cross_entropy(preds[4], labels[:, 5]),  # posY
            ]
            loss = sum(losses)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch}/{epochs} | Loss: {running_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), save_path)
    print(f"✅ Saved classifier to {save_path}")


if __name__ == "__main__":
    train_classifier("dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")
