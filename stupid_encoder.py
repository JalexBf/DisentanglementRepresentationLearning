import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

class DSpritesLazyDataset(Dataset):
    def __init__(self, npz_path):
        self.data = np.load(npz_path, allow_pickle=True, mmap_mode='r')
        self.imgs = self.data['imgs']
        self.latents_classes = self.data['latents_classes']

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        img = self.imgs[idx].astype(np.float32)
        img = torch.from_numpy(img).unsqueeze(0)
        img = (img > 0.5).float()
        img = img * 2.0 - 1.0
        classes = self.latents_classes[idx]
        shape = classes[1]
        scale = classes[2]
        angle = classes[3]
        pos_x = classes[4]
        pos_y = classes[5]
        labels = torch.tensor([shape, scale, angle, pos_x, pos_y], dtype=torch.float32)
        return img, labels

class LatentEncoder(nn.Module):
    def __init__(self, z_dim=5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, z_dim)
        )

    def forward(self, x, normalize=True):
        z = self.encoder(x)
        if normalize:
            z = (z - z.mean(dim=0, keepdim=True)) / (z.std(dim=0, keepdim=True) + 1e-5)
        return z


class LatentClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.shape_head   = nn.Linear(1, 3)    # z[0]
        self.scale_head   = nn.Linear(2, 6)    # z[1]
        self.orient_head  = nn.Linear(4, 40)   # z[2]
        self.pos_x_head   = nn.Linear(2, 32)   # z[3]
        self.pos_y_head   = nn.Linear(2, 32)   # z[4]

    def forward(self, z):
        return [
            self.shape_head(z[:, 0:1]),
            self.scale_head(z[:, 1:3]),
            self.orient_head(z[:, 3:7]),
            self.pos_x_head(z[:, 7:9]),
            self.pos_y_head(z[:, 9:11]),
        ]



def train_encoder_classifier(device='cuda', save_path='pretrained.pt'):
    dataset = DSpritesLazyDataset('dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=2)

    encoder = LatentEncoder(z_dim=11).to(device)
    classifier = LatentClassifier().to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()), lr=1e-4
    )

    head_names = ['shape', 'scale', 'rotation', 'pos_x', 'pos_y']
    weights = [0.0001, 2.0, 15.0, 0.5, 0.5]

    num_epochs = 100
    for epoch in range(1, num_epochs + 1):
        encoder.train()
        classifier.train()

        total_loss = 0
        correct = [0] * 5
        total = 0

        for img, labels in dataloader:
            img, labels = img.to(device), labels.to(device)
            z = encoder(img)
            preds = classifier(z)

            loss_terms = [
                weights[0] * F.cross_entropy(preds[0], labels[:, 0].long()),
                weights[1] * F.cross_entropy(preds[1], labels[:, 1].long()),
                weights[2] * F.cross_entropy(preds[2], labels[:, 2].long()),
                weights[3] * F.cross_entropy(preds[3], labels[:, 3].long()),
                weights[4] * F.cross_entropy(preds[4], labels[:, 4].long()),
            ]

            loss = sum(loss_terms) / sum(weights)
            total_loss += loss.item() * img.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for i, (pred, true) in enumerate(zip(preds, labels.T)):
                if i == 2:
                    continue
                correct[i] += (pred.argmax(dim=1) == true.long()).sum().item()
            total += img.size(0)

        rot_acc = (preds[2].argmax(dim=1) == labels[:, 2].long()).float().mean().item()
        accs = [c / total for c in correct]
        accs[2] = rot_acc

        avg_acc = sum(accs) / len(accs)
        print(f"Epoch {epoch} | Loss: {total_loss / total:.4f} | Accs: {[f'{a:.3f}' for a in accs]} | Avg: {avg_acc:.3f}")

        if all(a >= 0.9 for a in accs):
            print(f"Early stopping: All classification heads ≥ 0.9")
            torch.save({
                'encoder': encoder.state_dict(),
                'classifier': classifier.state_dict()
            }, save_path)
            return

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'encoder': encoder.state_dict(),
                'classifier': classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, f"checkpoint_epoch_{epoch}.pt")

        print(f"[DEBUG] Saving model at epoch {epoch} | Accs: {[round(a,3) for a in accs]}")


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_encoder_classifier(device=device)
