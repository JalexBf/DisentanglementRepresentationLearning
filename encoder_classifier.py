import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from unsupervised import DSpritesLazyDataset, LatentEncoder, LatentClassifier


z_dim = 30
batch_size = 64
epochs = 20
lr = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


dataset = DSpritesLazyDataset('dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


encoder = LatentEncoder(z_dim=z_dim).to(device)
classifier = LatentClassifier(z_dim=z_dim).to(device)

optimizer = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=lr)


for epoch in range(1, epochs + 1):
    total_loss = 0.0
    accs = np.zeros(5)

    for imgs, latents in dataloader:
        imgs = imgs.to(device)
        latents = latents.to(device)

        z = encoder(imgs)
        preds = classifier(z)

        losses = [
            F.cross_entropy(preds[0], latents[:, 1]),
            F.cross_entropy(preds[1], latents[:, 2]),
            F.cross_entropy(preds[2], latents[:, 3]),
            F.cross_entropy(preds[3], latents[:, 4]),
            F.cross_entropy(preds[4], latents[:, 5]),
        ]
        loss = sum(losses)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        with torch.no_grad():
            for j in range(5):
                pred_labels = preds[j].argmax(dim=1)
                accs[j] += (pred_labels == latents[:, j + 1]).sum().item()

    avg_accs = accs / len(dataset)
    print(f"Epoch {epoch} | Loss: {total_loss:.4f} | Accuracies: shape={avg_accs[0]:.2f}, scale={avg_accs[1]:.2f}, orient={avg_accs[2]:.2f}, posX={avg_accs[3]:.2f}, posY={avg_accs[4]:.2f}")


torch.save(encoder.state_dict(), "pretrained_encoder.pt")
torch.save(classifier.state_dict(), "pretrained_classifier.pt")
print("Saved encoder and classifier.")
