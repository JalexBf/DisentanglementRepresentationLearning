import torch
import numpy as np
from torchvision.utils import make_grid, save_image
from torch.utils.data import Dataset, DataLoader
from classifier import DSpritesFeatureClassifier


class DSpritesRealDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True, mmap_mode='r')
        self.imgs = data['imgs']
        self.labels = data['latents_classes']  # shape, scale, orient, posX, posY

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = torch.tensor(self.imgs[idx], dtype=torch.float32).unsqueeze(0) * 2.0 - 1.0
        label = torch.tensor(self.labels[idx][1:], dtype=torch.long)  # only [shape, scale, ...]
        return img, label


@torch.no_grad()
def match_and_save(generated_tensor, classifier, real_dataset, device, save_path="matched_pairs.png"):
    classifier.eval()
    real_imgs = real_dataset.imgs
    real_labels = real_dataset.labels[:, 1:]  # only shape, scale, orient, posX, posY

    matched_pairs = []

    for gen_img in generated_tensor[:16]:  # just 16 for visualization
        img = gen_img.unsqueeze(0).to(device)
        preds = classifier(img)
        pred_labels = [p.argmax(dim=1).item() for p in preds]  # predicted shape, scale, ...

        # Find matching real sample
        matches = np.where(np.all(real_labels == pred_labels, axis=1))[0]

        if len(matches) > 0:
            match_idx = matches[0]
            real_img = torch.tensor(real_imgs[match_idx], dtype=torch.float32).unsqueeze(0) * 2.0 - 1.0
        else:
            real_img = torch.zeros((1, 64, 64))  # fallback if no match found

        matched_pairs.append(gen_img)
        matched_pairs.append(real_img)

    # Normalize each image from [-1, 1] → [0, 1] manually
    matched_pairs = [(img + 1) * 0.5 for img in matched_pairs]
    grid = make_grid(matched_pairs, nrow=2, padding=2)

    save_image(grid, save_path)
    print(f"Saved matched image pairs to {save_path}")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load generated images
generated = torch.load("ddpm_generated_dsprites.pt")  # shape: (10000, 1, 64, 64)

# Load classifier
clf = DSpritesFeatureClassifier().to(device)
clf.load_state_dict(torch.load("dsprites_classifier.pt", map_location=device))

# Load real dataset
real_data = DSpritesRealDataset("dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")

# Match and save
match_and_save(generated, clf, real_data, device)

