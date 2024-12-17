import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def preprocess_labels(file_path):
    with h5py.File(file_path, 'r') as f:
        labels = f['labels'][:]

        # Normalize Hue Factors (Floor, Wall, Object) to [0.0, 1.0]
        labels[:, 0:3] = labels[:, 0:3] / 0.90  # Rescale Floor, Wall, Object Hue

        # Normalize Object Size to [0.0, 1.0]
        labels[:, 3] = (labels[:, 3] - 0.75) / (1.25 - 0.75)

        return labels


class Shapes3DDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.hdf5_file = None

        # Preprocess labels during initialization
        self.labels = preprocess_labels(file_path)

    def __len__(self):
        with h5py.File(self.file_path, 'r') as f:
            return len(f['images'])

    def __getitem__(self, idx):
        if self.hdf5_file is None:
            self.hdf5_file = h5py.File(self.file_path, 'r')
        image = self.hdf5_file['images'][idx] / 255.0  # Normalize image to [0.0, 1.0]
        label = self.labels[idx]  # Use preprocessed labels
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


file_path = "3dshapes.h5"
dataset = Shapes3DDataset(file_path)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# Verify DataLoader
for images, labels in dataloader:
    print("Batch Images shape:", images.shape)  # Expected: (batch_size, 64, 64, 3)
    print("Batch Labels shape:", labels.shape)  # Expected: (batch_size, 6)
    print("Sample normalized label:", labels[0])
    break
