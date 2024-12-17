import h5py
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch


factor_names = ["Floor Hue", "Wall Hue", "Object Hue", "Object Size", "Object Shape", "Camera Elevation"]
factor_ranges = {
    "Floor Hue": (0.0, 1.0),
    "Wall Hue": (0.0, 1.0),
    "Object Hue": (0.0, 1.0),
    "Object Size": (0.0, 1.0),
    "Object Shape": (0, 3),  # Discrete
    "Camera Elevation": (-30.0, 30.0),
}

file_path = "3dshapes.h5"


def check_dataset_shape(file_path):
    with h5py.File(file_path, 'r') as f:
        images = f['images']
        labels = f['labels']
        print("Images shape:", images.shape)  # Expected: (480000, 64, 64, 3)
        print("Labels shape:", labels.shape)  # Expected: (480000, 6)
        assert images.shape == (480000, 64, 64, 3), "Unexpected image shape!"
        assert labels.shape == (480000, 6), "Unexpected label shape!"

check_dataset_shape(file_path)


def verify_factor_ranges(file_path):
    with h5py.File(file_path, 'r') as f:
        labels = f['labels'][:]
        for i, factor_name in enumerate(factor_names):
            actual_min = labels[:, i].min()
            actual_max = labels[:, i].max()
            print(f"{factor_name}: Actual range [{actual_min}, {actual_max}], Expected range {factor_ranges[factor_name]}")
            assert actual_min == factor_ranges[factor_name][0], f"{factor_name} min does not match!"
            assert actual_max == factor_ranges[factor_name][1], f"{factor_name} max does not match!"

verify_factor_ranges(file_path)


def visualize_samples(file_path, num_samples=5):
    with h5py.File(file_path, 'r') as f:
        images = f['images']
        labels = f['labels']
        for idx in range(num_samples):
            plt.imshow(images[idx])
            plt.title("\n".join([f"{factor_names[i]}: {labels[idx, i]}" for i in range(len(factor_names))]), fontsize=8)
            plt.axis('off')
            plt.show()

visualize_samples(file_path)


class Shapes3DDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.hdf5_file = None

    def __len__(self):
        with h5py.File(self.file_path, 'r') as f:
            return len(f['images'])

    def __getitem__(self, idx):
        if self.hdf5_file is None:
            self.hdf5_file = h5py.File(self.file_path, 'r')
        image = self.hdf5_file['images'][idx] / 255.0  # Normalize images
        label = self.hdf5_file['labels'][idx]
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


dataset = Shapes3DDataset(file_path)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

def validate_dataloader(dataloader):
    for images, labels in dataloader:
        print("Batch Images shape:", images.shape)  # Expected: (batch_size, 3, 64, 64)
        print("Batch Labels shape:", labels.shape)  # Expected: (batch_size, 6)
        print("First batch labels:", labels)
        break

validate_dataloader(dataloader)
