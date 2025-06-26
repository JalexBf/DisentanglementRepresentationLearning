import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os

from classifier import DSpritesDataset, DSpritesFeatureClassifier



@torch.no_grad()
def evaluate_classifier(model, dataloader, device):
    model.eval()
    correct = [0] * 5
    total = 0

    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs)

        for i in range(5):
            pred_labels = preds[i].argmax(dim=1)
            correct[i] += (pred_labels == labels[:, i + 1]).sum().item()

        total += imgs.size(0)

    names = ['shape', 'scale', 'orient', 'posX', 'posY']
    for i in range(5):
        acc = correct[i] / total
        print(f"{names[i]} accuracy: {acc * 100:.2f}%")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = DSpritesDataset("dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False)

    # Load model from file
    model = DSpritesFeatureClassifier().to(device)
    model.load_state_dict(torch.load("dsprites_classifier.pt", map_location=device))

    # Evaluate it
    evaluate_classifier(model, dataloader, device)
