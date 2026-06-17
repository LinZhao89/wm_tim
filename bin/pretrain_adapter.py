"""Script to pre-train a feature adapter for domain transfer.

This script trains a small convolutional autoencoder on a given dataset.
The encoder part of this autoencoder can then be used as a "feature adapter"
to transform images from the source domain (e.g., wafermaps) into a feature
space that is more suitable for a backbone pre-trained on a different domain
(e.g., ImageNet).
"""
import logging
import os

import click
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import tqdm

from patchcore.datasets.wm811k import wm811kDataset, DatasetSplit
from patchcore.networks.feature_adapter import AdapterAutoencoder

LOGGER = logging.getLogger(__name__)

@click.command()
@click.argument("data_path", type=click.Path(exists=True, file_okay=False))
@click.argument("subdataset", type=str)
@click.option("--save_path", type=str, default="./adapter_weights.pth", help="Path to save the trained adapter weights.")
@click.option("--epochs", type=int, default=10, help="Number of epochs to train.")
@click.option("--batch_size", type=int, default=32, help="Batch size for training.")
@click.option("--learning_rate", type=float, default=1e-3, help="Learning rate for the optimizer.")
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
def pretrain_adapter(data_path, subdataset, save_path, epochs, batch_size, learning_rate, gpu):
    """
    Pre-trains a feature adapter autoencoder on the given dataset.
    """
    device = torch.device(f"cuda:{gpu[0]}" if torch.cuda.is_available() and gpu else "cpu")
    LOGGER.info(f"Using device: {device}")

    # 1. Create the dataset and dataloader
    # We need a dataset that returns 3-channel images.
    train_dataset = wm811kDataset(
        data_path,
        classname=subdataset,
        split=DatasetSplit.TRAIN,
        imagesize=128,
        apply_filter=False, # Do not apply the filter for pre-training
        grayscale=False, # Ensure we get 3-channel images
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # 2. Initialize the model, loss, and optimizer
    model = AdapterAutoencoder(in_channels=3).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    LOGGER.info("Starting pre-training of the feature adapter...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm.tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            desc=f"Epoch {epoch + 1}/{epochs}",
        )

        for i, batch in progress_bar:
            images = batch["image"].to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, images)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": total_loss / (i + 1)})

    LOGGER.info("Pre-training finished.")

    # 3. Save the encoder part of the model
    LOGGER.info(f"Saving trained adapter encoder to {save_path}")
    torch.save(model.encoder.state_dict(), save_path)
    LOGGER.info("Done.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pretrain_adapter()
