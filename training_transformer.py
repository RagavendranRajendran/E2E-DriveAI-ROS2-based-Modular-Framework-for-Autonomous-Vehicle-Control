import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformer_o1 import Transformer


# We assume you have the Transformer class from the previous snippet
# from my_transformer_file import Transformer

# For demonstration, let's define a small dataset that produces random data
class RandomSeqDataset(Dataset):
    """
    Produces random sequential data:
      - src of shape (25, 1024)  --> 25 timesteps, each 1024-dim
      - tgt of shape (15, 2)     --> 15 timesteps, each 2-dim (speed, steering)
    """

    def __init__(self, length=1000):
        super().__init__()
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Random src [25 x 1024]
        src = torch.randn(25, 1024)
        # Random tgt [15 x 2]
        tgt = torch.randn(15, 2)

        print(f"Source: {src.shape}")
        print(f"Target: {tgt}")

        # In a real dataset, these would be real sequential features & labels
        return src, tgt


# Create the model
model = Transformer(
    input_dim=1024,  # 512 image + 512 pcd = 1024 if flattened
    d_model=512,
    nheads=8,
    num_encoder_layers=2,  # For demo, fewer layers
    num_decoder_layers=2,  # For demo, fewer layers
    dim_feedforward=1024,
    dropout=0.1,
    output_dim=2  # speed + steering
)

# Create train & validation datasets
train_dataset = RandomSeqDataset(length=800)  # 800 samples for training
val_dataset   = RandomSeqDataset(length=200)  # 200 samples for validation

# Create corresponding DataLoaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=8, shuffle=False)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for src, tgt in dataloader:
        # Move data to GPU (if available)
        src = src.to(device)
        tgt = tgt.to(device)

        # (Optional) For training with teacher forcing,
        # you might provide the *shifted* target as the decoder input
        # and the unshifted target as the label. For simplicity,
        # we’ll just feed the entire `tgt` in this demo.

        optimizer.zero_grad()

        # Forward pass
        output = model(src, tgt)  # shape: (batch_size, 15, 2)

        # Compute loss (MSE between prediction and ground truth)
        loss = criterion(output, tgt)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)

            # Forward pass (no gradient needed)
            output = model(src, tgt)

            # Compute validation loss
            loss = criterion(output, tgt)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=10):
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch}/{epochs}] "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f}")


# Example usage
if __name__ == "__main__":
    train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=10)