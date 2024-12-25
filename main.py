from CustomDataset import CustomDataset
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

masterinfo_path = "/home/t14-3/Documents/readtest/masterinfo.csv"  # Replace with the actual path to your CSV file
config_path = "/home/t14-3/Documents/readtest/config.json"

# Initialize the dataset
dataset = CustomDataset(masterinfo_path, config_path)

def custom_collate_fn(batch):
    # Separate `x` and `y` from the batch
    imgs = [item[0]["Image"] for item in batch]  # Extract `img`
    pcds = [item[0]["PCD"] for item in batch]  # Extract `pcd`
    ys = [item[1] for item in batch]  # Extract `y`

    # Stack `imgs` (fixed-size tensors)
    imgs = torch.stack(imgs)

    # `pcds` remains as a list (variable-size arrays)
    # Stack `ys` (fixed-size tensors for `Speed` and `Steer`)
    speeds = torch.stack([y["Speed"] for y in ys])
    steers = torch.stack([y["Steer"] for y in ys])

    # Combine `Speed` and `Steer` into a single tensor if required


    return {"Image": imgs, "PCD": pcds, "Speed": speeds, "Steer": steers}

dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, collate_fn=custom_collate_fn)

print("Inspecting batches:")
for batch_idx, batch in enumerate(dataloader):
    print("*" * 80)
    print(f"Batch {batch_idx + 1}: ")
    img = batch["Image"]
    pcd = batch["PCD"]
    spd = batch["Speed"]
    stw = batch["Steer"]

    print(f"Img shape: {img.shape}")

    print(f"Speed shape: {spd.shape}")
    print(f"Steer shape: {stw.shape}")

    print(f"PCD shape: {len(pcd)}")
    for p in pcd:
        print(f"number of point rows: {len(p)}")
        ppp = []
        for pp in p:
            ppp.append(len(pp))
        print(f"No. of points in each row: {ppp}")

    print("--" * 40)

    if batch_idx == 2:
        break