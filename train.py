import os

DataSample = dict[str, str]

def collect_data(images_dir : str, labels_dir : str) -> list[DataSample]:
    images = sorted(os.listdir(images_dir))
    labels = sorted(os.listdir(labels_dir))
    
    data = []
    for image, label in zip(images, labels):
        data.append({"image": os.path.join(images_dir, image), "label": os.path.join(labels_dir, label)})
    
    return data

def collect_dataset(train_dir : str, val_dir : str, test_dir : str) -> dict[str, list[DataSample]]:
    dataset = {"train": [], "val": [], "test": []}
    
    dataset["train"] = collect_data(os.path.join(train_dir, "images"),
                                 os.path.join(train_dir, "labels"))
    dataset["val"] = collect_data(os.path.join(val_dir, "images"),
                                 os.path.join(val_dir, "labels"))
    dataset["test"] = collect_data(os.path.join(test_dir, "images"),
                                 os.path.join(test_dir, "labels"))
    return dataset


ROOT_DIR = "./"
DATASET_PATH = "./data/datasets/astrocyte-1.1"
os.chdir(ROOT_DIR)

dataset = collect_dataset(os.path.join(DATASET_PATH, "train"),
                          os.path.join(DATASET_PATH, "val"),
                          os.path.join(DATASET_PATH, "test"))


from monai.data.dataset import Dataset
from monai.data.dataloader import DataLoader
from monai.data import NumpyReader

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped, Spacingd, 
    RandFlipd, RandAffined, RandGaussianNoised, RandGaussianSmoothd, 
    NormalizeIntensityd, RandAdjustContrastd, Rand3DElasticd
)

train_transforms = Compose([
    LoadImaged(keys=["image", "label"], reader=NumpyReader),
    EnsureTyped(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["label"]),
    EnsureChannelFirstd(keys=["image"], channel_dim=-1),
    
    # Spacing
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
    RandAffined(
        keys=["image", "label"], 
        prob=0.7,
        rotate_range=(0.1, 0.1, 0.1), 
        scale_range=(0.1, 0.1, 0.1), 
        translate_range=(5, 5, 5), 
        mode=("bilinear", "nearest")
    ),
    Rand3DElasticd(keys=["image", "label"], prob=0.2, sigma_range=(5, 8), magnitude_range=(100, 200)),

    # Intensity
    NormalizeIntensityd(keys=["image"], channel_wise=True),  # (data - mean) / std
    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.5)),
    RandGaussianNoised(keys=["image"], prob=0.15, mean=0, std=0.05),
    RandGaussianSmoothd(keys=["image"], prob=0.1, sigma_x=(0.5, 1.5)),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"], reader=NumpyReader),
    EnsureTyped(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["label"]),
    EnsureChannelFirstd(keys=["image"], channel_dim=-1),
    
    # Intensity
    NormalizeIntensityd(keys=["image"], channel_wise=True),  # (data - mean) / std
])

train_dataset = Dataset(dataset["train"], train_transforms)
train_dataloader = DataLoader(train_dataset, batch_size=4)

val_dataset = Dataset(dataset["val"], val_transforms)
val_dataloader = DataLoader(val_dataset, batch_size=4)

test_dataset = Dataset(dataset["test"], val_transforms)
test_dataloader = DataLoader(test_dataset)

import torch
from monai.networks.nets import AttentionUnet
from monai.losses import DiceLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AttentionUnet(
    spatial_dims=2,
    in_channels=3,
    out_channels=1,
    channels=(64, 128, 256, 512, 1024),
    strides=(2, 2, 2, 2),
)
model = model.to(device)

loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, sigmoid=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

class Dice():
    def __init__ (self, smooth=1e-6):
        self.smooth = smooth
        self.by_classes = torch.tensor([])
        
    def __call__(self, outputs : torch.Tensor, targets : torch.Tensor):
        self.intersection = (outputs & targets).sum((2, 3))
        self.union = (outputs | targets).sum((2, 3))
        self.by_classes = (2 * self.intersection) / (self.union + self.intersection + self.smooth)
        return self.by_classes.mean(dim=0)
    
    def mean(self):
        return self.by_classes.mean(dim=1).mean(dim=0)

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(os.path.join("runs", "v.02"))

def write_logs(
    writer : SummaryWriter,
    train_loss : float,
    train_dice : float,
    val_loss : float,
    val_dice : float,
    step : int,
):
    # train
    writer.add_scalar("train/loss", train_loss, step)
    writer.add_scalar("train/mean_dice", train_dice, step)
    
    # validation
    writer.add_scalar("val/loss", val_loss, step)
    writer.add_scalar("val/mean_dice", val_dice, step)

    # compare
    writer.add_scalars("compare/loss", {"train": train_loss, "val": val_loss}, step)
    writer.add_scalars("compare/mean_dice", {"train": train_dice, "val": val_dice}, step)

from torchvision.utils import make_grid
from torchvision.utils import make_grid
def make_grid_image(images : list[torch.Tensor]):
    for i in range(len(images)):
        images[i] = torch.Tensor(images[i])[0].cpu()
        if images[i].size(0) == 1:
            images[i] = images[i].repeat(3, 1, 1)

    return make_grid(images)

from tqdm import tqdm

epochs = 100
dice = Dice()
best_dice = 0

for epoch in range(epochs):
    model.train()
    train_loss = 0
    train_mean_dice = 0
    for batch in tqdm(train_dataloader):
        batch  : dict[str, torch.Tensor]
        inputs, targets = batch["image"].to(device), batch["label"].to(device)
        
        outputs = model(inputs); outputs : torch.Tensor
        loss = loss_function(outputs, targets); loss : torch.Tensor
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        
        mask = (torch.sigmoid(outputs) > 0.3).int()
        targets = targets.int()
        
        dice(mask, targets); mean_dice = dice.mean()
        train_mean_dice += mean_dice
    
    val_loss = 0
    val_mean_dice = 0
    with torch.no_grad():
        model.eval()
        for batch in tqdm(val_dataloader):
            batch : dict[str, torch.Tensor]
            inputs, targets = batch["image"].to(device), batch["label"].to(device)

            outputs = model(inputs); outputs : torch.Tensor
            loss = loss_function(outputs, targets); loss : torch.Tensor
            val_loss += loss.item()
            
            mask = (torch.sigmoid(outputs) > 0.3).int()
            targets = targets.int()
            
            dice(mask, targets); mean_dice = dice.mean()
            val_mean_dice += mean_dice
    
            last_batch = batch
            last_batch["pred"] = outputs

    train_loss /= len(train_dataloader)
    train_mean_dice /= len(train_dataloader)
    val_loss /= len(val_dataloader)
    val_mean_dice /= len(val_dataloader)
    print(f"{epoch+1}/{epochs}: train_loss={train_loss:.5}, train_mean_dice={train_mean_dice:.5}")
    print(f"{epoch+1}/{epochs}: val_loss={val_loss:.5}, val_mean_dice={val_mean_dice:.5}")
    
    if best_dice < val_mean_dice:
        best_dice = val_mean_dice
        torch.save(model.state_dict(), os.path.join("checkpoints", "best.pth"))
    else:
        print(f"No improvement in val mean dice, skip saving checkpoint best_dice={best_dice:.5}")
    write_logs(writer, train_loss, train_mean_dice, val_loss, val_mean_dice, epoch+1)
    grid_image = make_grid_image([inputs, targets, outputs])
    writer.add_image("val/image_grid", grid_image, epoch+1)