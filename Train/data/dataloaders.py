from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def create_dataloaders(train_dir, val_dir, train_transform, val_transform, batch_size=32):
    train_dataset = ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = ImageFolder(root=val_dir, transform=val_transform)
    print(f"Number of images in train dataset: {len(train_dataset)}")
    print(f"Number of images in validation dataset: {len(val_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader, train_dataset, val_dataset
