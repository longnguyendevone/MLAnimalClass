import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_data_loaders(data_dir, batch_size=32):
    print(f"Loading data from {data_dir}...")
    
    # Standard ImageNet normalization
    stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])
    
    full_ds = datasets.ImageFolder(root=data_dir, transform=train_tf)
    classes = full_ds.classes
    
    train_len = int(0.8 * len(full_ds))
    val_len = len(full_ds) - train_len
    train_ds, val_ds = random_split(full_ds, [train_len, val_len])
    
    # Apply correct transforms (Hack for Subset)
    val_ds.dataset.transform = val_tf 

    loaders = {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
        'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    }
    return loaders, classes