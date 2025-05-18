import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Get the absolute path of the script's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define dataset paths using absolute paths
CSV_PATH = os.path.join(BASE_DIR, "eye_corner_locations.csv")
DATASET_DIR = os.path.join(BASE_DIR, "Columbia Gaze Data Set")

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

class GazeDataset(Dataset):
    def __init__(self, csv_file, img_root_dir, transform=None):
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        if not os.path.exists(img_root_dir):
            raise FileNotFoundError(f"Dataset directory not found: {img_root_dir}")

        self.data = pd.read_csv(csv_file)
        self.img_root_dir = img_root_dir
        self.transform = transform
        self._length = len(self.data)

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        img_folder = img_name.split("_")[0]
        img_path = os.path.join(self.img_root_dir, img_folder, img_name + ".jpg")

        if not os.path.exists(img_path):
            print(f"Attempting to load image from: {img_path}")
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")
        gaze_points = self.data.iloc[idx, 1:].values.astype(float)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(gaze_points, dtype=torch.float32)

def get_data_loaders(batch_size=32, train_split=0.8):
    dataset = GazeDataset(CSV_PATH, DATASET_DIR, transform)
    
    # Calculate split sizes
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    # Split dataset
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Dataset loaded with {len(dataset)} images")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader

if __name__ == "__main__":
    train_loader, val_loader = get_data_loaders()
