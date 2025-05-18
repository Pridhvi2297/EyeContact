import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
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

# Custom dataset class
class GazeDataset(Dataset):
    def __init__(self, csv_file, img_root_dir, transform=None):
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        if not os.path.exists(img_root_dir):
            raise FileNotFoundError(f"Dataset directory not found: {img_root_dir}")

        self.data = pd.read_csv(csv_file)
        self.img_root_dir = img_root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        img_folder = img_name.split("_")[0]  # Extract the folder name (e.g., "0001")
        img_path = os.path.join(self.img_root_dir, img_folder, img_name)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")
        gaze_points = self.data.iloc[idx, 1:].values.astype(float)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(gaze_points)

# Create dataset and data loader
dataset = GazeDataset(CSV_PATH, DATASET_DIR, transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
print(f"Dataset loaded with {len(dataset)} images.")

# Define the Gaze Correction Model
class GazeCorrectionModel(nn.Module):
    def __init__(self):
        super(GazeCorrectionModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 8)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_encoded = self.encoder(x)
        x_flattened = x_encoded.view(x_encoded.size(0), -1)
        gaze_corrected = self.fc(x_flattened)
        x_decoded = self.decoder(x_encoded)
        return x_decoded, gaze_corrected

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GazeCorrectionModel().to(device)
print("Model loaded on:", device)

# Define loss functions and optimizer
criterion_image = nn.MSELoss()
criterion_gaze = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    epoch_loss = 0
    for images, gaze_points in dataloader:
        images, gaze_points = images.to(device), gaze_points.to(device)
        optimizer.zero_grad()
        reconstructed_images, predicted_gaze = model(images)
        loss_img = criterion_image(reconstructed_images, images)
        loss_gaze = criterion_gaze(predicted_gaze, gaze_points)
        loss = loss_img + loss_gaze
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# Save trained model
torch.save(model.state_dict(), "gaze_correction.pth")
print("Model training complete and saved as gaze_correction.pth")

# Convert model to ONNX
dummy_input = torch.randn(1, 3, 64, 64).to(device)
torch.onnx.export(model, dummy_input, "gaze_correction.onnx", export_params=True)
print("Model converted to gaze_correction.onnx")
