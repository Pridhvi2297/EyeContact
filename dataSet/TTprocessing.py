import os
import cv2
import dlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Load eye position data
csv_path = "eye_corner_locations.csv"
df = pd.read_csv(csv_path)

# Initialize Dlib face detector & landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load images
data_dir = "Columbia Gaze Data Set"
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Custom dataset for gaze correction
class GazeDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        gaze_points = self.data.iloc[idx, 1:].values.astype(np.float32)
        
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(gaze_points)

# Load dataset
dataset = GazeDataset(csv_path, data_dir, transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
print(f"Dataset loaded with {len(dataset)} images.")

# Define model
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
        corrected_gaze = self.fc(x_flattened)
        x_decoded = self.decoder(x_encoded)
        return x_decoded, corrected_gaze

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GazeCorrectionModel().to(device)

# Define loss and optimizer
criterion_img = nn.MSELoss()
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
        loss_img = criterion_img(reconstructed_images, images)
        loss_gaze = criterion_gaze(predicted_gaze, gaze_points)
        loss = loss_img + loss_gaze
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# Save trained model
torch.save(model.state_dict(), "gaze_correction.pth")
print("Model training complete.")

# Convert to ONNX
dummy_input = torch.randn(1, 3, 64, 64).to(device)
torch.onnx.export(model, dummy_input, "gaze_correction.onnx", export_params=True)
print("Model converted to ONNX.")
