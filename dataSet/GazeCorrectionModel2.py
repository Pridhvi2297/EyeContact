import torch
import torch.nn as nn
import torch.optim as optim

class GazeCorrectionModel(nn.Module):
    def __init__(self):
        super(GazeCorrectionModel, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Fully connected layers for gaze direction prediction
        self.fc = nn.Sequential(
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 8)  # Predict 8 eye corner positions
        )

        # Decoder (Image Reconstruction)
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
        
        return x_decoded, gaze_corrected  # Return reconstructed image and corrected gaze

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GazeCorrectionModel().to(device)
print("Model loaded on:", device)
