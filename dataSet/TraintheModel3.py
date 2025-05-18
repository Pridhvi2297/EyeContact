import torch
import torch.nn as nn
import torch.optim as optim
from Preprocessing1 import get_data_loaders
from GazeCorrectionModel2 import GazeCorrectionModel, device

# Set default dtype
torch.set_default_dtype(torch.float32)

# Get train and validation dataloaders
train_loader, val_loader = get_data_loaders(batch_size=32)

# Initialize model
model = GazeCorrectionModel().to(device).float()

# Define loss functions with weights
criterion_image = nn.MSELoss()  
criterion_gaze = nn.MSELoss()   
lambda_gaze = 0.7  # Weight for gaze loss
lambda_image = 0.3  # Weight for image reconstruction loss

# Optimizer with learning rate scheduler
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1)

# Training loop
num_epochs = 20
best_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    batch_count = 0
    
    for batch_idx, (images, gaze_points) in enumerate(train_loader):
        images, gaze_points = images.to(device), gaze_points.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        reconstructed_images, predicted_gaze = model(images)
        
        # Compute weighted losses
        loss_img = lambda_image * criterion_image(reconstructed_images, images)
        loss_gaze = lambda_gaze * criterion_gaze(predicted_gaze, gaze_points)
        total_loss = loss_img + loss_gaze

        # Backward pass
        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item()
        batch_count += 1

        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], "
                  f"Loss: {total_loss.item():.4f}, Image Loss: {loss_img.item():.4f}, "
                  f"Gaze Loss: {loss_gaze.item():.4f}")

    avg_epoch_loss = epoch_loss / batch_count
    scheduler.step(avg_epoch_loss)

    # Save best model
    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_loss,
        }, "best_gaze_correction_model.pth")

    print(f"Epoch [{epoch+1}/{num_epochs}] Complete, Average Loss: {avg_epoch_loss:.4f}")

print("Training completed successfully!")
