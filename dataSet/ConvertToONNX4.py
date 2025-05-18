import torch
from GazeCorrectionModel2 import GazeCorrectionModel, device

# Load the trained model
model = GazeCorrectionModel().to(device)
model.load_state_dict(torch.load("best_gaze_correction_model.pth")['model_state_dict'])
model.eval()

# Create dummy input with correct dimensions (batch_size, channels, height, width)
dummy_input = torch.randn(1, 3, 64, 64, dtype=torch.float32).to(device)

# Export to ONNX
torch.onnx.export(
    model,                     # Model being exported
    dummy_input,              # Dummy input
    "gaze_correction.onnx",   # Output file name
    export_params=True,       # Store trained parameters
    opset_version=11,         # ONNX version
    do_constant_folding=True, # Optimize constant folding
    input_names=['input'],    # Input names
    output_names=['reconstructed_image', 'gaze_prediction'], # Output names
    dynamic_axes={
        'input': {0: 'batch_size'},
        'reconstructed_image': {0: 'batch_size'},
        'gaze_prediction': {0: 'batch_size'}
    }
)

print("Model successfully converted to gaze_correction.onnx")
