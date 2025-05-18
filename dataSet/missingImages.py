import os
import pandas as pd

# Get the absolute path of the script's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define dataset paths using absolute paths
CSV_PATH = os.path.join(BASE_DIR, "eye_corner_locations.csv")
DATASET_DIR = os.path.join(BASE_DIR, "Columbia Gaze Data Set")

# Check if CSV file exists
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

# Check if dataset directory exists
if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")

# Load CSV file
df = pd.read_csv(CSV_PATH)

missing_files = []
possible_extensions = [".jpg", ".jpeg", ".png"]

for idx in range(len(df)):
    img_name = df.iloc[idx, 0].strip()  # Remove leading/trailing spaces
    img_folder = img_name.split("_")[0]  # Extract folder name
    
    # Check for multiple possible extensions
    img_path = None
    for ext in possible_extensions:
        temp_path = os.path.join(DATASET_DIR, img_folder, img_name + ext)
        if os.path.exists(temp_path):
            img_path = temp_path
            break
    
    if img_path is None:  # Image not found
        missing_files.append(os.path.join(DATASET_DIR, img_folder, img_name))

print(f"Missing {len(missing_files)} images:", missing_files[:10000])