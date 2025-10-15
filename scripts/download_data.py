import os
import zipfile

# Define dataset path
dataset_path = "/workspace/data/"
zip_file_path = "/workspace/data/histopathologic-cancer-detection.zip"

# Function to download dataset using Kaggle API
def download_kaggle_dataset():
    print("Checking Kaggle API credentials...")

    # Ensure Kaggle API key is accessible
    kaggle_api_path = "/root/.kaggle/kaggle.json"
    if not os.path.exists(kaggle_api_path):
        raise FileNotFoundError("Kaggle API key missing! Please mount 'kaggle.json' in devcontainer.json.")

    # Download dataset
    print("Downloading dataset from Kaggle...")
    os.system("kaggle competitions download -c histopathologic-cancer-detection -p /workspace/data/")

    # Extract dataset
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(dataset_path)

    print("Dataset downloaded and extracted successfully!")

# Run download function
if not os.path.exists("/workspace/data/train/") or not os.path.exists("/workspace/data/test/"):
    download_kaggle_dataset()
else:
    print("Dataset already exists")