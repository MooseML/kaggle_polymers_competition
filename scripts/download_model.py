from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model

def download_models():
    """Downloads HOMO-LUMO Gap Backbone model from Hugging Face Hub"""
    
    # download EfficientNetV2S model
    three_d_homo_lumo_model_path = hf_hub_download(repo_id="MooseML/EfficientNet-Cancer-Detection", filename="efficientnet_cancer_model.h5")
    homo_lumo_model = load_model(three_d_homo_lumo_model_path)
    print("HOMO-LUMO Gap prediction model downloaded and loaded!")

    return homo_lumo_model

if __name__ == "__main__":
    download_models()