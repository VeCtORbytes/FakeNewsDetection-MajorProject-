import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python-models'))

# Download model from Hugging Face if not exists
def download_model_from_hf():
    """Download model from Hugging Face"""
    import torch
    
    MODEL_URL = os.getenv('MODEL_URL')
    CHECKPOINT_PATH = os.getenv('CHECKPOINT_PATH', './python-models/ensemble_best.pt')
    
    if not MODEL_URL:
        print("⚠️ MODEL_URL not set in .env, skipping download")
        return
    
    MODEL_PATH = Path(CHECKPOINT_PATH)
    
    if MODEL_PATH.exists():
        print(f"✅ Model already exists at {MODEL_PATH}")
        return
    
    print(f"⬇️ Downloading model from Hugging Face...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        torch.hub.load_state_dict_from_url(
            MODEL_URL,
            model_dir=str(MODEL_PATH.parent),
            file_name=MODEL_PATH.name,
            progress=True
        )
        print(f"✅ Model downloaded successfully to {MODEL_PATH}!")
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        raise

# Download model before importing inference
download_model_from_hf()

from inference import app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
