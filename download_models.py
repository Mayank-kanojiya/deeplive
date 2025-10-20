import os
import urllib.request
from tqdm import tqdm

def download_file(url, filepath):
    """Download file with progress bar"""
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, (downloaded * 100) // total_size)
            print(f'\rDownloading: {percent}% [{downloaded}/{total_size} bytes]', end='')
    
    urllib.request.urlretrieve(url, filepath, progress_hook)
    print()

def download_face_swap_models():
    """Download all available face swap models"""
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    models = {
        'inswapper_128_fp16.onnx': 'https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx',
        'simswap_256_fp16.onnx': 'https://huggingface.co/netrunner-exe/SimSwap-models/resolve/1afe43249c4d4b5d856cdd1a3708edf43fa830fd/simswap_256.onnx',
        'ghost_256_fp16.onnx': 'https://huggingface.co/hacksider/deep-live-cam/resolve/main/ghost_256_fp16.onnx',
        'hyperswap_128_fp16.onnx': 'https://huggingface.co/hacksider/deep-live-cam/resolve/main/hyperswap_128_fp16.onnx'
    }
    
    print("Downloading face swap models...")
    for model_name, url in models.items():
        model_path = os.path.join(models_dir, model_name)
        if not os.path.exists(model_path):
            try:
                print(f'\nDownloading {model_name}...')
                download_file(url, model_path)
                print(f'{model_name} downloaded successfully!')
            except Exception as e:
                print(f'\nFailed to download {model_name}: {e}')
        else:
            print(f'{model_name} already exists')

if __name__ == '__main__':
    download_face_swap_models()