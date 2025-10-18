import cv2
import numpy as np
import threading
from typing import Any
from modules.typing import Frame
from modules.core import update_status
import modules.globals
import os

try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    REALESRGAN_AVAILABLE = True
except ImportError:
    REALESRGAN_AVAILABLE = False

REAL_ESRGAN = None
THREAD_LOCK = threading.Lock()
NAME = "DLC.REAL-ESRGAN"

def pre_check() -> bool:
    if not REALESRGAN_AVAILABLE:
        update_status("Real-ESRGAN not available. Install with: pip install realesrgan", NAME)
        return False
    return True

def pre_start() -> bool:
    if not REALESRGAN_AVAILABLE:
        return False
    
    if get_real_esrgan() is None:
        return False
    return True

def get_real_esrgan() -> Any:
    global REAL_ESRGAN
    
    with THREAD_LOCK:
        if REAL_ESRGAN is None and REALESRGAN_AVAILABLE:
            try:
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
                REAL_ESRGAN = RealESRGANer(
                    scale=4,
                    model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
                    model=model,
                    tile=0,
                    tile_pad=10,
                    pre_pad=0,
                    half=True if modules.globals.execution_providers[0] == 'CUDAExecutionProvider' else False
                )
                update_status("Real-ESRGAN model loaded successfully.", NAME)
            except Exception as e:
                update_status(f"Error loading Real-ESRGAN model: {e}", NAME)
                REAL_ESRGAN = None
    return REAL_ESRGAN

def process_frame(temp_frame: Frame) -> Frame:
    if not REALESRGAN_AVAILABLE:
        return temp_frame
    
    real_esrgan = get_real_esrgan()
    if real_esrgan is None:
        return temp_frame
    
    try:
        enhanced_frame, _ = real_esrgan.enhance(temp_frame, outscale=4)
        return enhanced_frame
    except Exception as e:
        update_status(f"Error enhancing frame with Real-ESRGAN: {e}", NAME)
        return temp_frame

def process_frames(source_path: str, temp_frame_paths: list, progress: Any = None) -> None:
    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        if temp_frame is not None:
            result_frame = process_frame(temp_frame)
            cv2.imwrite(temp_frame_path, result_frame)
        if progress:
            progress.update(1)

def process_image(source_path: str, target_path: str, output_path: str) -> None:
    target_frame = cv2.imread(target_path)
    if target_frame is not None:
        result_frame = process_frame(target_frame)
        cv2.imwrite(output_path, result_frame)
        update_status(f"Enhanced image saved to: {output_path}", NAME)

def process_video(source_path: str, temp_frame_paths: list) -> None:
    import modules.processors.frame.core
    modules.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)