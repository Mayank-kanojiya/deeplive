from typing import Any, List
import cv2
import insightface
import threading
import numpy as np
import time
import queue
import modules.globals
import modules.processors.frame.core
from modules.core import update_status
from modules.face_analyser import get_one_face, get_many_faces, default_source_face
from modules.typing import Face, Frame
from modules.utilities import conditional_download, is_image, is_video
from modules.cluster_analysis import find_closest_centroid
import os

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = "DLC.FACE-SWAPPER-ENHANCED"

# Performance optimizations
class FrameBuffer:
    def __init__(self, size=3):
        self.buffer = []
        self.size = size
        
    def add_frame(self, frame):
        self.buffer.append(frame)
        if len(self.buffer) > self.size:
            self.buffer.pop(0)
    
    def get_interpolated_frame(self, current_frame):
        if len(self.buffer) < 2:
            return current_frame
        weights = [0.1, 0.3, 0.6]
        result = np.zeros_like(current_frame, dtype=np.float32)
        for i, frame in enumerate(self.buffer[-3:]):
            if i < len(weights):
                result += frame.astype(np.float32) * weights[i]
        return np.clip(result, 0, 255).astype(np.uint8)

class PerformanceMonitor:
    def __init__(self):
        self.frame_times = []
        self.target_fps = 30
        self.frame_start = 0
        
    def start_frame(self):
        self.frame_start = time.time()
        
    def end_frame(self):
        frame_time = time.time() - self.frame_start
        self.frame_times.append(frame_time)
        
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
            
        avg_time = sum(self.frame_times) / len(self.frame_times)
        current_fps = 1.0 / avg_time if avg_time > 0 else 0
        
        if current_fps < self.target_fps * 0.8:
            self.reduce_quality()
            
    def reduce_quality(self):
        if hasattr(modules.globals, 'face_enhancer_blend'):
            modules.globals.face_enhancer_blend = max(0.5, modules.globals.face_enhancer_blend * 0.9)

# Global instances
frame_buffer = FrameBuffer()
performance_monitor = PerformanceMonitor()

abs_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(abs_dir))), "models")

def optimize_gpu_performance():
    """Optimize GPU memory and performance"""
    if 'CUDAExecutionProvider' in modules.globals.execution_providers:
        try:
            import torch
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(0.8)
        except ImportError:
            pass

def get_face_swapper() -> Any:
    global FACE_SWAPPER
    
    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            optimize_gpu_performance()
            
            model_map = {
                'inswapper': 'inswapper_128_fp16.onnx',
                'simswap': 'simswap_256_fp16.onnx', 
                'ghost': 'ghost_256_fp16.onnx',
                'hyperswap': 'hyperswap_128_fp16.onnx'
            }
            
            selected_model = getattr(modules.globals, 'face_swap_model', 'auto')
            
            if selected_model == 'auto':
                model_files = [
                    "simswap_256_fp16.onnx",
                    "ghost_256_fp16.onnx", 
                    "hyperswap_128_fp16.onnx",
                    "inswapper_128_fp16.onnx"
                ]
            else:
                model_files = [model_map.get(selected_model, 'inswapper_128_fp16.onnx')]
            
            for model_file in model_files:
                model_path = os.path.join(models_dir, model_file)
                if os.path.exists(model_path):
                    try:
                        providers = modules.globals.execution_providers
                        FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=providers)
                        update_status(f"Enhanced face swapper model {model_file} loaded successfully.", NAME)
                        break
                    except Exception as e:
                        update_status(f"Error loading {model_file}: {e}", NAME)
                        continue
            
            if FACE_SWAPPER is None:
                update_status("No compatible face swapper model found.", NAME)
                return None
    return FACE_SWAPPER

def get_robust_faces(frame: Frame, angle_compensation=True):
    """Multi-angle face detection for perfect tracking"""
    if not angle_compensation:
        return get_many_faces(frame)
    
    faces = []
    angles = [0, 15, -15, 30, -30]
    
    for angle in angles:
        if angle == 0:
            rotated_frame = frame
        else:
            center = (frame.shape[1]//2, frame.shape[0]//2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_frame = cv2.warpAffine(frame, rotation_matrix, (frame.shape[1], frame.shape[0]))
        
        detected_faces = get_many_faces(rotated_frame)
        if detected_faces:
            for face in detected_faces:
                if angle != 0 and hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
                    # Transform landmarks back
                    inv_rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
                    landmarks_homogeneous = np.hstack([face.landmark_2d_106, np.ones((face.landmark_2d_106.shape[0], 1))])
                    face.landmark_2d_106 = (inv_rotation_matrix @ landmarks_homogeneous.T).T
                faces.append(face)
    
    # Remove duplicates based on bbox overlap
    unique_faces = []
    for face in faces:
        is_duplicate = False
        for existing_face in unique_faces:
            if face_overlap(face.bbox, existing_face.bbox) > 0.5:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_faces.append(face)
    
    return unique_faces

def face_overlap(bbox1, bbox2):
    """Calculate overlap ratio between two bounding boxes"""
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    
    # Calculate intersection
    xi1, yi1 = max(x1, x3), max(y1, y3)
    xi2, yi2 = min(x2, x4), min(y2, y4)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0
    
    intersection = (xi2 - xi1) * (yi2 - yi1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def create_adaptive_mouth_mask(face: Face, frame: Frame, motion_intensity: float = 1.0):
    """Enhanced mouth mask that adapts to mouth movements"""
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    if face is None or not hasattr(face, 'landmark_2d_106'):
        return mask, None, (0,0,0,0), None
    
    landmarks = face.landmark_2d_106
    if landmarks is None or landmarks.shape[0] < 106:
        return mask, None, (0,0,0,0), None
    
    try:
        # Extended mouth region for eating/sucking motions
        mouth_indices = [
            65, 66, 62, 70, 69, 18, 19, 20, 21, 22, 23, 24, 0, 8, 7, 6, 5, 4, 3, 2,
            61, 63, 64, 67, 68, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86
        ]
        
        # Filter valid indices
        valid_indices = [i for i in mouth_indices if i < landmarks.shape[0]]
        mouth_landmarks = landmarks[valid_indices].astype(np.float32)
        center = np.mean(mouth_landmarks, axis=0)
        
        # Adaptive expansion based on mouth opening
        mouth_height = np.max(mouth_landmarks[:, 1]) - np.min(mouth_landmarks[:, 1])
        mouth_width = np.max(mouth_landmarks[:, 0]) - np.min(mouth_landmarks[:, 0])
        
        # Dynamic expansion for eating/sucking
        expansion_factor = 1.0 + (motion_intensity * 0.5) + (mouth_height / max(mouth_width, 1) * 0.3)
        expanded_landmarks = (mouth_landmarks - center) * expansion_factor + center
        
        # Create multi-layer mask for smooth blending
        for layer, scale in enumerate([1.2, 1.0, 0.8]):
            layer_landmarks = (expanded_landmarks - center) * scale + center
            layer_landmarks = layer_landmarks.astype(np.int32)
            
            # Ensure landmarks are within frame bounds
            layer_landmarks[:, 0] = np.clip(layer_landmarks[:, 0], 0, frame.shape[1]-1)
            layer_landmarks[:, 1] = np.clip(layer_landmarks[:, 1], 0, frame.shape[0]-1)
            
            # Create convex hull for better coverage
            hull = cv2.convexHull(layer_landmarks)
            cv2.fillConvexPoly(mask, hull, 255 - (layer * 50))
        
        # Multi-stage blur for natural feathering
        mask = cv2.GaussianBlur(mask, (31, 31), 10)
        mask = cv2.GaussianBlur(mask, (15, 15), 5)
        
        # Calculate bounding box
        min_x, min_y = np.min(expanded_landmarks, axis=0).astype(int)
        max_x, max_y = np.max(expanded_landmarks, axis=0).astype(int)
        
        # Add padding and clamp to frame bounds
        padding = int(max(mouth_width, mouth_height) * 0.1)
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(frame.shape[1], max_x + padding)
        max_y = min(frame.shape[0], max_y + padding)
        
        mouth_cutout = frame[min_y:max_y, min_x:max_x].copy() if max_x > min_x and max_y > min_y else None
        mouth_box = (min_x, min_y, max_x, max_y)
        
        return mask, mouth_cutout, mouth_box, expanded_landmarks.astype(np.int32)
        
    except Exception as e:
        print(f"Error in create_adaptive_mouth_mask: {e}")
        return mask, None, (0,0,0,0), None

def swap_face_enhanced(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    """Enhanced face swap with performance optimizations"""
    performance_monitor.start_frame()
    
    face_swapper = get_face_swapper()
    if face_swapper is None:
        return temp_frame

    if source_face is None or target_face is None or temp_frame is None:
        return temp_frame
    
    if not isinstance(temp_frame, np.ndarray) or temp_frame.size == 0:
        return temp_frame
        
    if not hasattr(source_face, 'embedding') or not hasattr(target_face, 'kps'):
        return temp_frame
        
    if source_face.embedding is None or target_face.kps is None:
        return temp_frame

    original_frame = temp_frame.copy()

    if temp_frame.dtype != np.uint8:
        temp_frame = np.clip(temp_frame, 0, 255).astype(np.uint8)
    
    if not temp_frame.flags['C_CONTIGUOUS']:
        temp_frame = np.ascontiguousarray(temp_frame)

    try:
        swapped_frame_raw = face_swapper.get(temp_frame, target_face, source_face, paste_back=True)
        
        if swapped_frame_raw is None:
             return original_frame

        if not isinstance(swapped_frame_raw, np.ndarray):
            return original_frame

        if swapped_frame_raw.shape != temp_frame.shape:
             swapped_frame_raw = cv2.resize(swapped_frame_raw, (temp_frame.shape[1], temp_frame.shape[0]))

        if not np.isfinite(swapped_frame_raw).all():
            return original_frame
            
        swapped_frame = np.clip(swapped_frame_raw, 0, 255).astype(np.uint8)
        
        if not swapped_frame.flags['C_CONTIGUOUS']:
            swapped_frame = np.ascontiguousarray(swapped_frame)

    except Exception as e:
        print(f"Error during face swap: {e}")
        return original_frame

    # Apply enhanced mouth mask if enabled
    if getattr(modules.globals, "mouth_mask", False):
        mask, mouth_cutout, mouth_box, mouth_polygon = create_adaptive_mouth_mask(target_face, temp_frame)
        
        if mouth_cutout is not None and mouth_box != (0,0,0,0):
            swapped_frame = apply_mouth_area_enhanced(swapped_frame, mouth_cutout, mouth_box, mask, mouth_polygon)

    # Apply opacity blend
    opacity = getattr(modules.globals, "opacity", 1.0)
    opacity = max(0.0, min(1.0, opacity))

    final_swapped_frame = cv2.addWeighted(original_frame.astype(np.uint8), 1 - opacity, swapped_frame.astype(np.uint8), opacity, 0)
    final_swapped_frame = final_swapped_frame.astype(np.uint8)
    
    # Add to frame buffer for temporal smoothing
    frame_buffer.add_frame(final_swapped_frame)
    result = frame_buffer.get_interpolated_frame(final_swapped_frame)
    
    performance_monitor.end_frame()
    return result

def apply_mouth_area_enhanced(frame, mouth_cutout, mouth_box, face_mask, mouth_polygon):
    """Enhanced mouth area application with better blending"""
    if frame is None or mouth_cutout is None or mouth_box is None:
        return frame
    
    try:
        min_x, min_y, max_x, max_y = map(int, mouth_box)
        box_width = max_x - min_x
        box_height = max_y - min_y
        
        if box_width <= 0 or box_height <= 0:
            return frame
        
        frame_h, frame_w = frame.shape[:2]
        min_y, max_y = max(0, min_y), min(frame_h, max_y)
        min_x, max_x = max(0, min_x), min(frame_w, max_x)
        
        box_width = max_x - min_x
        box_height = max_y - min_y
        if box_width <= 0 or box_height <= 0:
            return frame
        
        roi = frame[min_y:max_y, min_x:max_x]
        if roi.size == 0:
            return frame
        
        # Resize mouth cutout to fit ROI
        if roi.shape[:2] != mouth_cutout.shape[:2]:
            if mouth_cutout.shape[0] > 0 and mouth_cutout.shape[1] > 0:
                resized_mouth_cutout = cv2.resize(mouth_cutout, (box_width, box_height), interpolation=cv2.INTER_LINEAR)
            else:
                return frame
        else:
            resized_mouth_cutout = mouth_cutout
        
        if resized_mouth_cutout is None or resized_mouth_cutout.size == 0:
            return frame
        
        # Create polygon mask
        polygon_mask_roi = np.zeros(roi.shape[:2], dtype=np.uint8)
        if mouth_polygon is not None:
            adjusted_polygon = mouth_polygon - [min_x, min_y]
            adjusted_polygon = np.clip(adjusted_polygon, 0, [box_width-1, box_height-1])
            cv2.fillPoly(polygon_mask_roi, [adjusted_polygon.astype(np.int32)], 255)
        
        # Feather the mask
        feathered_mask = cv2.GaussianBlur(polygon_mask_roi.astype(float), (21, 21), 7)
        max_val = feathered_mask.max()
        if max_val > 1e-6:
            feathered_mask = feathered_mask / max_val
        else:
            feathered_mask.fill(0.0)
        
        # Blend
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            feathered_mask_3channel = feathered_mask[:, :, np.newaxis]
            blended_roi = (resized_mouth_cutout.astype(np.float64) * feathered_mask_3channel +
                          roi.astype(np.float64) * (1.0 - feathered_mask_3channel))
            frame[min_y:max_y, min_x:max_x] = blended_roi.astype(np.uint8)
        
    except Exception as e:
        print(f"Error applying enhanced mouth area: {e}")
    
    return frame

def process_frame_enhanced(source_face: Face, temp_frame: Frame) -> Frame:
    """Enhanced frame processing with robust face detection"""
    if getattr(modules.globals, "opacity", 1.0) == 0:
        return temp_frame

    processed_frame = temp_frame
    
    if modules.globals.many_faces:
        many_faces = get_robust_faces(processed_frame, angle_compensation=True)
        if many_faces:
            current_swap_target = processed_frame.copy()
            for target_face in many_faces:
                current_swap_target = swap_face_enhanced(source_face, target_face, current_swap_target)
            processed_frame = current_swap_target
    else:
        target_face = get_one_face(processed_frame)
        if target_face:
            processed_frame = swap_face_enhanced(source_face, target_face, processed_frame)

    return processed_frame

# Export the enhanced functions
def process_frame(source_face: Face, temp_frame: Frame) -> Frame:
    return process_frame_enhanced(source_face, temp_frame)

def process_frames(source_path: str, temp_frame_paths: List[str], progress: Any = None) -> None:
    """Enhanced frame processing with performance monitoring"""
    source_face = None
    
    if not source_path or not os.path.exists(source_path):
        update_status(f"Error: Source path invalid: {source_path}", NAME)
        return
    
    try:
        source_img = cv2.imread(source_path, cv2.IMREAD_COLOR)
        if source_img is None:
            update_status(f"Error reading source image: {source_path}", NAME)
            return
        
        if len(source_img.shape) == 3 and source_img.shape[2] == 3:
            source_face = get_one_face(source_img)
            if source_face is None:
                update_status(f"No face detected in source image: {source_path}", NAME)
                return
        else:
            update_status(f"Source image not in proper format: {source_path}", NAME)
            return
            
    except Exception as e:
        update_status(f"Error processing source image {source_path}: {e}", NAME)
        return

    total_frames = len(temp_frame_paths)
    
    for i, temp_frame_path in enumerate(temp_frame_paths):
        try:
            if not os.path.exists(temp_frame_path) or not os.path.isfile(temp_frame_path):
                print(f"{NAME}: Frame file does not exist: {temp_frame_path}")
                if progress: progress.update(1)
                continue
                
            temp_frame = cv2.imread(temp_frame_path)
            if temp_frame is None:
                print(f"{NAME}: Could not read frame: {temp_frame_path}")
                if progress: progress.update(1)
                continue
                
            if temp_frame.size == 0 or not isinstance(temp_frame, np.ndarray):
                print(f"{NAME}: Invalid frame data: {temp_frame_path}")
                if progress: progress.update(1)
                continue
                
        except Exception as read_e:
            print(f"{NAME}: Error reading frame {temp_frame_path}: {read_e}")
            if progress: progress.update(1)
            continue

        result_frame = None
        try:
            result_frame = process_frame_enhanced(source_face, temp_frame)
            
            if result_frame is None:
                print(f"{NAME}: Processing returned None for frame {temp_frame_path}")
                result_frame = temp_frame

        except Exception as proc_e:
            print(f"{NAME}: Error processing frame {temp_frame_path}: {proc_e}")
            result_frame = temp_frame

        try:
            write_success = cv2.imwrite(temp_frame_path, result_frame)
            if not write_success:
                print(f"{NAME}: Failed to write processed frame to {temp_frame_path}")
        except Exception as write_e:
            print(f"{NAME}: Error writing frame {temp_frame_path}: {write_e}")

        if progress:
            progress.update(1)

def process_image(source_path: str, target_path: str, output_path: str) -> None:
    """Enhanced image processing"""
    try:
        target_frame = cv2.imread(target_path)
        if target_frame is None:
            update_status(f"Error: Could not read target image: {target_path}", NAME)
            return
    except Exception as read_e:
        update_status(f"Error reading target image {target_path}: {read_e}", NAME)
        return

    try:
        source_img = cv2.imread(source_path)
        if source_img is None:
            update_status(f"Error: Could not read source image: {source_path}", NAME)
            return
        source_face = get_one_face(source_img)
        if not source_face:
            update_status(f"Error: No face found in source image: {source_path}", NAME)
            return
    except Exception as src_e:
        update_status(f"Error reading source image {source_path}: {src_e}", NAME)
        return

    result = process_frame_enhanced(source_face, target_frame)

    if result is not None:
        write_success = cv2.imwrite(output_path, result)
        if write_success:
            update_status(f"Enhanced output image saved to: {output_path}", NAME)
        else:
            update_status(f"Error: Failed to write output image to {output_path}", NAME)
    else:
        update_status("Enhanced image processing failed.", NAME)

def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    """Enhanced video processing"""
    update_status("Processing video with enhanced face swapper.", NAME)
    modules.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)

def pre_check() -> bool:
    return True

def pre_start() -> bool:
    model_path = os.path.join(models_dir, "inswapper_128_fp16.onnx")
    if not os.path.exists(model_path):
        update_status(f"Model not found: {model_path}", NAME)
        return False
    
    if get_face_swapper() is None:
        return False
    
    return True