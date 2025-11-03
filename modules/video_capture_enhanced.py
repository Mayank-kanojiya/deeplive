import cv2
import numpy as np
from typing import Optional, Tuple, Callable
import platform
import threading
import queue
import time

# Only import Windows-specific library if on Windows
if platform.system() == "Windows":
    try:
        from pygrabber.dshow_graph import FilterGraph
    except ImportError:
        FilterGraph = None

class OptimizedVideoCapturer:
    def __init__(self, device_index: int):
        self.device_index = device_index
        self.frame_callback = None
        self._current_frame = None
        self._frame_ready = threading.Event()
        self.is_running = False
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=3)
        self.capture_thread = None
        self.last_frame_time = 0
        self.target_fps = 30
        self.frame_interval = 1.0 / self.target_fps

        # Initialize Windows-specific components if on Windows
        if platform.system() == "Windows" and FilterGraph is not None:
            try:
                self.graph = FilterGraph()
                devices = self.graph.get_input_devices()
                if self.device_index >= len(devices):
                    raise ValueError(f"Invalid device index {device_index}. Available devices: {len(devices)}")
            except Exception as e:
                print(f"Warning: Could not initialize DirectShow graph: {e}")
                self.graph = None
        else:
            self.graph = None

    def start(self, width: int = 960, height: int = 540, fps: int = 60) -> bool:
        """Initialize and start optimized video capture"""
        try:
            self.target_fps = fps
            self.frame_interval = 1.0 / fps
            
            if platform.system() == "Windows":
                # Windows-specific capture methods with optimization
                capture_methods = [
                    (self.device_index, cv2.CAP_DSHOW),
                    (self.device_index, cv2.CAP_MSMF),  # Media Foundation
                    (self.device_index, cv2.CAP_ANY),
                    (-1, cv2.CAP_ANY),
                    (0, cv2.CAP_ANY),
                ]

                for dev_id, backend in capture_methods:
                    try:
                        self.cap = cv2.VideoCapture(dev_id, backend)
                        if self.cap.isOpened():
                            break
                        self.cap.release()
                    except Exception:
                        continue
            else:
                # Unix-like systems
                self.cap = cv2.VideoCapture(self.device_index)

            if not self.cap or not self.cap.isOpened():
                raise RuntimeError("Failed to open camera")

            # Optimize capture settings for performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            
            # Additional optimizations
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize latency
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # Use MJPEG for better performance
            
            # Auto exposure and focus settings for better quality
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual exposure
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus

            self.is_running = True
            self.start_threaded_capture()
            return True

        except Exception as e:
            print(f"Failed to start optimized capture: {str(e)}")
            if self.cap:
                self.cap.release()
            return False

    def start_threaded_capture(self):
        """Separate thread for frame capture to prevent blocking"""
        def capture_loop():
            while self.is_running:
                current_time = time.time()
                
                # Frame rate limiting
                if current_time - self.last_frame_time < self.frame_interval:
                    time.sleep(0.001)  # Small sleep to prevent busy waiting
                    continue
                
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.last_frame_time = current_time
                    
                    # Drop oldest frame if queue is full (prevents memory buildup)
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    
                    try:
                        self.frame_queue.put(frame, block=False)
                        self._current_frame = frame
                        self._frame_ready.set()
                        
                        if self.frame_callback:
                            self.frame_callback(frame)
                    except queue.Full:
                        pass  # Skip frame if queue is full
                else:
                    time.sleep(0.01)  # Wait a bit if no frame available
                    
        self.capture_thread = threading.Thread(target=capture_loop, daemon=True)
        self.capture_thread.start()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the camera with zero-copy optimization"""
        if not self.is_running or self.cap is None:
            return False, None

        try:
            frame = self.frame_queue.get_nowait()
            return True, frame
        except queue.Empty:
            # Return last known frame if queue is empty
            if self._current_frame is not None:
                return True, self._current_frame
            return False, None

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get most recent frame without blocking"""
        try:
            # Get all available frames and return the latest
            latest_frame = None
            while True:
                try:
                    latest_frame = self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            return latest_frame
        except:
            return self._current_frame

    def wait_for_frame(self, timeout: float = 1.0) -> bool:
        """Wait for a new frame to be available"""
        return self._frame_ready.wait(timeout)

    def get_fps(self) -> float:
        """Get actual capture FPS"""
        if self.cap and self.cap.isOpened():
            return self.cap.get(cv2.CAP_PROP_FPS)
        return 0.0

    def get_resolution(self) -> Tuple[int, int]:
        """Get current capture resolution"""
        if self.cap and self.cap.isOpened():
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return width, height
        return 0, 0

    def set_exposure(self, exposure: float):
        """Set camera exposure (-13 to -1, lower = darker)"""
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)

    def set_brightness(self, brightness: float):
        """Set camera brightness (0 to 255)"""
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)

    def set_contrast(self, contrast: float):
        """Set camera contrast (0 to 255)"""
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_CONTRAST, contrast)

    def release(self) -> None:
        """Stop capture and release resources"""
        self.is_running = False
        
        # Wait for capture thread to finish
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        
        # Clear the queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def set_frame_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """Set callback for frame processing"""
        self.frame_callback = callback

    def is_opened(self) -> bool:
        """Check if capture is opened and running"""
        return self.is_running and self.cap is not None and self.cap.isOpened()

# Backward compatibility
class VideoCapturer(OptimizedVideoCapturer):
    """Backward compatible class name"""
    pass

# Utility functions for camera management
def list_cameras() -> list:
    """List available cameras"""
    cameras = []
    
    if platform.system() == "Windows" and FilterGraph is not None:
        try:
            graph = FilterGraph()
            devices = graph.get_input_devices()
            for i, device in enumerate(devices):
                cameras.append({"index": i, "name": device})
        except Exception as e:
            print(f"Error listing cameras with DirectShow: {e}")
            # Fallback to OpenCV detection
            for i in range(10):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    cameras.append({"index": i, "name": f"Camera {i}"})
                    cap.release()
    else:
        # Unix-like systems or fallback
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cameras.append({"index": i, "name": f"Camera {i}"})
                cap.release()
    
    return cameras

def test_camera(device_index: int) -> bool:
    """Test if a camera is working"""
    try:
        cap = cv2.VideoCapture(device_index)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            return ret and frame is not None
    except Exception:
        pass
    return False

def get_optimal_camera_settings(device_index: int) -> dict:
    """Get optimal settings for a camera"""
    settings = {
        "width": 960,
        "height": 540,
        "fps": 30,
        "exposure": -6,
        "brightness": 128,
        "contrast": 128
    }
    
    try:
        cap = cv2.VideoCapture(device_index)
        if cap.isOpened():
            # Test different resolutions
            resolutions = [(1920, 1080), (1280, 720), (960, 540), (640, 480)]
            for width, height in resolutions:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                if actual_width == width and actual_height == height:
                    settings["width"] = width
                    settings["height"] = height
                    break
            
            # Test FPS
            for fps in [60, 30, 25, 15]:
                cap.set(cv2.CAP_PROP_FPS, fps)
                actual_fps = cap.get(cv2.CAP_PROP_FPS)
                if abs(actual_fps - fps) < 5:  # Allow some tolerance
                    settings["fps"] = fps
                    break
            
            cap.release()
    except Exception as e:
        print(f"Error getting optimal settings: {e}")
    
    return settings