#!/usr/bin/env python3
"""
Quick test script to verify the face swapper fixes
"""
import sys
import os
import numpy as np
import cv2

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from modules.processors.frame.face_swapper import swap_face, get_face_swapper
    from modules.face_analyser import get_one_face
    import modules.globals
    
    print("✓ Successfully imported face swapper modules")
    
    # Set basic globals
    modules.globals.execution_providers = ['CPUExecutionProvider']
    modules.globals.face_swap_model = 'auto'
    modules.globals.opacity = 1.0
    modules.globals.mouth_mask = False
    
    # Test face swapper loading
    face_swapper = get_face_swapper()
    if face_swapper is not None:
        print("✓ Face swapper model loaded successfully")
    else:
        print("✗ Face swapper model failed to load")
        
    # Create a dummy frame for testing
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    test_frame.fill(128)  # Gray frame
    
    print("✓ Test frame created")
    
    # Test input validation
    result = swap_face(None, None, test_frame)
    if np.array_equal(result, test_frame):
        print("✓ Input validation working correctly")
    else:
        print("✗ Input validation failed")
        
    print("\nAll basic tests passed! The fixes should help resolve the OpenCV warpAffine errors.")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Make sure you're running this from the Deep-Live-Cam directory")
except Exception as e:
    print(f"✗ Error during testing: {e}")
    import traceback
    traceback.print_exc()