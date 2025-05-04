#!/usr/bin/env python
"""
Test script to verify the fix for the "Python integer out of bounds for uint8" error
in the steganography module.
"""

import sys
import os
import cv2
import numpy as np
import traceback

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from utils.steganography import Steganography

def test_all_methods():
    """Test all steganography methods to ensure they work without errors."""
    print("Testing steganography methods...")
    
    # Create test images of different sizes
    test_sizes = [
        (64, 64),      # Very small (may use fallback for QR)
        (200, 200),    # Medium sized
        (512, 512, 3)  # Larger color image
    ]
    
    test_images = []
    for size in test_sizes:
        if len(size) == 2:
            img = np.random.randint(0, 256, size, dtype=np.uint8)
            print(f"Created grayscale test image: {size[0]}x{size[1]}")
        else:
            img = np.random.randint(0, 256, size, dtype=np.uint8)
            print(f"Created color test image: {size[0]}x{size[1]}x{size[2]}")
        test_images.append(img)
    
    # Create a test message
    test_message = "This is a test message to verify the fix."
    
    # Initialize the steganography module
    stego = Steganography()
    
    # Test each method
    methods = ["lsb", "dct", "qrcode"]
    
    for method in methods:
        print(f"\n{'='*50}")
        print(f"Testing {method.upper()} method:")
        print(f"{'='*50}")
        
        for i, test_img in enumerate(test_images):
            img_type = "grayscale" if len(test_img.shape) == 2 else "color"
            img_size = f"{test_img.shape[0]}x{test_img.shape[1]}"
            if img_type == "color":
                img_size += f"x{test_img.shape[2]}"
                
            print(f"\nImage {i+1}: {img_type} ({img_size})")
            print(f"{'-'*30}")
            
            try:
                # Encode
                print(f"  Encoding message...")
                stego_img, key = stego.encode(test_img.copy(), test_message, method=method, encrypt=True)
                print(f"  ✓ Successfully encoded message")
                
                # Check for negative values or out-of-bounds
                if np.any(stego_img < 0) or np.any(stego_img > 255):
                    print(f"  ✗ ERROR: Encoded image has values outside valid range: [{np.min(stego_img)}, {np.max(stego_img)}]")
                else:
                    print(f"  ✓ Encoded image has valid values range: [{np.min(stego_img)}, {np.max(stego_img)}]")
                
                try:
                    # Decode
                    decoded_msg = stego.decode(stego_img, method=method, key=key, encrypted=True)
                    if decoded_msg == test_message:
                        print(f"  ✓ Successfully decoded exact message match")
                    else:
                        print(f"  ⚠ Message decoded but with differences")
                        print(f"    Original: {test_message[:30]}...")
                        print(f"    Decoded: {decoded_msg[:30]}...")
                except Exception as e:
                    print(f"  ✗ Error decoding message: {str(e)}")
                    
            except Exception as e:
                print(f"  ✗ Error with {method} method: {str(e)}")
                if "fallback" in str(e).lower() or "fall back" in str(e).lower():
                    print(f"  ℹ This is expected for small images - fallback should be used")
    
    print("\nTest summary:")
    print("LSB: Should work on all image sizes")
    print("DCT: May have reduced capacity on small images")
    print("QR Code: Should use LSB fallback on small images")
    print("\nTest completed.")

if __name__ == "__main__":
    test_all_methods() 