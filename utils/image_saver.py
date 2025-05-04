"""
Utility functions for safely saving images with proper file extension handling.
This module helps prevent the "could not find a writer for the specified extension" error.
"""

import os
import cv2
import numpy as np
from PIL import Image

def save_image(image, file_path, quality=95):
    """
    Save an image with proper file extension handling and fallback options.
    
    Args:
        image: numpy array image (BGR format for OpenCV)
        file_path: destination path for the image
        quality: JPEG quality (0-100) if saving as JPEG
        
    Returns:
        bool: True if successful, False otherwise
        str: Path of the saved file or error message
    """
    try:
        # Ensure image is a valid numpy array
        if not isinstance(image, np.ndarray):
            return False, "Input is not a valid image array"
            
        # Check if image data is valid
        if image.size == 0 or image.min() < 0 or image.max() > 255:
            # Attempt to fix the image data if it's out of range
            image = np.clip(image, 0, 255).astype(np.uint8)
            
        # Ensure the file has a valid extension
        file_path = ensure_valid_extension(file_path)
        
        # Try OpenCV first as it's generally faster
        success = cv2.imwrite(file_path, image)
        
        if success:
            return True, file_path
        
        # If OpenCV fails, try PIL as a fallback
        return save_with_pil(image, file_path, quality)
    
    except Exception as e:
        return False, f"Error saving image: {str(e)}"

def save_with_pil(image, file_path, quality=95):
    """Use PIL to save the image as a fallback method"""
    try:
        # Convert from BGR (OpenCV) to RGB (PIL) if image has 3 channels
        if len(image.shape) == 3 and image.shape[2] == 3:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(image)
            
        # Save with appropriate quality setting
        _, ext = os.path.splitext(file_path)
        if ext.lower() in ['.jpg', '.jpeg']:
            pil_image.save(file_path, quality=quality)
        else:
            pil_image.save(file_path)
            
        return True, file_path
    except Exception as e:
        return False, f"PIL fallback failed: {str(e)}"

def ensure_valid_extension(file_path):
    """Ensure the file has a valid image extension that OpenCV or PIL can handle"""
    _, extension = os.path.splitext(file_path)
    
    # List of extensions supported by OpenCV or PIL
    valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp']
    
    # If no extension or invalid, default to PNG
    if not extension or extension.lower() not in valid_extensions:
        file_path = os.path.splitext(file_path)[0] + '.png'
        
    return file_path

def check_image_libraries():
    """Check if image processing libraries are properly installed and working"""
    status = {
        'opencv': {'installed': False, 'version': None, 'writers': []},
        'pil': {'installed': False, 'version': None}
    }
    
    # Check OpenCV
    try:
        status['opencv']['installed'] = True
        status['opencv']['version'] = cv2.__version__
        
        # Check available writers (formats OpenCV can save)
        test_image = np.zeros((10, 10, 3), dtype=np.uint8)
        for ext in ['.png', '.jpg', '.bmp', '.tiff']:
            try:
                temp_path = f"test_img{ext}"
                result = cv2.imwrite(temp_path, test_image)
                if result and os.path.exists(temp_path):
                    status['opencv']['writers'].append(ext)
                    os.remove(temp_path)
            except:
                pass
    except:
        pass
    
    # Check PIL
    try:
        status['pil']['installed'] = True
        status['pil']['version'] = Image.__version__
    except:
        pass
    
    return status 