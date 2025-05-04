#!/usr/bin/env python3
"""
Image Saving Fix Utility for InvisiMark

This script diagnoses and fixes issues with saving images in the InvisiMark application.
It checks for common problems like missing libraries, incorrect file extensions, and
provides workarounds.

Usage:
    python fix_image_saving.py
"""

import os
import sys
import subprocess
import importlib
import platform
import tempfile
import numpy as np

def print_header(text):
    """Print a section header"""
    print("\n" + "="*80)
    print(f" {text}")
    print("="*80)

def print_status(status, message):
    """Print a status message with color if available"""
    if status:
        print(f"[ \033[92mOK\033[0m ] {message}")
    else:
        print(f"[\033[91mFAIL\033[0m] {message}")

def check_dependencies():
    """Check if all required dependencies for image processing are installed"""
    print_header("Checking Image Processing Dependencies")
    
    dependencies = {
        'opencv-python': 'cv2',
        'pillow': 'PIL',
        'numpy': 'numpy'
    }
    
    all_installed = True
    
    for package, module in dependencies.items():
        try:
            imported = importlib.import_module(module)
            version = getattr(imported, '__version__', 'unknown')
            print_status(True, f"{package} is installed (version: {version})")
        except ImportError:
            print_status(False, f"{package} is NOT installed")
            all_installed = False
    
    if not all_installed:
        print("\nSome dependencies are missing. Install them with:")
        print("pip install opencv-python pillow numpy")
        
    return all_installed

def check_opencv_build():
    """Check OpenCV build information to ensure image codecs are available"""
    print_header("Checking OpenCV Build Information")
    
    try:
        import cv2
        
        # Get build information
        build_info = cv2.getBuildInformation()
        
        # Check for image codecs
        codecs = {
            'PNG': 'PNG:' in build_info,
            'JPEG': 'JPEG:' in build_info,
            'TIFF': 'TIFF:' in build_info,
            'WEBP': 'WEBP:' in build_info
        }
        
        for codec, available in codecs.items():
            print_status(available, f"{codec} support in OpenCV")
            
        return all(codecs.values())
    
    except Exception as e:
        print_status(False, f"Failed to check OpenCV build: {str(e)}")
        return False

def test_image_saving():
    """Test if image saving works with different methods and extensions"""
    print_header("Testing Image Saving")
    
    # Import required modules
    try:
        import cv2
        import numpy as np
        from PIL import Image
        
        # Create a test image (RGB gradient)
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        for i in range(100):
            for j in range(100):
                test_img[i, j] = [i * 2.55, j * 2.55, 128]
        
        results = {}
        
        # Test directory
        test_dir = tempfile.mkdtemp()
        print(f"Using temporary directory: {test_dir}")
        
        # Test OpenCV image writing with different extensions
        extensions = ['.png', '.jpg', '.bmp', '.tiff']
        for ext in extensions:
            filepath = os.path.join(test_dir, f"test_opencv{ext}")
            
            try:
                success = cv2.imwrite(filepath, test_img)
                filesize = os.path.getsize(filepath) if os.path.exists(filepath) else 0
                
                print_status(success and filesize > 0, 
                             f"OpenCV can save {ext} files " +
                             (f"({filesize} bytes)" if filesize > 0 else "(failed)"))
                results[f'opencv_{ext}'] = success and filesize > 0
                
                # Clean up
                if os.path.exists(filepath):
                    os.remove(filepath)
            
            except Exception as e:
                print_status(False, f"OpenCV failed to save {ext}: {str(e)}")
                results[f'opencv_{ext}'] = False
        
        # Test PIL image writing
        for ext in extensions:
            filepath = os.path.join(test_dir, f"test_pil{ext}")
            
            try:
                pil_img = Image.fromarray(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
                pil_img.save(filepath)
                filesize = os.path.getsize(filepath) if os.path.exists(filepath) else 0
                
                print_status(filesize > 0, 
                             f"PIL can save {ext} files " +
                             (f"({filesize} bytes)" if filesize > 0 else "(failed)"))
                results[f'pil_{ext}'] = filesize > 0
                
                # Clean up
                if os.path.exists(filepath):
                    os.remove(filepath)
            
            except Exception as e:
                print_status(False, f"PIL failed to save {ext}: {str(e)}")
                results[f'pil_{ext}'] = False
        
        # Test if our utility module works
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        try:
            from utils.image_saver import save_image
            
            filepath = os.path.join(test_dir, "test_utility.png")
            success, message = save_image(test_img, filepath)
            
            print_status(success, f"Utility save_image function: {message}")
            results['utility'] = success
            
            # Clean up
            if os.path.exists(filepath):
                os.remove(filepath)
                
        except Exception as e:
            print_status(False, f"Utility module test failed: {str(e)}")
            results['utility'] = False
        
        # Remove test directory
        try:
            os.rmdir(test_dir)
        except:
            pass
        
        return results
    
    except Exception as e:
        print_status(False, f"Image saving test failed: {str(e)}")
        return {}

def fix_opencv_extension_issue():
    """Apply fix for OpenCV extension issue in the application code"""
    print_header("Applying Fixes")
    
    # Check if our utility module exists
    util_path = os.path.join('utils', 'image_saver.py')
    if os.path.exists(util_path):
        print_status(True, f"Found utils/image_saver.py - No need to recreate it")
    else:
        print_status(False, f"Unable to find {util_path}")
        
    # Check if the gui.py has been patched
    gui_path = os.path.join('app', 'gui.py')
    if os.path.exists(gui_path):
        try:
            with open(gui_path, 'r') as file:
                content = file.read()
                
            if "ensure_valid_extension" in content or "save_with_pil" in content:
                print_status(True, "GUI code appears to be already patched")
            else:
                print_status(False, "GUI code may not be patched yet. You should run the full repair.")
        except Exception as e:
            print_status(False, f"Error checking GUI code: {str(e)}")
    else:
        print_status(False, f"Unable to find {gui_path}")
    
    # Print instructions for the user
    print("\nTo apply all fixes:")
    print("1. Make sure you're using the patched version of gui.py")
    print("2. Make sure utils/image_saver.py is present")
    print("3. Run the application with:")
    print("   python main.py")
    
    # Offer to automatically update requirements.txt
    try:
        req_path = 'requirements.txt'
        if os.path.exists(req_path):
            with open(req_path, 'r') as file:
                content = file.read()
                
            missing_reqs = []
            for req in ['pillow', 'opencv-python']:
                if req not in content.lower():
                    missing_reqs.append(req)
            
            if missing_reqs:
                print("\nSome dependencies may be missing from requirements.txt.")
                print(f"Missing: {', '.join(missing_reqs)}")
                choice = input("Add them to requirements.txt? (y/n): ").lower().strip()
                
                if choice == 'y':
                    with open(req_path, 'a') as file:
                        file.write("\n# Added by fix_image_saving.py\n")
                        for req in missing_reqs:
                            file.write(f"{req}\n")
                    print_status(True, "Updated requirements.txt")
    except Exception as e:
        print_status(False, f"Error updating requirements.txt: {str(e)}")

def main():
    """Main function to diagnose and fix image saving issues"""
    print_header("InvisiMark Image Saving Fix Utility")
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    
    # Run checks
    deps_ok = check_dependencies()
    if not deps_ok:
        print("\nPlease install missing dependencies first.")
        return
    
    opencv_ok = check_opencv_build()
    test_results = test_image_saving()
    
    # Analyze results
    print_header("Diagnosis")
    
    if all(test_results.values()):
        print_status(True, "All image saving tests passed!")
        print("\nIf you're still experiencing issues, check these potential problems:")
        print("1. File permissions - Make sure you have write access to the save location")
        print("2. Disk space - Ensure you have enough free space")
        print("3. Path validity - Watch for invalid characters in filenames")
    else:
        print_status(False, "Some image saving tests failed.")
        
        # Analyze which tests failed
        opencv_fails = any(not v for k, v in test_results.items() if k.startswith('opencv_'))
        pil_fails = any(not v for k, v in test_results.items() if k.startswith('pil_'))
        
        if opencv_fails and not pil_fails:
            print("\nOpenCV image saving is broken, but PIL works fine.")
            print("The utility module should provide a fallback to PIL.")
        elif pil_fails and not opencv_fails:
            print("\nPIL image saving is broken, but OpenCV works fine.")
            print("The default saving path should work.")
        elif pil_fails and opencv_fails:
            print("\nBoth OpenCV and PIL failed to save images.")
            print("This suggests a more serious system problem.")
            
        # Apply fixes
        fix_opencv_extension_issue()

if __name__ == "__main__":
    main() 