#!/usr/bin/env python3
"""
InvisiMark CLI Demo
This script demonstrates the core functionality of InvisiMark without the GUI.
"""

import os
import sys
import cv2
import numpy as np
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.steganography import Steganography
from utils.watermarking import Watermarking
from models.tamper_detector import TamperDetector

def demo_steganography(input_image_path, output_dir):
    """Demonstrate steganography capabilities"""
    print("\n=== Steganography Demo ===")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize steganography
    stego = Steganography()
    
    # Load cover image
    print(f"Loading cover image from: {input_image_path}")
    cover_img = cv2.imread(input_image_path)
    if cover_img is None:
        print(f"Error: Could not load image from {input_image_path}")
        return
    
    # Define secret message
    secret_message = "This is a secret message hidden by InvisiMark!"
    print(f"Secret message: {secret_message}")
    
    # LSB Encoding
    print("\n--- LSB Steganography ---")
    lsb_output_path = os.path.join(output_dir, "lsb_stego.png")
    lsb_stego, lsb_key = stego.encode(cover_img, secret_message, method='lsb')
    cv2.imwrite(lsb_output_path, lsb_stego)
    print(f"LSB steganography saved to: {lsb_output_path}")
    print(f"Encryption key: {lsb_key.decode()}")
    
    # Extract the message
    extracted_message = stego.decode(lsb_stego, method='lsb', key=lsb_key)
    print(f"Extracted message: {extracted_message}")
    
    # DCT Encoding
    print("\n--- DCT Steganography ---")
    dct_output_path = os.path.join(output_dir, "dct_stego.png")
    dct_stego, dct_key = stego.encode(cover_img, secret_message, method='dct')
    cv2.imwrite(dct_output_path, dct_stego)
    print(f"DCT steganography saved to: {dct_output_path}")
    print(f"Encryption key: {dct_key.decode()}")
    
    # QR Code Encoding
    print("\n--- QR Code Steganography ---")
    qr_output_path = os.path.join(output_dir, "qr_stego.png")
    qr_stego, qr_key = stego.encode(cover_img, secret_message, method='qrcode')
    cv2.imwrite(qr_output_path, qr_stego)
    print(f"QR Code steganography saved to: {qr_output_path}")
    print(f"Encryption key: {qr_key.decode()}")
    
    return lsb_stego, dct_stego

def demo_watermarking(input_image_path, output_dir):
    """Demonstrate watermarking capabilities"""
    print("\n=== Watermarking Demo ===")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize watermarking
    watermark = Watermarking()
    
    # Load original image
    print(f"Loading original image from: {input_image_path}")
    original_img = cv2.imread(input_image_path)
    if original_img is None:
        print(f"Error: Could not load image from {input_image_path}")
        return
    
    # Define watermark text
    watermark_text = "Copyright © InvisiMark 2023"
    print(f"Watermark text: {watermark_text}")
    
    # DWT Watermarking
    print("\n--- DWT Watermarking ---")
    dwt_output_path = os.path.join(output_dir, "dwt_watermarked.png")
    dwt_watermarked, _ = watermark.embed_watermark(original_img, watermark_text, method='dwt', alpha=0.5)
    cv2.imwrite(dwt_output_path, dwt_watermarked)
    print(f"DWT watermarking saved to: {dwt_output_path}")
    
    # Verify the watermark
    _, dwt_score = watermark.extract_watermark(dwt_watermarked, watermark_text, method='dwt')
    print(f"DWT watermark verification score: {dwt_score:.4f}")
    
    # DCT Watermarking
    print("\n--- DCT Watermarking ---")
    dct_output_path = os.path.join(output_dir, "dct_watermarked.png")
    dct_watermarked, _ = watermark.embed_watermark(original_img, watermark_text, method='dct', alpha=0.5)
    cv2.imwrite(dct_output_path, dct_watermarked)
    print(f"DCT watermarking saved to: {dct_output_path}")
    
    # Fragile Watermarking for Anti-Deepfake
    print("\n--- Fragile Watermarking (Anti-Deepfake) ---")
    fragile_output_path = os.path.join(output_dir, "fragile_watermarked.png")
    fragile_watermarked, _ = watermark.embed_watermark(original_img, f"{watermark_text}_FRAGILE", method='fragile')
    cv2.imwrite(fragile_output_path, fragile_watermarked)
    print(f"Fragile watermarking saved to: {fragile_output_path}")
    
    # Create a tampered version
    print("\n--- Creating Tampered Version ---")
    tampered_output_path = os.path.join(output_dir, "tampered_watermarked.png")
    tampered = fragile_watermarked.copy()
    
    # Apply some noise to simulate tampering
    noise = np.random.normal(0, 15, tampered.shape).astype(np.uint8)
    tampered = cv2.add(tampered, noise)
    cv2.imwrite(tampered_output_path, tampered)
    print(f"Tampered image saved to: {tampered_output_path}")
    
    # Verify the original vs tampered
    _, original_score = watermark.extract_watermark(fragile_watermarked, f"{watermark_text}_FRAGILE", method='fragile')
    print(f"Original fragile watermark verification score: {original_score:.4f}")
    
    _, tampered_score = watermark.extract_watermark(tampered, f"{watermark_text}_FRAGILE", method='fragile')
    print(f"Tampered fragile watermark verification score: {tampered_score:.4f}")
    
    return dwt_watermarked, fragile_watermarked, tampered

def demo_tamper_detection(original_image_path, tampered_image_path, output_dir):
    """Demonstrate tamper detection capabilities"""
    print("\n=== Tamper Detection Demo ===")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize detector
    detector = TamperDetector(input_shape=(128, 128, 3))
    
    # Load images
    print(f"Loading original image from: {original_image_path}")
    original_img = cv2.imread(original_image_path)
    if original_img is None:
        print(f"Error: Could not load image from {original_image_path}")
        return
    
    print(f"Loading tampered image from: {tampered_image_path}")
    tampered_img = cv2.imread(tampered_image_path)
    if tampered_img is None:
        print(f"Error: Could not load image from {tampered_image_path}")
        return
    
    # Build model
    print("\nBuilding CNN tamper detection model...")
    detector.build_model(model_type='cnn')
    
    # Generate training data
    print("Generating training data from original image...")
    X, y = detector.generate_training_data([original_img], tamper_types=['noise', 'blur', 'jpeg', 'crop'])
    print(f"Generated {len(X)} training samples")
    
    # Train model
    print("Training model (this is a minimal demo with 2 epochs)...")
    detector.train(X, y, model_type='cnn', epochs=2, batch_size=2)
    print("Model training completed!")
    
    # Test on original image
    tampered_original, confidence_original = detector.detect_tampering(original_img, model_type='cnn')
    print(f"\nOriginal image detection: {'TAMPERED' if tampered_original else 'AUTHENTIC'} (confidence: {confidence_original:.4f})")
    
    # Test on tampered image
    tampered_detected, confidence_tampered = detector.detect_tampering(tampered_img, model_type='cnn')
    print(f"Tampered image detection: {'TAMPERED' if tampered_detected else 'AUTHENTIC'} (confidence: {confidence_tampered:.4f})")
    
    print("\nNote: For a real-world application, you would train on a larger dataset and use more epochs.")

def main():
    parser = argparse.ArgumentParser(description="InvisiMark CLI Demo")
    parser.add_argument('--image', default='sample.png', help='Path to an input image')
    parser.add_argument('--output', default='demo_output', help='Output directory for demo results')
    args = parser.parse_args()
    
    # Check if input image exists
    if not os.path.exists(args.image):
        print(f"Error: Input image not found at {args.image}")
        print("Please provide a valid image path or create a sample image.")
        
        # Create a sample image
        print("\nCreating a sample image for demonstration...")
        sample_dir = os.path.dirname(args.image)
        os.makedirs(sample_dir, exist_ok=True)
        
        # Create a simple colored gradient image
        width, height = 512, 512
        sample_img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create a simple gradient
        for y in range(height):
            for x in range(width):
                sample_img[y, x, 0] = int(255 * (x / width))  # Blue channel
                sample_img[y, x, 1] = int(255 * (y / height))  # Green channel
                sample_img[y, x, 2] = 128  # Red channel
                
        cv2.imwrite(args.image, sample_img)
        print(f"Created sample image: {args.image}")
    
    # Run demos
    print("\n========== INVISIMARK DEMO ==========")
    print("This demo will demonstrate the core functionality of InvisiMark.")
    
    # Run steganography demo
    stego_img1, stego_img2 = demo_steganography(args.image, args.output)
    
    # Run watermarking demo
    watermarked_img, fragile_img, tampered_img = demo_watermarking(args.image, args.output)
    
    # Run tamper detection demo
    tampered_path = os.path.join(args.output, "tampered_watermarked.png")
    demo_tamper_detection(args.image, tampered_path, args.output)
    
    print("\n========== DEMO COMPLETE ==========")
    print(f"All output files can be found in the '{args.output}' directory.")
    print("Run the full application with GUI using: python main.py")

if __name__ == "__main__":
    main() 