#!/usr/bin/env python3
import sys
import os
import argparse

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.gui import main

def show_info():
    """Display information about the simplified LSB steganography approach"""
    print("=" * 80)
    print("InvisiMark: Simplified Steganography Edition")
    print("=" * 80)
    print("This version of InvisiMark uses only the LSB (Least Significant Bit) method")
    print("for steganography, providing better reliability and performance.")
    print("")
    print("Benefits of LSB-only approach:")
    print("  • More reliable across different image types")
    print("  • Faster encoding and decoding")
    print("  • Better compatibility with various image formats")
    print("  • Simpler implementation with fewer errors")
    print("")
    print("For more information, see simplified_steganography.md")
    print("=" * 80)
    print("")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InvisiMark - Steganography & Watermarking Tool")
    parser.add_argument('--silent', action='store_true', help="Don't show initial information")
    args = parser.parse_args()
    
    if not args.silent:
        show_info()
    
    # Run the GUI application with or without the popup info
    main(not args.silent)