# InvisiMark

A comprehensive Digital Image Processing project that provides steganography, invisible watermarking, and AI-powered tamper detection in a single package.

## Features

- **Image Steganography**: Hide secret messages or files inside images using LSB (Least Significant Bit) method
- **Invisible Watermarking**: Embed invisible watermarks using DCT (Discrete Cosine Transform) technique
- **AI-Powered Tamper Detection**: Detect if an image has been modified using machine learning
- **User-Friendly GUI**: Intuitive interface for all operations with real-time feedback
- **Reliable File Handling**: Enhanced file saving with automatic format detection and fallback options

## Installation

### Prerequisites

- Python 3.8+ 
- Required packages (will be installed automatically)

### Step-by-Step Installation

1. Clone this repository:
   ```
   git clone https://github.com/brighteyekid/InvisiMark
   cd InvisiMark
   ```

2. Set up a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # OR
   venv\Scripts\activate  # Windows
   ```

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

### Troubleshooting Installation

If you encounter Qt platform plugin errors when running the GUI:

- **Linux users**: Install the required Qt dependencies:
  ```
  sudo apt-get install libxcb-xinerama0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 libxcb-randr0 libxcb-xkb1 libxkbcommon-x11-0
  ```

- **Virtual environment users**: Install PyQt5 with additional options:
  ```
  pip uninstall PyQt5
  pip install PyQt5 --config-settings --confirm-license= --verbose
  ```

## Usage

### Running the Application

Start the GUI application with:
```
python main.py
```

Or for Linux/macOS users, use the convenient script:
```
./run.sh
```

### Command-Line Demo

For a quick demonstration of all features without the GUI:
```
python demo.py --image path/to/image.jpg --output demo_results
```

### Steganography Features

1. **Encode a Message**:
   - Load a cover image
   - Enter your secret message
   - Enable/disable encryption
   - Save the resulting stego image

2. **Decode a Message**:
   - Load a stego image
   - Provide the decryption key if encryption was used
   - Extract the hidden message

### Watermarking Features

1. **Embed a Watermark**:
   - Load an original image
   - Enter watermark text
   - Adjust watermark strength using the slider
   - Save watermarked image

2. **Verify a Watermark**:
   - Load a watermarked image
   - Enter the original watermark text
   - Check verification results

### Tamper Detection

1. **Detect Tampering**:
   - Load an image
   - Select model type (CNN or autoencoder)
   - Train a demo model or load a pre-trained one
   - Analyze the image for evidence of tampering

## Recent Updates

### Updates (v1.4.0)
- **Simplified Watermarking**: Now using only DCT frequency domain method for watermarking
- **Improved File Saving**: Added reliable image saving with automatic format detection 
- **Robust Error Handling**: Better handling of file operations with fallback options

### Updates (v1.3.0)
- **Simplified Steganography**: Now using only LSB (Least Significant Bit) method for improved reliability
- **Removed Complex Methods**: Simplified codebase for better maintainability
- **Performance Improvements**: Faster operation with more efficient code

### Bug Fixes (v1.2.1)
- **Fixed Integer Out of Bounds Error**: Resolved "Python integer -2 out of bounds for uint8" error
- **Improved Error Handling**: All modules now gracefully handle exceptions with informative error messages
- **Fixed Image Saving Issues**: Better handling of file extensions and fallback options

## Technical Details

### Steganography Method

- **LSB (Least Significant Bit)**: Hides data in the least significant bits of pixel values with minimal visual impact. This technique modifies the last bit of each pixel's color value to encode the hidden message, resulting in changes that are imperceptible to the human eye.

### Watermarking Technique

- **DCT (Discrete Cosine Transform)**: Embeds watermarks in the frequency domain of the image. This method transforms the image from the spatial domain to the frequency domain, embeds the watermark in the mid-frequency components, and then transforms it back. This approach provides a good balance between imperceptibility and robustness.

### AI Tamper Detection

- **CNN-based Detection**: Uses a convolutional neural network to classify authentic vs. tampered images
- **Autoencoder-based Detection**: Learns the patterns of authentic images and detects anomalies

## Project Structure

- `/app`: GUI application code
- `/utils`: Steganography and watermarking implementations, including image saving utilities
- `/models`: AI-based tamper detection models
- `/data`: Data directory for training and testing

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
