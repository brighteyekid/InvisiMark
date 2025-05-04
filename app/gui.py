import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QFileDialog, QTabWidget, QLineEdit, 
                             QComboBox, QCheckBox, QRadioButton, QButtonGroup, QGroupBox,
                             QMessageBox, QTextEdit, QScrollArea, QProgressBar, QSplitter,
                             QFrame, QSlider)
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize

import qrcode
from PIL import Image
import io
import base64

# Import the steganography, watermarking, and tamper detection modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.steganography import Steganography
from utils.watermarking import Watermarking
from models.tamper_detector import TamperDetector
from utils.image_saver import save_image

# Utility functions for the GUI
def convert_cv_to_pixmap(cv_img):
    """Convert OpenCV image to QPixmap"""
    height, width, channel = cv_img.shape
    bytes_per_line = 3 * width
    q_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
    return QPixmap.fromImage(q_img)

def load_image(file_path):
    """Load image from file and convert to OpenCV format"""
    try:
        img = cv2.imread(file_path)
        if img is None:
            return None
        return img
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def resize_pixmap(pixmap, max_width, max_height):
    """Resize QPixmap to fit within max dimensions while preserving aspect ratio"""
    if pixmap.width() > max_width or pixmap.height() > max_height:
        return pixmap.scaled(max_width, max_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return pixmap

class ProcessingThread(QThread):
    """Thread for running image processing operations in background"""
    finished = pyqtSignal(object, object, str)  # Results, key, message
    progress = pyqtSignal(int)  # Progress percentage
    error = pyqtSignal(str)  # Error message
    
    def __init__(self, operation, *args, **kwargs):
        super().__init__()
        self.operation = operation
        self.args = args
        self.kwargs = kwargs
    
    def run(self):
        try:
            if self.operation == 'steganography_encode':
                # Unpack args: steganography_instance, cover_image, message, method, encrypt
                stego, cover_img, message, method, encrypt = self.args
                
                # Update progress
                self.progress.emit(25)
                
                # Perform encoding
                stego_img, key = stego.encode(cover_img, message, method, encrypt=encrypt, **self.kwargs)
                
                self.progress.emit(90)
                
                # Return results
                self.finished.emit(stego_img, key, f"Message hidden using {method} method")
                
            elif self.operation == 'steganography_decode':
                # Unpack args: steganography_instance, stego_image, method, key, encrypted
                stego, stego_img, method, key, encrypted = self.args
                
                self.progress.emit(25)
                
                # Perform decoding
                message = stego.decode(stego_img, method, key, encrypted)
                
                self.progress.emit(90)
                
                # Return results
                self.finished.emit(message, None, f"Message extracted using {method} method")
                
            elif self.operation == 'watermark_embed':
                # Unpack args: watermarking_instance, image, watermark_text, method
                watermark, img, text, method = self.args
                
                self.progress.emit(25)
                
                # Perform watermarking
                watermarked_img, text = watermark.embed_watermark(img, text, method, **self.kwargs)
                
                self.progress.emit(90)
                
                # Return results
                self.finished.emit(watermarked_img, text, f"Watermark embedded using {method} method")
                
            elif self.operation == 'watermark_extract':
                # Unpack args: watermarking_instance, watermarked_image, watermark_text, method
                watermark, watermarked_img, text, method = self.args
                
                self.progress.emit(25)
                
                # Perform extraction
                result, score = watermark.extract_watermark(watermarked_img, text, method, **self.kwargs)
                
                self.progress.emit(90)
                
                # Return results
                self.finished.emit((result, score), None, f"Watermark verification score: {score:.2f}")
                
            elif self.operation == 'tampering_detect':
                # Unpack args: tamper_detector, image, model_type
                detector, img, model_type = self.args
                
                self.progress.emit(25)
                
                # Perform detection
                tampered, confidence = detector.detect_tampering(img, model_type)
                
                self.progress.emit(90)
                
                # Return results
                message = f"{'TAMPERED' if tampered else 'AUTHENTIC'} (confidence: {confidence:.2f})"
                self.finished.emit((tampered, confidence), None, message)
                
            else:
                self.error.emit(f"Unknown operation: {self.operation}")
                
            # Final progress
            self.progress.emit(100)
            
        except Exception as e:
            self.error.emit(f"Error in processing thread: {str(e)}")
            self.progress.emit(0)

class InvisiMarkApp(QMainWindow):
    """Main application window for InvisiMark"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize processing modules
        self.steganography = Steganography()
        self.watermarking = Watermarking()
        self.tamper_detector = TamperDetector()
        
        # Initialize state variables
        self.cover_image = None
        self.stego_image = None
        self.watermarked_image = None
        self.current_key = None
        self.current_watermark_text = None
        
        # Model loading state
        self.model_loaded = False
        
        # Setup UI
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle('InvisiMark - Steganography & Watermarking Tool')
        self.setGeometry(100, 100, 1200, 800)
        
        # Create main tab widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # Create tabs for different functionalities
        self.stego_tab = QWidget()
        self.watermarking_tab = QWidget()
        self.validation_tab = QWidget()
        
        # Add tabs to widget
        self.tabs.addTab(self.stego_tab, "Steganography")
        self.tabs.addTab(self.watermarking_tab, "Watermarking")
        self.tabs.addTab(self.validation_tab, "Validation & Tamper Detection")
        
        # Set up tab contents
        self.setup_steganography_tab()
        self.setup_watermarking_tab()
        self.setup_validation_tab()
        
        # Status bar for messages
        self.statusBar().showMessage('Ready')
        
        # Set style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QTabWidget::pane {
                border: 1px solid #999;
                background: white;
            }
            QTabBar::tab {
                background: #e4e4e4;
                border: 1px solid #999;
                padding: 5px 10px;
                min-width: 150px;
            }
            QTabBar::tab:selected {
                background: #4a86e8;
                color: white;
            }
            QPushButton {
                background-color: #4a86e8;
                color: white;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3d73c5;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #888888;
            }
            QLabel {
                font-size: 12px;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #999;
                margin-top: 10px;
                padding-top: 10px;
            }
        """)
        
        # Show the main window
        self.show()
        
    def setup_steganography_tab(self):
        """Set up the steganography tab with encode/decode sections"""
        layout = QVBoxLayout()
        
        # Create tabs for encode and decode
        stego_tabs = QTabWidget()
        encode_tab = QWidget()
        decode_tab = QWidget()
        
        stego_tabs.addTab(encode_tab, "Encode Message")
        stego_tabs.addTab(decode_tab, "Decode Message")
        
        # Encode tab
        encode_layout = QHBoxLayout()
        
        # Left panel for cover image
        left_panel = QVBoxLayout()
        
        # Cover image group
        cover_group = QGroupBox("Cover Image")
        cover_layout = QVBoxLayout()
        
        # Image display
        self.cover_image_label = QLabel("No image selected")
        self.cover_image_label.setAlignment(Qt.AlignCenter)
        self.cover_image_label.setMinimumSize(400, 300)
        self.cover_image_label.setStyleSheet("background-color: #e0e0e0; border: 1px solid #999;")
        
        # Load image button
        load_cover_btn = QPushButton("Load Cover Image")
        load_cover_btn.clicked.connect(self.load_cover_image)
        
        cover_layout.addWidget(self.cover_image_label)
        cover_layout.addWidget(load_cover_btn)
        cover_group.setLayout(cover_layout)
        
        left_panel.addWidget(cover_group)
        
        # Right panel for message input and encoding options
        right_panel = QVBoxLayout()
        
        # Message group
        message_group = QGroupBox("Secret Message")
        message_layout = QVBoxLayout()
        
        # Message input
        message_label = QLabel("Enter your secret message:")
        self.message_input = QTextEdit()
        
        # Method selection
        method_label = QLabel("Steganography Method:")
        self.stego_method_combo = QComboBox()
        self.stego_method_combo.addItems(["lsb"])
        
        # Encryption options
        self.encrypt_check = QCheckBox("Encrypt Message")
        self.encrypt_check.setChecked(True)
        
        # Encode button
        encode_btn = QPushButton("Encode Message")
        encode_btn.clicked.connect(self.encode_message)
        
        message_layout.addWidget(message_label)
        message_layout.addWidget(self.message_input)
        message_layout.addWidget(method_label)
        message_layout.addWidget(self.stego_method_combo)
        message_layout.addWidget(self.encrypt_check)
        message_layout.addWidget(encode_btn)
        message_group.setLayout(message_layout)
        
        # Output group
        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout()
        
        # Stego image display
        self.stego_image_label = QLabel("No stego image yet")
        self.stego_image_label.setAlignment(Qt.AlignCenter)
        self.stego_image_label.setMinimumSize(400, 200)
        self.stego_image_label.setStyleSheet("background-color: #e0e0e0; border: 1px solid #999;")
        
        # Save button
        save_stego_btn = QPushButton("Save Stego Image")
        save_stego_btn.clicked.connect(self.save_stego_image)
        
        # Key display
        key_label = QLabel("Encryption Key:")
        self.key_display = QLineEdit()
        self.key_display.setReadOnly(True)
        
        # Copy key button
        copy_key_btn = QPushButton("Copy Key")
        copy_key_btn.clicked.connect(self.copy_key_to_clipboard)
        
        output_layout.addWidget(self.stego_image_label)
        output_layout.addWidget(save_stego_btn)
        output_layout.addWidget(key_label)
        output_layout.addWidget(self.key_display)
        output_layout.addWidget(copy_key_btn)
        output_group.setLayout(output_layout)
        
        right_panel.addWidget(message_group)
        right_panel.addWidget(output_group)
        
        # Progress bar
        self.encode_progress = QProgressBar()
        self.encode_progress.setValue(0)
        right_panel.addWidget(self.encode_progress)
        
        encode_layout.addLayout(left_panel)
        encode_layout.addLayout(right_panel)
        encode_tab.setLayout(encode_layout)
        
        # Decode tab
        decode_layout = QHBoxLayout()
        
        # Left panel for stego image
        decode_left_panel = QVBoxLayout()
        
        # Stego image group
        stego_group = QGroupBox("Stego Image")
        stego_layout = QVBoxLayout()
        
        # Image display
        self.decode_image_label = QLabel("No image selected")
        self.decode_image_label.setAlignment(Qt.AlignCenter)
        self.decode_image_label.setMinimumSize(400, 300)
        self.decode_image_label.setStyleSheet("background-color: #e0e0e0; border: 1px solid #999;")
        
        # Load image button
        load_stego_btn = QPushButton("Load Stego Image")
        load_stego_btn.clicked.connect(self.load_stego_image)
        
        stego_layout.addWidget(self.decode_image_label)
        stego_layout.addWidget(load_stego_btn)
        stego_group.setLayout(stego_layout)
        
        decode_left_panel.addWidget(stego_group)
        
        # Right panel for decoding options and output
        decode_right_panel = QVBoxLayout()
        
        # Decoding options group
        decode_options_group = QGroupBox("Decoding Options")
        decode_options_layout = QVBoxLayout()
        
        # Method selection
        decode_method_label = QLabel("Steganography Method:")
        self.decode_method_combo = QComboBox()
        self.decode_method_combo.addItems(["lsb"])
        
        # Encryption options
        self.decrypt_check = QCheckBox("Message is Encrypted")
        self.decrypt_check.setChecked(True)
        
        # Key input
        key_input_label = QLabel("Decryption Key:")
        self.key_input = QLineEdit()
        
        # Decode button
        decode_btn = QPushButton("Decode Message")
        decode_btn.clicked.connect(self.decode_message)
        
        decode_options_layout.addWidget(decode_method_label)
        decode_options_layout.addWidget(self.decode_method_combo)
        decode_options_layout.addWidget(self.decrypt_check)
        decode_options_layout.addWidget(key_input_label)
        decode_options_layout.addWidget(self.key_input)
        decode_options_layout.addWidget(decode_btn)
        decode_options_group.setLayout(decode_options_layout)
        
        # Extracted message group
        extracted_group = QGroupBox("Extracted Message")
        extracted_layout = QVBoxLayout()
        
        # Extracted message display
        self.extracted_message = QTextEdit()
        self.extracted_message.setReadOnly(True)
        
        # Save message button
        save_message_btn = QPushButton("Save Message to File")
        save_message_btn.clicked.connect(self.save_extracted_message)
        
        extracted_layout.addWidget(self.extracted_message)
        extracted_layout.addWidget(save_message_btn)
        extracted_group.setLayout(extracted_layout)
        
        decode_right_panel.addWidget(decode_options_group)
        decode_right_panel.addWidget(extracted_group)
        
        # Progress bar
        self.decode_progress = QProgressBar()
        self.decode_progress.setValue(0)
        decode_right_panel.addWidget(self.decode_progress)
        
        decode_layout.addLayout(decode_left_panel)
        decode_layout.addLayout(decode_right_panel)
        decode_tab.setLayout(decode_layout)
        
        layout.addWidget(stego_tabs)
        self.stego_tab.setLayout(layout)
    
    def load_cover_image(self):
        """Load a cover image for steganography"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Cover Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        
        if file_path:
            # Load the image
            self.cover_image = load_image(file_path)
            
            if self.cover_image is not None:
                # Display the image
                pixmap = convert_cv_to_pixmap(self.cover_image)
                pixmap = resize_pixmap(pixmap, 400, 300)
                self.cover_image_label.setPixmap(pixmap)
                
                # Update status
                self.statusBar().showMessage(f"Loaded cover image: {os.path.basename(file_path)}")
            else:
                QMessageBox.warning(self, "Error", "Could not load the image file.")
    
    def load_stego_image(self):
        """Load a stego image for decoding"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Stego Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        
        if file_path:
            # Load the image
            self.stego_image = load_image(file_path)
            
            if self.stego_image is not None:
                # Display the image
                pixmap = convert_cv_to_pixmap(self.stego_image)
                pixmap = resize_pixmap(pixmap, 400, 300)
                self.decode_image_label.setPixmap(pixmap)
                
                # Update status
                self.statusBar().showMessage(f"Loaded stego image: {os.path.basename(file_path)}")
            else:
                QMessageBox.warning(self, "Error", "Could not load the image file.")
    
    def encode_message(self):
        """Encode a message into the cover image"""
        if self.cover_image is None:
            QMessageBox.warning(self, "Error", "Please load a cover image first.")
            return
        
        # Get the message
        message = self.message_input.toPlainText()
        if not message:
            QMessageBox.warning(self, "Error", "Please enter a message to hide.")
            return
        
        # Get the encoding method
        method = self.stego_method_combo.currentText()
        
        # Get encryption option
        encrypt = self.encrypt_check.isChecked()
        
        # Disable UI during processing
        self.encode_progress.setValue(0)
        
        # Create thread for processing
        self.encode_thread = ProcessingThread('steganography_encode', 
                                             self.steganography, 
                                             self.cover_image, 
                                             message, 
                                             method, 
                                             encrypt)
        
        # Connect signals
        self.encode_thread.progress.connect(self.encode_progress.setValue)
        self.encode_thread.finished.connect(self.encoding_finished)
        self.encode_thread.error.connect(self.process_error)
        
        # Start processing
        self.encode_thread.start()
        
        # Update status
        self.statusBar().showMessage(f"Encoding message using {method} method...")
    
    def encoding_finished(self, stego_img, key, message):
        """Handle completion of encoding process"""
        # Save the stego image
        self.stego_image = stego_img
        
        # Save the key
        self.current_key = key
        
        # Display the stego image
        pixmap = convert_cv_to_pixmap(stego_img)
        pixmap = resize_pixmap(pixmap, 400, 200)
        self.stego_image_label.setPixmap(pixmap)
        
        # Display the key if encrypted
        if key is not None:
            self.key_display.setText(key.decode())
        else:
            self.key_display.clear()
            
        # Update status
        self.statusBar().showMessage(message)
    
    def decode_message(self):
        """Decode a message from the stego image"""
        if self.stego_image is None:
            QMessageBox.warning(self, "Error", "Please load a stego image first.")
            return
        
        # Get the decoding method
        method = self.decode_method_combo.currentText()
        
        # Get encryption option
        encrypted = self.decrypt_check.isChecked()
        
        # Get key if needed
        key = None
        if encrypted:
            key_text = self.key_input.text()
            if not key_text:
                QMessageBox.warning(self, "Error", "Please enter a decryption key.")
                return
            key = key_text.encode()
        
        # Disable UI during processing
        self.decode_progress.setValue(0)
        
        # Create thread for processing
        self.decode_thread = ProcessingThread('steganography_decode', 
                                             self.steganography, 
                                             self.stego_image, 
                                             method, 
                                             key, 
                                             encrypted)
        
        # Connect signals
        self.decode_thread.progress.connect(self.decode_progress.setValue)
        self.decode_thread.finished.connect(self.decoding_finished)
        self.decode_thread.error.connect(self.process_error)
        
        # Start processing
        self.decode_thread.start()
        
        # Update status
        self.statusBar().showMessage(f"Decoding message using {method} method...")
    
    def decoding_finished(self, message, _, status_message):
        """Handle completion of decoding process"""
        # Display the extracted message
        self.extracted_message.setText(message)
        
        # Update status
        self.statusBar().showMessage(status_message)
    
    def save_stego_image(self):
        if self.stego_image is not None:
            filepath, _ = QFileDialog.getSaveFileName(self, "Save Stego Image", "", "PNG Images (*.png);;JPEG Images (*.jpg);;BMP Images (*.bmp);;All Files (*)")
            if filepath:
                try:
                    # Use the image_saver utility for reliable file saving
                    success, result = save_image(self.stego_image, filepath)
                    
                    if success:
                        self.stego_filepath = result
                        QMessageBox.information(self, "Success", f"Stego image saved to {os.path.basename(result)}")
                        self.statusBar().showMessage(f"Saved stego image to: {os.path.basename(result)}")
                    else:
                        QMessageBox.warning(self, "Warning", f"Could not save image: {result}")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to save stego image: {str(e)}")
    
    def save_extracted_message(self):
        """Save the extracted message to a file"""
        message = self.extracted_message.toPlainText()
        if not message:
            QMessageBox.warning(self, "Error", "No message to save.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Extracted Message", "", "Text Files (*.txt)")
        
        if file_path:
            with open(file_path, 'w') as f:
                f.write(message)
            
            # Update status
            self.statusBar().showMessage(f"Saved extracted message to: {os.path.basename(file_path)}")
    
    def copy_key_to_clipboard(self):
        """Copy the encryption key to clipboard"""
        if self.key_display.text():
            # Get the clipboard
            clipboard = QApplication.clipboard()
            
            # Set text to clipboard
            clipboard.setText(self.key_display.text())
            
            # Update status
            self.statusBar().showMessage("Encryption key copied to clipboard")

    def process_error(self, error_message):
        """Handle processing errors"""
        QMessageBox.warning(self, "Error", error_message)
        
    def setup_watermarking_tab(self):
        """Set up the watermarking tab"""
        layout = QVBoxLayout()
        
        # Create tabs for embedding and verification
        watermark_tabs = QTabWidget()
        embed_tab = QWidget()
        verify_tab = QWidget()
        
        watermark_tabs.addTab(embed_tab, "Embed Watermark")
        watermark_tabs.addTab(verify_tab, "Verify Watermark")
        
        # Embed tab
        embed_layout = QHBoxLayout()
        
        # Left panel for original image
        left_panel = QVBoxLayout()
        
        # Original image group
        original_group = QGroupBox("Original Image")
        original_layout = QVBoxLayout()
        
        # Image display
        self.original_image_label = QLabel("No image selected")
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setMinimumSize(400, 300)
        self.original_image_label.setStyleSheet("background-color: #e0e0e0; border: 1px solid #999;")
        
        # Load image button
        load_original_btn = QPushButton("Load Original Image")
        load_original_btn.clicked.connect(self.load_original_image)
        
        original_layout.addWidget(self.original_image_label)
        original_layout.addWidget(load_original_btn)
        original_group.setLayout(original_layout)
        
        left_panel.addWidget(original_group)
        
        # Right panel for watermark options and output
        right_panel = QVBoxLayout()
        
        # Watermark options group
        watermark_options_group = QGroupBox("Watermark Options")
        options_layout = QVBoxLayout()
        
        # Watermark text input
        watermark_label = QLabel("Watermark Text:")
        self.watermark_text = QLineEdit()
        self.watermark_text.setPlaceholderText("Enter watermark text (e.g., copyright info)")
        
        # Method selection
        method_label = QLabel("Watermark Method:")
        self.watermark_method_combo = QComboBox()
        self.watermark_method_combo.addItem("dct")
        
        # Strength slider (only for dwt and dct)
        strength_label = QLabel("Watermark Strength:")
        self.strength_slider = QSlider(Qt.Horizontal)
        self.strength_slider.setRange(1, 10)
        self.strength_slider.setValue(5)
        self.strength_value = QLabel("0.5")
        
        # Connect slider to value update
        self.strength_slider.valueChanged.connect(lambda v: self.strength_value.setText(f"{v/10:.1f}"))
        
        # Add anti-deepfake option
        self.add_fragile_check = QCheckBox("Add Anti-Deepfake Protection")
        self.add_fragile_check.setChecked(True)
        
        # Embed button
        embed_btn = QPushButton("Embed Watermark")
        embed_btn.clicked.connect(self.embed_watermark)
        
        options_layout.addWidget(watermark_label)
        options_layout.addWidget(self.watermark_text)
        options_layout.addWidget(method_label)
        options_layout.addWidget(self.watermark_method_combo)
        
        # Add strength slider in a horizontal layout
        strength_layout = QHBoxLayout()
        strength_layout.addWidget(strength_label)
        strength_layout.addWidget(self.strength_slider)
        strength_layout.addWidget(self.strength_value)
        options_layout.addLayout(strength_layout)
        
        options_layout.addWidget(self.add_fragile_check)
        options_layout.addWidget(embed_btn)
        watermark_options_group.setLayout(options_layout)
        
        # Output group
        output_group = QGroupBox("Watermarked Image")
        output_layout = QVBoxLayout()
        
        # Watermarked image display
        self.watermarked_image_label = QLabel("No watermarked image yet")
        self.watermarked_image_label.setAlignment(Qt.AlignCenter)
        self.watermarked_image_label.setMinimumSize(400, 200)
        self.watermarked_image_label.setStyleSheet("background-color: #e0e0e0; border: 1px solid #999;")
        
        # Save button
        save_watermarked_btn = QPushButton("Save Watermarked Image")
        save_watermarked_btn.clicked.connect(self.save_watermarked_image)
        
        output_layout.addWidget(self.watermarked_image_label)
        output_layout.addWidget(save_watermarked_btn)
        output_group.setLayout(output_layout)
        
        right_panel.addWidget(watermark_options_group)
        right_panel.addWidget(output_group)
        
        # Progress bar
        self.watermark_progress = QProgressBar()
        self.watermark_progress.setValue(0)
        right_panel.addWidget(self.watermark_progress)
        
        embed_layout.addLayout(left_panel)
        embed_layout.addLayout(right_panel)
        embed_tab.setLayout(embed_layout)
        
        # Verify tab
        verify_layout = QHBoxLayout()
        
        # Left panel for watermarked image
        verify_left_panel = QVBoxLayout()
        
        # Watermarked image group
        watermarked_group = QGroupBox("Watermarked Image")
        watermarked_layout = QVBoxLayout()
        
        # Image display
        self.verify_image_label = QLabel("No image selected")
        self.verify_image_label.setAlignment(Qt.AlignCenter)
        self.verify_image_label.setMinimumSize(400, 300)
        self.verify_image_label.setStyleSheet("background-color: #e0e0e0; border: 1px solid #999;")
        
        # Load image button
        load_watermarked_btn = QPushButton("Load Watermarked Image")
        load_watermarked_btn.clicked.connect(self.load_watermarked_image)
        
        watermarked_layout.addWidget(self.verify_image_label)
        watermarked_layout.addWidget(load_watermarked_btn)
        watermarked_group.setLayout(watermarked_layout)
        
        verify_left_panel.addWidget(watermarked_group)
        
        # Right panel for verification options and results
        verify_right_panel = QVBoxLayout()
        
        # Verification options group
        verify_options_group = QGroupBox("Verification Options")
        verify_options_layout = QVBoxLayout()
        
        # Watermark text input
        verify_text_label = QLabel("Watermark Text to Verify:")
        self.verify_text = QLineEdit()
        self.verify_text.setPlaceholderText("Enter the watermark text to verify")
        
        # Method selection
        verify_method_label = QLabel("Watermark Method:")
        self.verify_method_combo = QComboBox()
        self.verify_method_combo.addItem("dct")
        
        # Verify button
        verify_btn = QPushButton("Verify Watermark")
        verify_btn.clicked.connect(self.verify_watermark)
        
        verify_options_layout.addWidget(verify_text_label)
        verify_options_layout.addWidget(self.verify_text)
        verify_options_layout.addWidget(verify_method_label)
        verify_options_layout.addWidget(self.verify_method_combo)
        verify_options_layout.addWidget(verify_btn)
        verify_options_group.setLayout(verify_options_layout)
        
        # Results group
        results_group = QGroupBox("Verification Results")
        results_layout = QVBoxLayout()
        
        # Results display
        self.verify_results = QLabel("No verification results yet")
        self.verify_results.setAlignment(Qt.AlignCenter)
        self.verify_results.setMinimumHeight(100)
        self.verify_results.setStyleSheet("background-color: #e0e0e0; border: 1px solid #999; font-size: 14px;")
        
        # Visual result (extracted watermark or similarity map)
        self.verify_visual = QLabel("")
        self.verify_visual.setAlignment(Qt.AlignCenter)
        self.verify_visual.setMinimumSize(300, 150)
        self.verify_visual.setStyleSheet("background-color: #e0e0e0; border: 1px solid #999;")
        
        results_layout.addWidget(self.verify_results)
        results_layout.addWidget(self.verify_visual)
        results_group.setLayout(results_layout)
        
        verify_right_panel.addWidget(verify_options_group)
        verify_right_panel.addWidget(results_group)
        
        # Progress bar
        self.verify_progress = QProgressBar()
        self.verify_progress.setValue(0)
        verify_right_panel.addWidget(self.verify_progress)
        
        verify_layout.addLayout(verify_left_panel)
        verify_layout.addLayout(verify_right_panel)
        verify_tab.setLayout(verify_layout)
        
        layout.addWidget(watermark_tabs)
        self.watermarking_tab.setLayout(layout)
    
    def load_original_image(self):
        """Load an original image for watermarking"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Original Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        
        if file_path:
            # Load the image
            self.original_image = load_image(file_path)
            
            if self.original_image is not None:
                # Display the image
                pixmap = convert_cv_to_pixmap(self.original_image)
                pixmap = resize_pixmap(pixmap, 400, 300)
                self.original_image_label.setPixmap(pixmap)
                
                # Update status
                self.statusBar().showMessage(f"Loaded original image: {os.path.basename(file_path)}")
            else:
                QMessageBox.warning(self, "Error", "Could not load the image file.")
    
    def load_watermarked_image(self):
        """Load a watermarked image for verification"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Watermarked Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        
        if file_path:
            # Load the image
            self.watermarked_image = load_image(file_path)
            
            if self.watermarked_image is not None:
                # Display the image
                pixmap = convert_cv_to_pixmap(self.watermarked_image)
                pixmap = resize_pixmap(pixmap, 400, 300)
                self.verify_image_label.setPixmap(pixmap)
                
                # Update status
                self.statusBar().showMessage(f"Loaded watermarked image: {os.path.basename(file_path)}")
            else:
                QMessageBox.warning(self, "Error", "Could not load the image file.")
    
    def embed_watermark(self):
        """Embed a watermark into the original image"""
        if self.original_image is None:
            QMessageBox.warning(self, "Error", "Please load an original image first.")
            return
        
        # Get the watermark text
        watermark_text = self.watermark_text.text()
        if not watermark_text:
            QMessageBox.warning(self, "Error", "Please enter watermark text.")
            return
        
        # Get the watermarking method
        method = self.watermark_method_combo.currentText()
        
        # Get strength value
        alpha = self.strength_slider.value() / 10.0
        
        # Prepare additional args
        kwargs = {'alpha': alpha}
        
        # Handle anti-deepfake option
        if self.add_fragile_check.isChecked() and method != 'fragile':
            # First apply the regular watermark
            self.watermark_progress.setValue(0)
            self.statusBar().showMessage(f"Embedding watermark using {method} method...")
            
            # Create thread for processing
            self.embed_thread = ProcessingThread('watermark_embed', 
                                               self.watermarking, 
                                               self.original_image.copy(), 
                                               watermark_text, 
                                               method,
                                               **kwargs)
            
            # Connect signals
            self.embed_thread.progress.connect(self.watermark_progress.setValue)
            self.embed_thread.finished.connect(lambda img, text, msg: self.embed_fragile_signature(img, text))
            self.embed_thread.error.connect(self.process_error)
            
            # Start processing
            self.embed_thread.start()
        else:
            # Just apply the selected watermark
            self.watermark_progress.setValue(0)
            
            # Create thread for processing
            self.embed_thread = ProcessingThread('watermark_embed', 
                                               self.watermarking, 
                                               self.original_image, 
                                               watermark_text, 
                                               method,
                                               **kwargs)
            
            # Connect signals
            self.embed_thread.progress.connect(self.watermark_progress.setValue)
            self.embed_thread.finished.connect(self.watermarking_finished)
            self.embed_thread.error.connect(self.process_error)
            
            # Start processing
            self.embed_thread.start()
            
            # Update status
            self.statusBar().showMessage(f"Embedding watermark using {method} method...")
    
    def embed_fragile_signature(self, watermarked_img, watermark_text):
        """Add a fragile signature on top of already watermarked image"""
        # Add "FRAGILE" to the watermark text
        fragile_text = f"{watermark_text}_FRAGILE"
        
        # Apply fragile watermark
        self.embed_thread = ProcessingThread('watermark_embed', 
                                           self.watermarking, 
                                           watermarked_img, 
                                           fragile_text, 
                                           'fragile')
        
        # Connect signals
        self.embed_thread.progress.connect(self.watermark_progress.setValue)
        self.embed_thread.finished.connect(self.watermarking_finished)
        self.embed_thread.error.connect(self.process_error)
        
        # Start processing
        self.embed_thread.start()
        
        # Update status
        self.statusBar().showMessage("Adding anti-deepfake protection...")
    
    def watermarking_finished(self, watermarked_img, watermark_text, message):
        """Handle completion of watermarking process"""
        # Save the watermarked image
        self.watermarked_image = watermarked_img
        
        # Save the watermark text
        self.current_watermark_text = watermark_text
        
        # Display the watermarked image
        pixmap = convert_cv_to_pixmap(watermarked_img)
        pixmap = resize_pixmap(pixmap, 400, 200)
        self.watermarked_image_label.setPixmap(pixmap)
        
        # Update status
        self.statusBar().showMessage(message)
    
    def verify_watermark(self):
        """Verify a watermark in the image"""
        if self.watermarked_image is None:
            QMessageBox.warning(self, "Error", "Please load a watermarked image first.")
            return
        
        # Get the watermark text
        watermark_text = self.verify_text.text()
        if not watermark_text:
            QMessageBox.warning(self, "Error", "Please enter watermark text to verify.")
            return
        
        # Get the watermarking method
        method = self.verify_method_combo.currentText()
        
        # Clear previous results
        self.verify_results.setText("Verifying...")
        self.verify_visual.clear()
        
        # Disable UI during processing
        self.verify_progress.setValue(0)
        
        # Create thread for processing
        self.verify_thread = ProcessingThread('watermark_extract', 
                                            self.watermarking, 
                                            self.watermarked_image, 
                                            watermark_text, 
                                            method)
        
        # Connect signals
        self.verify_thread.progress.connect(self.verify_progress.setValue)
        self.verify_thread.finished.connect(self.verification_finished)
        self.verify_thread.error.connect(self.process_error)
        
        # Start processing
        self.verify_thread.start()
        
        # Update status
        self.statusBar().showMessage(f"Verifying watermark using {method} method...")
    
    def verification_finished(self, result, _, message):
        """Handle completion of watermark verification process"""
        extracted, score = result
        
        # Update results display
        if score > 0.7:
            # Good score, likely authentic
            self.verify_results.setText(f"✅ AUTHENTIC - Similarity: {score:.2f}")
            self.verify_results.setStyleSheet("background-color: #c8e6c9; border: 1px solid #999; font-size: 14px; font-weight: bold;")
        elif score > 0.5:
            # Medium score, might be authentic but manipulated
            self.verify_results.setText(f"⚠️ POSSIBLY MODIFIED - Similarity: {score:.2f}")
            self.verify_results.setStyleSheet("background-color: #fff9c4; border: 1px solid #999; font-size: 14px; font-weight: bold;")
        else:
            # Low score, likely tampered
            self.verify_results.setText(f"❌ TAMPERED - Similarity: {score:.2f}")
            self.verify_results.setStyleSheet("background-color: #ffcdd2; border: 1px solid #999; font-size: 14px; font-weight: bold;")
        
        # If the result is a visual, display it
        if isinstance(extracted, np.ndarray):
            # Normalize and scale the extracted watermark for visualization
            if extracted.dtype == np.bool or np.max(extracted) <= 1.0:
                visual = (extracted * 255).astype(np.uint8)
            else:
                visual = extracted.astype(np.uint8)
            
            # Ensure 3 channels for display
            if len(visual.shape) == 2:
                visual = cv2.cvtColor(visual, cv2.COLOR_GRAY2BGR)
            
            # Display
            pixmap = convert_cv_to_pixmap(visual)
            pixmap = resize_pixmap(pixmap, 300, 150)
            self.verify_visual.setPixmap(pixmap)
        
        # Update status
        self.statusBar().showMessage(message)
    
    def save_watermarked_image(self):
        if self.watermarked_image is not None:
            filepath, _ = QFileDialog.getSaveFileName(self, "Save Watermarked Image", "", "PNG Images (*.png);;JPEG Images (*.jpg);;BMP Images (*.bmp);;All Files (*)")
            if filepath:
                try:
                    # Use the image_saver utility for reliable file saving
                    success, result = save_image(self.watermarked_image, filepath)
                    
                    if success:
                        self.watermarked_filepath = result
                        QMessageBox.information(self, "Success", f"Watermarked image saved to {os.path.basename(result)}")
                        self.statusBar().showMessage(f"Saved watermarked image to: {os.path.basename(result)}")
                    else:
                        QMessageBox.warning(self, "Warning", f"Could not save image: {result}")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to save watermarked image: {str(e)}")
                    
    def setup_validation_tab(self):
        """Set up the validation and tamper detection tab"""
        layout = QVBoxLayout()
        
        # Main layout with two sections
        main_layout = QSplitter(Qt.Vertical)
        
        # Top section for tamper detection
        tamper_widget = QWidget()
        tamper_layout = QVBoxLayout()
        
        # Tamper detection title
        tamper_title = QLabel("AI-Powered Tamper Detection")
        tamper_title.setFont(QFont('Arial', 14, QFont.Bold))
        tamper_title.setAlignment(Qt.AlignCenter)
        tamper_layout.addWidget(tamper_title)
        
        # Tamper detection content
        tamper_content = QHBoxLayout()
        
        # Left panel for image input
        tamper_left = QVBoxLayout()
        
        # Image group
        image_group = QGroupBox("Image to Analyze")
        image_layout = QVBoxLayout()
        
        # Image display
        self.tamper_image_label = QLabel("No image selected")
        self.tamper_image_label.setAlignment(Qt.AlignCenter)
        self.tamper_image_label.setMinimumSize(400, 300)
        self.tamper_image_label.setStyleSheet("background-color: #e0e0e0; border: 1px solid #999;")
        
        # Load image button
        load_tamper_btn = QPushButton("Load Image")
        load_tamper_btn.clicked.connect(self.load_tamper_image)
        
        image_layout.addWidget(self.tamper_image_label)
        image_layout.addWidget(load_tamper_btn)
        image_group.setLayout(image_layout)
        
        tamper_left.addWidget(image_group)
        
        # Right panel for detection options and results
        tamper_right = QVBoxLayout()
        
        # Options group
        options_group = QGroupBox("Detection Options")
        options_layout = QVBoxLayout()
        
        # Model type selection
        model_label = QLabel("Detection Model:")
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["cnn", "autoencoder"])
        
        # Model status
        self.model_status = QLabel("Model Status: Not loaded")
        self.model_status.setStyleSheet("color: #d32f2f;")
        
        # Load model button
        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.clicked.connect(self.load_tamper_model)
        
        # Detect button
        self.detect_btn = QPushButton("Detect Tampering")
        self.detect_btn.clicked.connect(self.detect_tampering)
        self.detect_btn.setEnabled(False)
        
        # Train model button (optional for demo)
        self.train_model_btn = QPushButton("Train Demo Model")
        self.train_model_btn.clicked.connect(self.train_demo_model)
        
        options_layout.addWidget(model_label)
        options_layout.addWidget(self.model_type_combo)
        options_layout.addWidget(self.model_status)
        options_layout.addWidget(self.load_model_btn)
        options_layout.addWidget(self.detect_btn)
        options_layout.addWidget(self.train_model_btn)
        options_group.setLayout(options_layout)
        
        # Results group
        results_group = QGroupBox("Detection Results")
        results_layout = QVBoxLayout()
        
        # Results display
        self.tamper_results = QLabel("No detection results yet")
        self.tamper_results.setAlignment(Qt.AlignCenter)
        self.tamper_results.setMinimumHeight(100)
        self.tamper_results.setStyleSheet("background-color: #e0e0e0; border: 1px solid #999; font-size: 14px;")
        
        # Confidence meter
        confidence_layout = QHBoxLayout()
        confidence_label = QLabel("Confidence:")
        self.confidence_progress = QProgressBar()
        self.confidence_progress.setRange(0, 100)
        self.confidence_progress.setValue(0)
        
        confidence_layout.addWidget(confidence_label)
        confidence_layout.addWidget(self.confidence_progress)
        
        results_layout.addWidget(self.tamper_results)
        results_layout.addLayout(confidence_layout)
        results_group.setLayout(results_layout)
        
        tamper_right.addWidget(options_group)
        tamper_right.addWidget(results_group)
        
        # Progress bar
        self.tamper_progress = QProgressBar()
        self.tamper_progress.setValue(0)
        tamper_right.addWidget(self.tamper_progress)
        
        tamper_content.addLayout(tamper_left)
        tamper_content.addLayout(tamper_right)
        
        tamper_layout.addLayout(tamper_content)
        tamper_widget.setLayout(tamper_layout)
        
        # Bottom section for deepfake detection
        deepfake_widget = QWidget()
        deepfake_layout = QVBoxLayout()
        
        # Deepfake detection title
        deepfake_title = QLabel("Anti-Deepfake Verification")
        deepfake_title.setFont(QFont('Arial', 14, QFont.Bold))
        deepfake_title.setAlignment(Qt.AlignCenter)
        deepfake_layout.addWidget(deepfake_title)
        
        # Deepfake content
        deepfake_content = QHBoxLayout()
        
        # Info panel
        deepfake_info = QGroupBox("About Anti-Deepfake Protection")
        info_layout = QVBoxLayout()
        
        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_text.setHtml("""
        <h3>Anti-Deepfake Protection</h3>
        <p>Images processed with InvisiMark's anti-deepfake feature contain a special 
        fragile watermark that can detect if the image has been used in a deepfake generation process.</p>
        
        <p>This protection works by embedding a fragile signature that breaks if the image is:</p>
        <ul>
            <li>Used in a deepfake model</li>
            <li>Modified with style transfer</li>
            <li>Manipulated or edited</li>
            <li>Recompressed significantly</li>
        </ul>
        
        <p>To verify if an image contains an intact fragile watermark:</p>
        <ol>
            <li>Load a watermarked image</li>
            <li>Enter the original watermark text with "_FRAGILE" appended</li>
            <li>Use the "fragile" method to verify</li>
        </ol>
        
        <p>A low verification score indicates the image may have been tampered with or used in deepfake generation.</p>
        """)
        
        info_layout.addWidget(info_text)
        deepfake_info.setLayout(info_layout)
        
        # Quickcheck panel
        deepfake_check = QGroupBox("Quick Deepfake Check")
        check_layout = QVBoxLayout()
        
        # Quick check instructions
        check_instructions = QLabel("Check if an image has been modified/used in a deepfake:")
        
        # Text input
        fragile_text_label = QLabel("Enter the original watermark text + \"_FRAGILE\":")
        self.fragile_text = QLineEdit()
        self.fragile_text.setPlaceholderText("e.g., Copyright2023_FRAGILE")
        
        # Check button
        fragile_check_btn = QPushButton("Check Fragile Watermark")
        fragile_check_btn.clicked.connect(self.check_fragile_watermark)
        
        # Results display
        self.fragile_results = QLabel("No check results yet")
        self.fragile_results.setAlignment(Qt.AlignCenter)
        self.fragile_results.setMinimumHeight(60)
        self.fragile_results.setStyleSheet("background-color: #e0e0e0; border: 1px solid #999; font-size: 14px;")
        
        check_layout.addWidget(check_instructions)
        check_layout.addWidget(fragile_text_label)
        check_layout.addWidget(self.fragile_text)
        check_layout.addWidget(fragile_check_btn)
        check_layout.addWidget(self.fragile_results)
        deepfake_check.setLayout(check_layout)
        
        deepfake_content.addWidget(deepfake_info)
        deepfake_content.addWidget(deepfake_check)
        
        deepfake_layout.addLayout(deepfake_content)
        deepfake_widget.setLayout(deepfake_layout)
        
        # Add widgets to splitter
        main_layout.addWidget(tamper_widget)
        main_layout.addWidget(deepfake_widget)
        
        # Set initial sizes
        main_layout.setSizes([500, 300])
        
        layout.addWidget(main_layout)
        self.validation_tab.setLayout(layout)
    
    def load_tamper_image(self):
        """Load an image for tamper detection"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image for Tamper Detection", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        
        if file_path:
            # Load the image
            self.tamper_image = load_image(file_path)
            
            if self.tamper_image is not None:
                # Display the image
                pixmap = convert_cv_to_pixmap(self.tamper_image)
                pixmap = resize_pixmap(pixmap, 400, 300)
                self.tamper_image_label.setPixmap(pixmap)
                
                # Update status
                self.statusBar().showMessage(f"Loaded image for analysis: {os.path.basename(file_path)}")
                
                # Enable the detect button if model is loaded
                if self.model_loaded:
                    self.detect_btn.setEnabled(True)
            else:
                QMessageBox.warning(self, "Error", "Could not load the image file.")
    
    def load_tamper_model(self):
        """Load the tamper detection model"""
        try:
            # In a real app, we would load a pre-trained model
            model_type = self.model_type_combo.currentText()
            
            # For demonstration, we're initializing a new model
            self.tamper_detector.build_model(model_type)
            
            # Update UI
            self.model_loaded = True
            self.model_status.setText(f"Model Status: Loaded ({model_type})")
            self.model_status.setStyleSheet("color: #2e7d32;")
            
            # Enable detect button if an image is loaded
            if hasattr(self, 'tamper_image') and self.tamper_image is not None:
                self.detect_btn.setEnabled(True)
                
            # Update status
            self.statusBar().showMessage(f"Loaded {model_type} tamper detection model")
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load model: {str(e)}")
    
    def train_demo_model(self):
        """Train a demo model for tamper detection"""
        # Check if we have an image to train on
        if not hasattr(self, 'tamper_image') or self.tamper_image is None:
            QMessageBox.warning(self, "Error", "Please load an image first for demo training.")
            return
        
        # Disable buttons during training
        self.train_model_btn.setEnabled(False)
        self.detect_btn.setEnabled(False)
        self.load_model_btn.setEnabled(False)
        
        # Update status
        self.model_status.setText("Model Status: Training...")
        self.statusBar().showMessage("Training demo model (this is just a simulation for the demo)")
        
        # Get model type
        model_type = self.model_type_combo.currentText()
        
        # Build the model first
        self.tamper_detector.build_model(model_type)
        
        # Create a minimal training dataset from the current image
        # This is just for demo purposes
        images = [self.tamper_image]
        tampered_types = ['noise', 'blur', 'jpeg', 'crop']
        
        # Generate training data
        self.tamper_progress.setValue(10)
        X, y = self.tamper_detector.generate_training_data(images, tampered_types)
        
        # Update progress
        self.tamper_progress.setValue(30)
        
        # Train the model with minimal epochs
        self.tamper_detector.train(X, y, model_type, epochs=2, batch_size=2)
        
        # Update UI
        self.tamper_progress.setValue(100)
        self.model_loaded = True
        self.model_status.setText(f"Model Status: Trained Demo ({model_type})")
        self.model_status.setStyleSheet("color: #2e7d32;")
        
        # Re-enable buttons
        self.train_model_btn.setEnabled(True)
        self.detect_btn.setEnabled(True)
        self.load_model_btn.setEnabled(True)
        
        # Update status
        self.statusBar().showMessage("Demo model trained successfully! (Note: This is a simplified demo)")
    
    def detect_tampering(self):
        """Detect if an image has been tampered with"""
        if not self.model_loaded:
            QMessageBox.warning(self, "Error", "Please load or train a model first.")
            return
            
        if not hasattr(self, 'tamper_image') or self.tamper_image is None:
            QMessageBox.warning(self, "Error", "Please load an image for analysis.")
            return
        
        # Clear previous results
        self.tamper_results.setText("Analyzing...")
        self.confidence_progress.setValue(0)
        
        # Update status
        self.statusBar().showMessage("Analyzing image for tampering...")
        
        # Get model type
        model_type = self.model_type_combo.currentText()
        
        # Create thread for processing
        self.tamper_thread = ProcessingThread('tampering_detect', 
                                            self.tamper_detector, 
                                            self.tamper_image, 
                                            model_type)
        
        # Connect signals
        self.tamper_thread.progress.connect(self.tamper_progress.setValue)
        self.tamper_thread.finished.connect(self.tampering_detection_finished)
        self.tamper_thread.error.connect(self.process_error)
        
        # Start processing
        self.tamper_thread.start()
    
    def tampering_detection_finished(self, result, _, message):
        """Handle completion of tamper detection process"""
        tampered, confidence = result
        
        # Update results display
        if tampered:
            self.tamper_results.setText("❌ IMAGE TAMPERED")
            self.tamper_results.setStyleSheet("background-color: #ffcdd2; border: 1px solid #999; font-size: 18px; font-weight: bold;")
        else:
            self.tamper_results.setText("✅ IMAGE AUTHENTIC")
            self.tamper_results.setStyleSheet("background-color: #c8e6c9; border: 1px solid #999; font-size: 18px; font-weight: bold;")
        
        # Update confidence meter
        confidence_pct = int(confidence * 100)
        self.confidence_progress.setValue(confidence_pct)
        
        # Update progress bar color based on confidence
        if confidence > 0.7:
            self.confidence_progress.setStyleSheet("QProgressBar::chunk { background-color: #4caf50; }")
        elif confidence > 0.5:
            self.confidence_progress.setStyleSheet("QProgressBar::chunk { background-color: #ff9800; }")
        else:
            self.confidence_progress.setStyleSheet("QProgressBar::chunk { background-color: #f44336; }")
            
        # Update status
        self.statusBar().showMessage(message)
    
    def check_fragile_watermark(self):
        """Check if a fragile watermark is intact"""
        if not hasattr(self, 'tamper_image') or self.tamper_image is None:
            QMessageBox.warning(self, "Error", "Please load an image first.")
            return
            
        # Get the watermark text
        fragile_text = self.fragile_text.text()
        if not fragile_text:
            QMessageBox.warning(self, "Error", "Please enter watermark text with _FRAGILE suffix.")
            return
            
        # Clear previous results
        self.fragile_results.setText("Checking fragile watermark...")
        
        # Verify the fragile watermark
        try:
            intact, score = self.watermarking.extract_watermark(self.tamper_image, fragile_text, 'fragile')
            
            # Update results display
            if score > 0.7:
                self.fragile_results.setText(f"✅ WATERMARK INTACT - Score: {score:.2f}")
                self.fragile_results.setStyleSheet("background-color: #c8e6c9; border: 1px solid #999; font-size: 14px; font-weight: bold;")
                self.statusBar().showMessage("Fragile watermark is intact. Image has NOT been used in deepfake generation.")
            else:
                self.fragile_results.setText(f"❌ WATERMARK BROKEN - Score: {score:.2f}")
                self.fragile_results.setStyleSheet("background-color: #ffcdd2; border: 1px solid #999; font-size: 14px; font-weight: bold;")
                self.statusBar().showMessage("Fragile watermark is broken! Image may have been tampered with or used in deepfake generation.")
                
        except Exception as e:
            self.fragile_results.setText("Error checking watermark")
            self.fragile_results.setStyleSheet("background-color: #fff3e0; border: 1px solid #999; font-size: 14px;")
            self.statusBar().showMessage(f"Error: {str(e)}")
            
# Main function to start the application
def main(show_info=True):
    # Fix for Qt platform plugin error
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''
    if 'DISPLAY' not in os.environ:
        os.environ['DISPLAY'] = ':0'
        
    try:
        app = QApplication(sys.argv)
        window = InvisiMarkApp()
        
        # Show information about LSB-only steganography if requested
        if show_info:
            QMessageBox.information(None, "InvisiMark: Simplified Edition", 
                "This version uses only the LSB steganography method for better reliability.\n\n"
                "The DCT and QR code methods have been removed.\n\n"
                "For more details, see the simplified_steganography.md file.")
        
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Error starting application: {e}")
        
        # Provide troubleshooting information if it's a Qt platform plugin error
        if "Could not load the Qt platform plugin" in str(e):
            print("\nTROUBLESHOOTING: Qt platform plugin error detected")
            print("1. Install the required system packages:")
            print("   sudo apt-get install libxcb-xinerama0 libxcb-icccm4 libxcb-image0")
            print("   sudo apt-get install libxcb-keysyms1 libxcb-render-util0 libxcb-randr0")
            print("   sudo apt-get install libxcb-xkb1 libxkbcommon-x11-0")
            print("\n2. Reinstall PyQt5 with:")
            print("   pip uninstall PyQt5")
            print("   pip install PyQt5 --config-settings --confirm-license= --verbose")
            
            # Fallback to CLI mode
            print("\nWould you like to run the CLI demo instead? (y/n)")
            choice = input().lower().strip()
            if choice == 'y':
                from importlib import import_module
                demo = import_module('demo')
                demo.main()
        else:
            # General error
            print("\nPlease check the installation requirements and try again.")
            print("You can still use the command-line demo with: python demo.py") 