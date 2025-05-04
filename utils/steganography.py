import cv2
import numpy as np
from PIL import Image
import os
from cryptography.fernet import Fernet
import base64

class Steganography:
    def __init__(self):
        """Initialize steganography tools"""
        self.methods = {
            'lsb': self.lsb_encode,
        }
        
        self.extraction_methods = {
            'lsb': self.lsb_decode,
        }
    
    def _generate_encryption_key(self):
        """Generate a key for encryption/decryption"""
        return Fernet.generate_key()
    
    def _encrypt_message(self, message, key=None):
        """Encrypt a message using Fernet symmetric encryption"""
        if key is None:
            key = self._generate_encryption_key()
            
        cipher = Fernet(key)
        encrypted = cipher.encrypt(message.encode())
        return encrypted, key
    
    def _decrypt_message(self, encrypted_message, key):
        """Decrypt a message using Fernet symmetric encryption"""
        cipher = Fernet(key)
        return cipher.decrypt(encrypted_message).decode()
    
    def _text_to_binary(self, text):
        """Convert text to binary representation"""
        if type(text) == str:
            text = text.encode()
        
        binary = ''.join([format(byte, '08b') for byte in text])
        return binary
    
    def _binary_to_text(self, binary):
        """Convert binary representation to text"""
        bytes_data = int(binary, 2).to_bytes((len(binary) + 7) // 8, byteorder='big')
        return bytes_data
    
    def lsb_encode(self, cover_image, message, key=None, encrypt=True):
        """
        Hide message in the least significant bit of the image
        
        Args:
            cover_image: numpy array of image
            message: string to hide
            key: encryption key (optional)
            encrypt: whether to encrypt the message
            
        Returns:
            stego_image: image with hidden message
            key: encryption key (if generated)
        """
        # Make a copy of the image
        stego_image = cover_image.copy()
        
        # Encrypt message if requested
        if encrypt:
            message_bytes, key = self._encrypt_message(message, key)
            binary_message = self._text_to_binary(message_bytes)
        else:
            binary_message = self._text_to_binary(message)
        
        # Add message length at the beginning
        message_length = len(binary_message)
        length_binary = format(message_length, '032b')  # 32 bits for length
        binary_message = length_binary + binary_message
        
        # Check if image is large enough
        total_pixels = cover_image.shape[0] * cover_image.shape[1]
        if len(binary_message) > total_pixels * 3:
            raise ValueError("Message too large for the cover image")
        
        # Flatten the image
        flat_image = stego_image.reshape(-1)
        
        # Hide the message - fixed to prevent out-of-bounds integer error
        idx = 0
        for i in range(len(binary_message)):
            if idx >= len(flat_image):
                break
                
            bit = int(binary_message[i])
            # Use safer bit manipulation method
            pixel_value = flat_image[idx]
            
            # Clear the LSB (set to 0) and then OR with the bit value
            if bit == 1:
                flat_image[idx] = (pixel_value & 0xFE) | 1  # Set LSB to 1
            else:
                flat_image[idx] = pixel_value & 0xFE  # Set LSB to 0
                
            idx += 1
        
        # Reshape back to the original shape
        stego_image = flat_image.reshape(cover_image.shape)
        
        return stego_image, key
    
    def lsb_decode(self, stego_image, key=None, encrypted=True):
        """
        Extract message hidden in the least significant bit
        
        Args:
            stego_image: image with hidden message
            key: decryption key (if message was encrypted)
            encrypted: whether the message was encrypted
            
        Returns:
            message: extracted message
        """
        # Flatten the image
        flat_image = stego_image.reshape(-1)
        
        # Extract the length first (32 bits)
        length_binary = ''
        for i in range(32):
            bit = flat_image[i] & 1
            length_binary += str(bit)
        
        message_length = int(length_binary, 2)
        
        # Validate message length
        if message_length <= 0 or message_length > len(flat_image) - 32:
            raise ValueError(f"Invalid message length detected: {message_length}")
        
        # Extract the message
        binary_message = ''
        for i in range(32, 32 + message_length):
            if i < len(flat_image):
                bit = flat_image[i] & 1
                binary_message += str(bit)
        
        # Convert binary to bytes
        try:
            message_bytes = self._binary_to_text(binary_message)
        except Exception as e:
            raise ValueError(f"Failed to convert binary to text: {str(e)}")
        
        # Decrypt if necessary
        if encrypted:
            if key is None:
                raise ValueError("Decryption key is required")
            try:
                message = self._decrypt_message(message_bytes, key)
            except Exception as e:
                raise ValueError(f"Failed to decrypt message: {str(e)}")
        else:
            try:
                message = message_bytes.decode()
            except UnicodeDecodeError:
                raise ValueError("Failed to decode message. The message may be corrupted or encrypted.")
        
        return message
    
    def encode(self, cover_image, message, method='lsb', key=None, encrypt=True, **kwargs):
        """
        Wrapper method to encode a message using the specified method
        
        Args:
            cover_image: numpy array of image
            message: string to hide
            method: steganography method (only 'lsb' supported)
            key: encryption key (optional)
            encrypt: whether to encrypt the message
            **kwargs: additional arguments for specific methods
            
        Returns:
            stego_image: image with hidden message
            key: encryption key (if generated)
        """
        if method not in self.methods:
            raise ValueError(f"Method {method} not supported. Choose from {list(self.methods.keys())}")
        
        return self.methods[method](cover_image, message, key, encrypt, **kwargs)
    
    def decode(self, stego_image, method='lsb', key=None, encrypted=True):
        """
        Wrapper method to decode a message using the specified method
        
        Args:
            stego_image: image with hidden message
            method: steganography method (only 'lsb' supported)
            key: decryption key (if message was encrypted)
            encrypted: whether the message was encrypted
            
        Returns:
            message: extracted message
        """
        if method not in self.extraction_methods:
            raise ValueError(f"Method {method} not supported. Choose from {list(self.extraction_methods.keys())}")
        
        return self.extraction_methods[method](stego_image, key, encrypted) 