import cv2
import numpy as np
from scipy.fftpack import dct, idct
import hashlib

class Watermarking:
    def __init__(self):
        """Initialize watermarking tools"""
        # Only DCT method is now available
        self.methods = {
            'dct': self.dct_embed
        }
        
        self.extraction_methods = {
            'dct': self.dct_extract
        }
    
    def _generate_watermark(self, text, shape):
        """
        Generate a binary watermark from text
        
        Args:
            text: text to be used as watermark
            shape: shape of the watermark (rows, cols)
            
        Returns:
            watermark: binary watermark image
        """
        # Create a seed from the text
        seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % 10000000
        np.random.seed(seed)
        
        # Generate random binary watermark
        watermark = np.random.randint(0, 2, shape).astype(np.float32)
        
        return watermark
    
    def _image_to_blocks(self, image, block_size=8):
        """Split image into blocks of size block_size x block_size"""
        height, width = image.shape[:2]
        blocks = []
        
        for i in range(0, height - height % block_size, block_size):
            for j in range(0, width - width % block_size, block_size):
                blocks.append(image[i:i+block_size, j:j+block_size])
        
        return blocks
    
    def dct_embed(self, image, watermark_text, alpha=0.1):
        """
        Embed watermark using Discrete Cosine Transform
        
        Args:
            image: numpy array of image
            watermark_text: text watermark
            alpha: strength of watermark
            
        Returns:
            watermarked: watermarked image
        """
        # Copy the image
        watermarked = image.copy()
        
        # Convert to grayscale if needed
        if len(image.shape) > 2:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image
        
        # Generate watermark
        rows, cols = gray_image.shape
        wm_size = (rows // 8, cols // 8)  # One bit per 8x8 block
        watermark = self._generate_watermark(watermark_text, wm_size)
        
        # Flatten the watermark
        watermark_flat = watermark.flatten()
        
        # Process the image in 8x8 blocks
        blocks = self._image_to_blocks(gray_image)
        
        block_idx = 0
        for i in range(0, rows - rows % 8, 8):
            for j in range(0, cols - cols % 8, 8):
                if block_idx < len(watermark_flat):
                    # Get the current 8x8 block
                    block = gray_image[i:i+8, j:j+8].astype(float)
                    
                    # Apply DCT
                    dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                    
                    # Embed watermark in mid-frequency component
                    if watermark_flat[block_idx] > 0:
                        dct_block[4, 4] = np.ceil(dct_block[4, 4] / alpha) * alpha
                    else:
                        dct_block[4, 4] = np.floor(dct_block[4, 4] / alpha) * alpha
                    
                    # Apply inverse DCT
                    watermarked_block = idct(idct(dct_block.T, norm='ortho').T, norm='ortho')
                    
                    # Update the image
                    if len(image.shape) > 2:
                        # For color images, update Y channel in YCrCb
                        if i == 0 and j == 0:  # First time only, convert to YCrCb
                            watermarked_ycrcb = cv2.cvtColor(watermarked, cv2.COLOR_BGR2YCrCb)
                        
                        # Update Y channel
                        watermarked_ycrcb[i:i+8, j:j+8, 0] = np.clip(watermarked_block, 0, 255).astype(np.uint8)
                        
                        # Convert back to BGR if this is the last block
                        if block_idx == len(watermark_flat) - 1:
                            watermarked = cv2.cvtColor(watermarked_ycrcb, cv2.COLOR_YCrCb2BGR)
                    else:
                        # For grayscale images, update directly
                        watermarked[i:i+8, j:j+8] = np.clip(watermarked_block, 0, 255).astype(np.uint8)
                    
                    block_idx += 1
        
        return watermarked, watermark_text
    
    def dct_extract(self, watermarked, watermark_text, threshold=0):
        """
        Extract watermark using Discrete Cosine Transform
        
        Args:
            watermarked: watermarked image
            watermark_text: text used to generate watermark
            threshold: detection threshold
            
        Returns:
            extracted: extracted watermark
            similarity: similarity score
        """
        # Convert to grayscale
        if len(watermarked.shape) > 2:
            gray_image = cv2.cvtColor(watermarked, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = watermarked
        
        # Generate expected watermark
        rows, cols = gray_image.shape
        wm_size = (rows // 8, cols // 8)
        expected_watermark = self._generate_watermark(watermark_text, wm_size)
        
        # Flatten expected watermark
        expected_flat = expected_watermark.flatten()
        
        # Initialize extracted watermark
        extracted_flat = np.zeros_like(expected_flat)
        
        # Process the image in 8x8 blocks
        block_idx = 0
        for i in range(0, rows - rows % 8, 8):
            for j in range(0, cols - cols % 8, 8):
                if block_idx < len(expected_flat):
                    # Get the current 8x8 block
                    block = gray_image[i:i+8, j:j+8].astype(float)
                    
                    # Apply DCT
                    dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                    
                    # Check mid-frequency component
                    u, v = 4, 4
                    value = dct_block[u, v]
                    
                    # Determine bit based on coefficient magnitude
                    extracted_flat[block_idx] = 1 if value > threshold else 0
                    
                    block_idx += 1
        
        # Reshape to original watermark size
        extracted = extracted_flat.reshape(wm_size)
        
        # Calculate similarity with expected watermark
        similarity = np.sum(extracted == expected_watermark) / expected_watermark.size
        
        return extracted, similarity
    
    def embed_watermark(self, image, watermark_text, method='dct', **kwargs):
        """
        Wrapper method to embed a watermark using the DCT method
        
        Args:
            image: numpy array of image
            watermark_text: text watermark
            method: watermarking method (only 'dct' is supported)
            **kwargs: additional arguments for the DCT method
            
        Returns:
            watermarked: watermarked image
            watermark_text: the watermark text (for verification)
        """
        if method not in self.methods:
            raise ValueError(f"Method {method} not supported. Only DCT method is available.")
        
        return self.methods[method](image, watermark_text, **kwargs)
    
    def extract_watermark(self, watermarked, watermark_text, method='dct', **kwargs):
        """
        Wrapper method to extract or verify a watermark
        
        Args:
            watermarked: watermarked or manipulated image
            watermark_text: text used for watermarking
            method: watermarking method (only 'dct' is supported)
            **kwargs: additional arguments for the DCT method
            
        Returns:
            extracted: extracted watermark
            metric: similarity score
        """
        if method not in self.extraction_methods:
            raise ValueError(f"Method {method} not supported. Only DCT method is available.")
        
        return self.extraction_methods[method](watermarked, watermark_text, **kwargs) 