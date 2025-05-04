import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import cv2
import os
import pickle
from sklearn.model_selection import train_test_split

class TamperDetector:
    def __init__(self, input_shape=(128, 128, 3)):
        """
        Initialize the tamper detection model
        
        Args:
            input_shape: shape of input images
        """
        self.input_shape = input_shape
        self.cnn_model = None
        self.autoencoder_model = None
        self.models = {
            'cnn': self._build_cnn_model,
            'autoencoder': self._build_autoencoder_model
        }
    
    def _build_cnn_model(self):
        """Build a lightweight CNN for tamper detection"""
        model = models.Sequential([
            layers.Input(shape=self.input_shape),
            
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Flatten and dense layers
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _build_autoencoder_model(self):
        """Build an autoencoder for anomaly detection"""
        # Encoder
        input_img = layers.Input(shape=self.input_shape)
        
        # Encoder
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
        
        # Decoder
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        decoded = layers.Conv2D(self.input_shape[2], (3, 3), activation='sigmoid', padding='same')(x)
        
        # Autoencoder model
        autoencoder = models.Model(input_img, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return autoencoder
    
    def build_model(self, model_type='cnn'):
        """
        Build the specified model type
        
        Args:
            model_type: 'cnn' or 'autoencoder'
            
        Returns:
            model: the constructed model
        """
        if model_type not in self.models:
            raise ValueError(f"Model type {model_type} not supported. Choose from {list(self.models.keys())}")
        
        if model_type == 'cnn':
            self.cnn_model = self.models[model_type]()
            return self.cnn_model
        elif model_type == 'autoencoder':
            self.autoencoder_model = self.models[model_type]()
            return self.autoencoder_model
    
    def _preprocess_image(self, image):
        """
        Preprocess image for model input
        
        Args:
            image: input image
            
        Returns:
            processed: preprocessed image
        """
        # Resize if needed
        if image.shape[:2] != self.input_shape[:2]:
            image = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))
        
        # Ensure 3 channels
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Normalize to 0-1
        image = image.astype('float32') / 255.0
        
        return image
    
    def generate_training_data(self, original_images, tamper_types=['noise', 'blur', 'jpeg', 'crop']):
        """
        Generate training data by applying different tamper operations
        
        Args:
            original_images: list of original images
            tamper_types: list of tampering types to apply
            
        Returns:
            X: processed images
            y: labels (0: tampered, 1: original)
        """
        X = []
        y = []
        
        for image in original_images:
            # Preprocess original image
            processed = self._preprocess_image(image)
            X.append(processed)
            y.append(1)  # Original
            
            # Generate tampered versions
            for tamper_type in tamper_types:
                tampered = self._apply_tampering(image, tamper_type)
                tampered_processed = self._preprocess_image(tampered)
                X.append(tampered_processed)
                y.append(0)  # Tampered
        
        return np.array(X), np.array(y)
    
    def _apply_tampering(self, image, tamper_type):
        """
        Apply different tampering operations on the image
        
        Args:
            image: input image
            tamper_type: type of tampering to apply
            
        Returns:
            tampered: tampered image
        """
        tampered = image.copy()
        
        if tamper_type == 'noise':
            # Add random noise
            noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
            tampered = cv2.add(tampered, noise)
        
        elif tamper_type == 'blur':
            # Apply Gaussian blur
            tampered = cv2.GaussianBlur(tampered, (5, 5), 0)
        
        elif tamper_type == 'jpeg':
            # JPEG compression artifacts
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 15]  # Low quality
            _, buffer = cv2.imencode('.jpg', tampered, encode_param)
            tampered = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
        
        elif tamper_type == 'crop':
            # Small crop and resize back
            height, width = tampered.shape[:2]
            crop_factor = 0.9
            x = int(width * (1 - crop_factor) / 2)
            y = int(height * (1 - crop_factor) / 2)
            w = int(width * crop_factor)
            h = int(height * crop_factor)
            
            cropped = tampered[y:y+h, x:x+w]
            tampered = cv2.resize(cropped, (width, height))
            
        elif tamper_type == 'deepfake':
            # Simulate deepfake artifacts
            # Apply color shift and slight blurring
            hsv = cv2.cvtColor(tampered, cv2.COLOR_BGR2HSV)
            hsv[:, :, 0] = (hsv[:, :, 0] + 10) % 180  # Hue shift
            tampered = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            tampered = cv2.GaussianBlur(tampered, (3, 3), 0)
        
        return tampered
    
    def train(self, X, y, model_type='cnn', validation_split=0.2, epochs=10, batch_size=32):
        """
        Train the model
        
        Args:
            X: training images
            y: training labels
            model_type: 'cnn' or 'autoencoder'
            validation_split: fraction of data to use for validation
            epochs: number of training epochs
            batch_size: batch size for training
            
        Returns:
            history: training history
        """
        if model_type == 'cnn':
            if self.cnn_model is None:
                self.build_model('cnn')
            
            history = self.cnn_model.fit(
                X, y,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size
            )
            return history
        
        elif model_type == 'autoencoder':
            if self.autoencoder_model is None:
                self.build_model('autoencoder')
            
            # For autoencoder, train only on original images
            X_orig = X[y == 1]
            
            history = self.autoencoder_model.fit(
                X_orig, X_orig,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size
            )
            
            # Compute reconstruction error threshold
            reconstructions = self.autoencoder_model.predict(X_orig)
            mse = np.mean(np.square(X_orig - reconstructions), axis=(1, 2, 3))
            
            # Set threshold as mean + 2*std of reconstruction errors
            self.reconstruction_threshold = np.mean(mse) + 2 * np.std(mse)
            
            return history
    
    def detect_tampering(self, image, model_type='cnn'):
        """
        Detect if an image has been tampered with
        
        Args:
            image: input image
            model_type: 'cnn' or 'autoencoder'
            
        Returns:
            tampered: boolean indicating if image is tampered
            confidence: confidence score
        """
        processed = self._preprocess_image(image)
        processed = np.expand_dims(processed, axis=0)  # Add batch dimension
        
        if model_type == 'cnn':
            if self.cnn_model is None:
                raise ValueError("CNN model not trained. Call train() first.")
            
            prediction = self.cnn_model.predict(processed)[0][0]
            tampered = prediction < 0.5
            confidence = 1 - prediction if tampered else prediction
            
            return tampered, float(confidence)
        
        elif model_type == 'autoencoder':
            if self.autoencoder_model is None:
                raise ValueError("Autoencoder model not trained. Call train() first.")
            
            # Reconstruct the image
            reconstruction = self.autoencoder_model.predict(processed)
            
            # Calculate reconstruction error
            mse = np.mean(np.square(processed - reconstruction))
            
            # If error is above threshold, image is tampered
            tampered = mse > self.reconstruction_threshold
            
            # Normalize confidence score (0-1)
            confidence = min(1.0, mse / (2 * self.reconstruction_threshold)) if tampered else (1.0 - mse / self.reconstruction_threshold)
            
            return tampered, float(confidence)
    
    def save_model(self, filepath, model_type='cnn'):
        """
        Save the model
        
        Args:
            filepath: path to save model
            model_type: 'cnn' or 'autoencoder'
        """
        if model_type == 'cnn':
            if self.cnn_model is None:
                raise ValueError("CNN model not trained. Call train() first.")
            self.cnn_model.save(filepath)
            
        elif model_type == 'autoencoder':
            if self.autoencoder_model is None:
                raise ValueError("Autoencoder model not trained. Call train() first.")
            
            # Save the model
            self.autoencoder_model.save(filepath)
            
            # Save the reconstruction threshold
            threshold_path = filepath + '_threshold.pkl'
            with open(threshold_path, 'wb') as f:
                pickle.dump(self.reconstruction_threshold, f)
    
    def load_model(self, filepath, model_type='cnn'):
        """
        Load a trained model
        
        Args:
            filepath: path to load model from
            model_type: 'cnn' or 'autoencoder'
            
        Returns:
            model: loaded model
        """
        if model_type == 'cnn':
            self.cnn_model = models.load_model(filepath)
            return self.cnn_model
            
        elif model_type == 'autoencoder':
            # Load the model
            self.autoencoder_model = models.load_model(filepath)
            
            # Load the reconstruction threshold
            threshold_path = filepath + '_threshold.pkl'
            if os.path.exists(threshold_path):
                with open(threshold_path, 'rb') as f:
                    self.reconstruction_threshold = pickle.load(f)
            else:
                # Set a default threshold
                self.reconstruction_threshold = 0.1
                print("Warning: Threshold file not found. Using default threshold.")
                
            return self.autoencoder_model 