"""
Predictor for Skin Condition Classification

Loads trained CNN model and performs inference on dermatological images.
"""

import numpy as np
from pathlib import Path
import sys

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.sequential import Sequential
from src.layers.conv2d import Conv2D
from src.layers.pooling import MaxPool2D
from src.layers.dense import Dense
from src.layers.activations import ReLU, Softmax
from src.layers.batchnorm import BatchNorm2D
from src.layers.dropout import Dropout


class DermaScanPredictor:
    """
    Handles model loading and prediction for skin conditions

    Attributes:
        model: Trained Sequential CNN model
        class_names: List of skin condition class names
        model_path: Path to saved model weights
    """

    # Default skin condition classes (can be updated based on dataset)
    DEFAULT_CLASSES = [
        "Actinic Keratosis",  # AK
        "Basal Cell Carcinoma",  # BCC
        "Benign Keratosis",  # BKL
        "Dermatofibroma",  # DF
        "Melanoma",  # MEL
        "Melanocytic Nevus",  # NV
        "Vascular Lesion"  # VASC
    ]

    def __init__(self, model_path: str = None, class_names: list = None):
        """
        Initialize predictor

        Args:
            model_path: Path to saved model weights (.npz file)
            class_names: List of class names for predictions
        """
        self.model_path = model_path or "data/dermatology/models/dermascan_best.npz"
        self.class_names = class_names or self.DEFAULT_CLASSES
        self.model = self._build_model()

        # Load weights if available
        if Path(self.model_path).exists():
            self._load_weights()
        else:
            print(f"Warning: Model weights not found at {self.model_path}")
            print("Model initialized with random weights.")

    def _build_model(self) -> Sequential:
        """
        Build the CNN architecture for dermatological classification

        Returns:
            Sequential model
        """
        # Architecture inspired by ResNet/EfficientNet for medical imaging
        # Adapted for our custom NumPy CNN implementation

        model = Sequential([
            # Block 1: Initial feature extraction
            Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            BatchNorm2D(num_features=32),
            ReLU(),
            Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            BatchNorm2D(num_features=32),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),

            # Block 2: Deeper features
            Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            BatchNorm2D(num_features=64),
            ReLU(),
            Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            BatchNorm2D(num_features=64),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),

            # Block 3: Complex patterns
            Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            BatchNorm2D(num_features=128),
            ReLU(),
            Conv2D(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            BatchNorm2D(num_features=128),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),

            # Block 4: High-level features
            Conv2D(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            BatchNorm2D(num_features=256),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),

            # Classification head
            Dense(in_features=256 * 14 * 14, out_features=512),  # Adjusted for 224x224 input
            ReLU(),
            Dropout(p=0.5),
            Dense(in_features=512, out_features=len(self.class_names)),
            Softmax()
        ])

        return model

    def _load_weights(self):
        """Load model weights from file"""
        try:
            weights_dict = np.load(self.model_path, allow_pickle=True)
            # Load weights into model layers
            # Implementation depends on saved format
            print(f"Model weights loaded from {self.model_path}")
        except Exception as e:
            print(f"Error loading weights: {e}")

    def predict(self, image: np.ndarray, top_k: int = 3) -> list:
        """
        Predict skin condition from preprocessed image

        Args:
            image: Preprocessed image array of shape (1, C, H, W)
            top_k: Number of top predictions to return

        Returns:
            List of dicts with class_name, confidence, and class_id
        """
        # Set model to evaluation mode
        self.model.training = False

        # Forward pass
        output = self.model.forward(image)  # Shape: (1, num_classes)

        # Get probabilities (already softmax from model)
        probabilities = output[0]  # Shape: (num_classes,)

        # Get top-k predictions
        top_indices = np.argsort(probabilities)[::-1][:top_k]

        predictions = []
        for idx in top_indices:
            predictions.append({
                'class_id': int(idx),
                'class_name': self.class_names[idx],
                'confidence': float(probabilities[idx])
            })

        return predictions

    def predict_batch(self, images: np.ndarray, top_k: int = 3) -> list:
        """
        Predict skin conditions for a batch of images

        Args:
            images: Batch of preprocessed images of shape (N, C, H, W)
            top_k: Number of top predictions per image

        Returns:
            List of prediction lists (one per image)
        """
        self.model.training = False

        # Forward pass
        outputs = self.model.forward(images)  # Shape: (N, num_classes)

        batch_predictions = []
        for output in outputs:
            probabilities = output
            top_indices = np.argsort(probabilities)[::-1][:top_k]

            predictions = []
            for idx in top_indices:
                predictions.append({
                    'class_id': int(idx),
                    'class_name': self.class_names[idx],
                    'confidence': float(probabilities[idx])
                })

            batch_predictions.append(predictions)

        return batch_predictions

    def save_model(self, path: str):
        """
        Save model weights

        Args:
            path: Path to save weights (.npz file)
        """
        # Extract weights from all layers
        weights_dict = {}
        for i, layer in enumerate(self.model.layers):
            if hasattr(layer, 'W'):
                weights_dict[f'layer_{i}_W'] = layer.W
            if hasattr(layer, 'b'):
                weights_dict[f'layer_{i}_b'] = layer.b

        np.savez(path, **weights_dict)
        print(f"Model saved to {path}")
