"""
Image Processor for Dermatological Images

Handles preprocessing of skin images for model inference.
"""

import numpy as np
from PIL import Image
import io
from typing import Union


class ImageProcessor:
    """
    Processes dermatological images for CNN inference

    Attributes:
        target_size: Target image dimensions (height, width)
        mean: Normalization mean values per channel
        std: Normalization std values per channel
    """

    def __init__(self, target_size=(224, 224)):
        """
        Initialize image processor

        Args:
            target_size: Target dimensions for resizing (H, W)
        """
        self.target_size = target_size
        # ImageNet normalization (can be updated with dataset-specific values)
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def process_uploaded_image(self, image_bytes: bytes) -> np.ndarray:
        """
        Process an uploaded image file

        Args:
            image_bytes: Raw image bytes from upload

        Returns:
            Preprocessed image array of shape (1, C, H, W)
        """
        # Load image from bytes
        image = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Process the image
        processed = self.process_image(image)

        # Add batch dimension
        return np.expand_dims(processed, axis=0)

    def process_image(self, image: Image.Image) -> np.ndarray:
        """
        Process a PIL image

        Args:
            image: PIL Image object

        Returns:
            Preprocessed image array of shape (C, H, W)
        """
        # Resize
        image = image.resize(self.target_size, Image.LANCZOS)

        # Convert to numpy array and normalize to [0, 1]
        img_array = np.array(image, dtype=np.float32) / 255.0

        # Normalize using mean and std
        img_array = (img_array - self.mean) / self.std

        # Convert from (H, W, C) to (C, H, W)
        img_array = np.transpose(img_array, (2, 0, 1))

        return img_array

    def denormalize(self, image: np.ndarray) -> np.ndarray:
        """
        Reverse normalization for visualization

        Args:
            image: Normalized image array (C, H, W)

        Returns:
            Denormalized image array (H, W, C) in range [0, 255]
        """
        # Convert from (C, H, W) to (H, W, C)
        img = np.transpose(image, (1, 2, 0))

        # Denormalize
        img = (img * self.std) + self.mean

        # Clip to [0, 1] and scale to [0, 255]
        img = np.clip(img, 0, 1) * 255

        return img.astype(np.uint8)

    def augment_image(self, image: np.ndarray,
                     rotation: int = 0,
                     flip_horizontal: bool = False,
                     flip_vertical: bool = False) -> np.ndarray:
        """
        Apply data augmentation transformations

        Args:
            image: Input image array (C, H, W)
            rotation: Rotation angle in degrees (0, 90, 180, 270)
            flip_horizontal: Whether to flip horizontally
            flip_vertical: Whether to flip vertically

        Returns:
            Augmented image array
        """
        img = image.copy()

        # Convert to (H, W, C) for transformations
        img = np.transpose(img, (1, 2, 0))

        # Apply rotation
        if rotation != 0:
            k = rotation // 90
            img = np.rot90(img, k=k)

        # Apply flips
        if flip_horizontal:
            img = np.fliplr(img)
        if flip_vertical:
            img = np.flipud(img)

        # Convert back to (C, H, W)
        img = np.transpose(img, (2, 0, 1))

        return img
