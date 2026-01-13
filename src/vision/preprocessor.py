"""
Image Preprocessor
Preprocess medical images with quality checks before analysis.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, List
from loguru import logger


class ImagePreprocessor:
    """Preprocess medical images with quality validation"""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize preprocessor.
        
        Args:
            target_size: Output image dimensions (height, width)
        """
        self.target_size = target_size
        self.min_sharpness = 100.0  # Laplacian variance threshold
        logger.info(f"ImagePreprocessor initialized (target: {target_size})")
    
    def preprocess(
        self, 
        image_path: str
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        Preprocess image with quality validation.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (processed_image, error_message)
            - If success: (normalized_image, None)
            - If failure: (None, error_description)
        """
        try:
            # Load image as grayscale
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return None, f"Failed to load image: {image_path}"
            
            # Quality check: Sharpness
            sharpness = cv2.Laplacian(image, cv2.CV_64F).var()
            if sharpness < self.min_sharpness:
                logger.warning(f"Low sharpness: {sharpness:.2f}")
                return None, f"Image too blurry (sharpness: {sharpness:.1f}). Please retake."
            
            # Apply CLAHE for contrast normalization
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            image = clahe.apply(image)
            
            # Resize to target size
            image = cv2.resize(image, self.target_size)
            
            # Normalize to 0-1 range
            image = image.astype(np.float32) / 255.0
            
            # Convert to 3-channel for CLIP models
            image = np.stack([image] * 3, axis=-1)
            
            return image, None
            
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            return None, f"Processing error: {str(e)}"
    
    def preprocess_pil(self, pil_image: Image.Image) -> np.ndarray:
        """
        Preprocess PIL Image directly.
        
        Args:
            pil_image: PIL Image object
            
        Returns:
            Preprocessed numpy array
        """
        # Convert to grayscale
        if pil_image.mode != 'L':
            pil_image = pil_image.convert('L')
        
        # Convert to numpy
        image = np.array(pil_image)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
        
        # Resize
        image = cv2.resize(image, self.target_size)
        
        # Normalize and expand channels
        image = image.astype(np.float32) / 255.0
        image = np.stack([image] * 3, axis=-1)
        
        return image
    
    def detect_artifacts(self, image: np.ndarray) -> List[str]:
        """
        Detect common image artifacts that may affect diagnosis.
        
        Args:
            image: Preprocessed image array (0-1 range)
            
        Returns:
            List of detected artifact types
        """
        artifacts = []
        
        # Handle 3-channel images by taking first channel
        if len(image.shape) == 3:
            img = image[:, :, 0]
        else:
            img = image
        
        # Check for metal artifacts (very bright spots)
        bright_pixels = np.sum(img > 0.95) / img.size
        if bright_pixels > 0.01:
            artifacts.append("metal_artifact_possible")
            logger.warning(f"Possible metal artifact: {bright_pixels:.1%} bright pixels")
        
        # Check for motion blur using Laplacian variance
        img_uint8 = (img * 255).astype(np.uint8)
        laplacian_var = cv2.Laplacian(img_uint8, cv2.CV_64F).var()
        if laplacian_var < 50:
            artifacts.append("motion_blur_detected")
            logger.warning(f"Possible motion blur: Laplacian variance = {laplacian_var:.1f}")
        
        # Check for exposure issues
        mean_brightness = np.mean(img)
        if mean_brightness < 0.2:
            artifacts.append("underexposed")
            logger.warning(f"Underexposed image: mean = {mean_brightness:.2f}")
        elif mean_brightness > 0.8:
            artifacts.append("overexposed")
            logger.warning(f"Overexposed image: mean = {mean_brightness:.2f}")
        
        return artifacts
    
    def check_image_quality(
        self, 
        image: np.ndarray
    ) -> dict:
        """
        Comprehensive image quality assessment.
        
        Args:
            image: Input image array
            
        Returns:
            Dict with quality metrics and assessment
        """
        # Handle 3-channel images
        if len(image.shape) == 3:
            img = image[:, :, 0]
        else:
            img = image
        
        img_uint8 = (img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)
        
        # Calculate metrics
        sharpness = cv2.Laplacian(img_uint8, cv2.CV_64F).var()
        contrast = img.std()
        brightness = img.mean()
        
        # Detect artifacts
        artifacts = self.detect_artifacts(image)
        
        # Quality assessment
        quality_score = 1.0
        issues = []
        
        if sharpness < self.min_sharpness:
            quality_score -= 0.3
            issues.append("low_sharpness")
        
        if contrast < 0.1:
            quality_score -= 0.2
            issues.append("low_contrast")
        
        if brightness < 0.2 or brightness > 0.8:
            quality_score -= 0.2
            issues.append("exposure_issue")
        
        if artifacts:
            quality_score -= 0.1 * len(artifacts)
            issues.extend(artifacts)
        
        return {
            "quality_score": max(0, quality_score),
            "sharpness": sharpness,
            "contrast": contrast,
            "brightness": brightness,
            "artifacts": artifacts,
            "issues": issues,
            "acceptable": quality_score >= 0.6,
            "needs_review": quality_score < 0.8
        }
