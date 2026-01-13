"""
GradCAM Explainer
Generate visual explanations for model predictions using Gradient-weighted Class Activation Mapping.
"""

import torch
import numpy as np
import cv2
from typing import Dict, Tuple, Optional
from loguru import logger


class GradCAMExplainer:
    """Generate visual explanations using GradCAM for medical image analysis"""
    
    def __init__(self, model=None, target_layer=None):
        """
        Initialize GradCAM explainer.
        
        Args:
            model: PyTorch model (optional, can set later)
            target_layer: Layer to compute gradients from
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._hooks = []
        
        if model and target_layer:
            self._register_hooks()
    
    def set_model(self, model, target_layer):
        """Set model and target layer after initialization"""
        self.model = model
        self.target_layer = target_layer
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on target layer"""
        # Clear existing hooks
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        
        # Forward hook to capture activations
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        # Backward hook to capture gradients
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))
    
    def generate_heatmap(
        self, 
        image_tensor: torch.Tensor, 
        target_class: int = None
    ) -> np.ndarray:
        """
        Generate GradCAM heatmap for target class.
        
        Args:
            image_tensor: Input image tensor [1, C, H, W]
            target_class: Class index to explain (None = use predicted class)
            
        Returns:
            Heatmap as numpy array [H, W]
        """
        if self.model is None:
            logger.warning("No model set for GradCAM")
            return np.zeros((224, 224))
        
        self.model.eval()
        
        # Forward pass
        output = self.model(image_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass for target class
        self.model.zero_grad()
        
        if len(output.shape) == 2:
            # Classification output
            target = output[0, target_class]
        else:
            # Handle other output shapes
            target = output.flatten()[target_class]
        
        target.backward()
        
        if self.gradients is None or self.activations is None:
            logger.warning("Gradients or activations not captured")
            return np.zeros((224, 224))
        
        # Compute GradCAM
        # Global average pool the gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        
        # ReLU to keep positive contributions
        cam = torch.relu(cam)
        
        # Normalize to 0-1
        cam = cam - cam.min()
        cam_max = cam.max()
        if cam_max > 0:
            cam = cam / cam_max
        
        # Convert to numpy and resize
        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        
        return cam
    
    def overlay_heatmap(
        self, 
        image: np.ndarray, 
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Overlay heatmap on original image.
        
        Args:
            image: Original image [H, W] or [H, W, C]
            heatmap: GradCAM heatmap [H, W]
            alpha: Blending factor (0=image only, 1=heatmap only)
            colormap: OpenCV colormap
            
        Returns:
            Blended image [H, W, 3]
        """
        # Convert heatmap to colormap
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
        
        # Handle grayscale images
        if len(image.shape) == 2:
            image = cv2.cvtColor(
                (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8),
                cv2.COLOR_GRAY2BGR
            )
        elif image.max() <= 1:
            image = (image * 255).astype(np.uint8)
        
        # Resize if needed
        if image.shape[:2] != heatmap_colored.shape[:2]:
            heatmap_colored = cv2.resize(heatmap_colored, (image.shape[1], image.shape[0]))
        
        # Blend
        blended = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
        
        return blended
    
    def explain_prediction(
        self, 
        image: np.ndarray, 
        image_tensor: torch.Tensor,
        class_names: list = None,
        target_class: int = None
    ) -> Dict:
        """
        Generate complete explanation for a prediction.
        
        Args:
            image: Original image for overlay
            image_tensor: Preprocessed tensor for model
            class_names: List of class names
            target_class: Class to explain
            
        Returns:
            Dict with heatmap, overlay, and metadata
        """
        # Generate heatmap
        heatmap = self.generate_heatmap(image_tensor, target_class)
        
        # Create overlay
        overlay = self.overlay_heatmap(image, heatmap)
        
        # Find top activated regions
        threshold = 0.5
        high_activation_mask = heatmap > threshold
        high_activation_percent = np.mean(high_activation_mask) * 100
        
        # Get centroid of high activation region
        if high_activation_mask.any():
            y_indices, x_indices = np.where(high_activation_mask)
            centroid = (int(np.mean(x_indices)), int(np.mean(y_indices)))
        else:
            centroid = (112, 112)  # Center
        
        result = {
            "heatmap": heatmap,
            "overlay": overlay,
            "target_class": target_class,
            "class_name": class_names[target_class] if class_names and target_class is not None else None,
            "high_activation_percent": high_activation_percent,
            "attention_centroid": centroid,
            "explanation": f"Model focused on {high_activation_percent:.1f}% of the image area"
        }
        
        return result
    
    def __del__(self):
        """Clean up hooks"""
        for hook in self._hooks:
            hook.remove()
