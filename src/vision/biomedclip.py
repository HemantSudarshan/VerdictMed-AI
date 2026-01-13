"""
BiomedCLIP Analyzer
Medical image analysis using BiomedCLIP for zero-shot classification.
"""

import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Union
from loguru import logger

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed. Using mock vision analyzer.")


class BiomedCLIPAnalyzer:
    """Medical image analysis using BiomedCLIP"""
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize BiomedCLIP analyzer.
        
        Args:
            device: Device to run model on ('cuda', 'cpu', or None for auto)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self._loaded = False
        
        # Medical finding prompts for chest X-rays
        self.chest_xray_findings = [
            "normal chest radiograph",
            "pneumonia with lung consolidation",
            "pulmonary edema",
            "pleural effusion",
            "lung nodule or mass",
            "cardiomegaly",
            "pneumothorax",
            "atelectasis",
            "tuberculosis pattern"
        ]
        
        logger.info(f"BiomedCLIPAnalyzer initialized (device: {self.device})")
    
    def _load_model(self):
        """Lazy load the model on first use"""
        if self._loaded:
            return
        
        try:
            import open_clip
            
            logger.info(f"Loading BiomedCLIP on {self.device}...")
            
            self.model, self.preprocess, _ = open_clip.create_model_and_transforms(
                'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self.tokenizer = open_clip.get_tokenizer(
                'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
            )
            
            self._loaded = True
            logger.info("BiomedCLIP loaded successfully")
            
        except ImportError:
            logger.warning("open_clip not installed. Using mock mode.")
            self._loaded = False
        except Exception as e:
            logger.error(f"Failed to load BiomedCLIP: {e}")
            self._loaded = False
    
    def analyze_chest_xray(
        self, 
        image: Union[str, np.ndarray, Image.Image],
        custom_prompts: Optional[List[str]] = None
    ) -> Dict:
        """
        Analyze chest X-ray and return findings with confidence.
        
        Args:
            image: Image path, numpy array, or PIL Image
            custom_prompts: Optional custom finding prompts
            
        Returns:
            Dict with findings, top finding, confidence, and review flags
        """
        # Load model if not already loaded
        self._load_model()
        
        if not self._loaded:
            return self._mock_analysis()
        
        try:
            # Convert to PIL Image
            pil_image = self._to_pil(image)
            
            # Preprocess
            image_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            
            # Use custom prompts or defaults
            prompts = custom_prompts or self.chest_xray_findings
            
            # Tokenize prompts
            text_tokens = self.tokenizer(prompts).to(self.device)
            
            # Get embeddings and calculate similarity
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                text_features = self.model.encode_text(text_tokens)
                
                # Normalize
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarity scores
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                probs = similarity[0].cpu().numpy()
            
            # Build results
            findings = []
            for i, (finding, prob) in enumerate(zip(prompts, probs)):
                findings.append({
                    "finding": finding,
                    "confidence": float(prob)
                })
            
            # Sort by confidence
            findings.sort(key=lambda x: x["confidence"], reverse=True)
            
            top_finding = findings[0]
            
            # Safety checks
            needs_review = False
            review_reason = None
            
            # Low confidence check
            if top_finding["confidence"] < 0.4:
                needs_review = True
                review_reason = "Low confidence - ambiguous image"
            
            # Close second finding check
            if len(findings) > 1:
                diff = findings[0]["confidence"] - findings[1]["confidence"]
                if diff < 0.1:
                    needs_review = True
                    review_reason = f"Uncertain between {findings[0]['finding']} and {findings[1]['finding']}"
            
            return {
                "findings": findings[:5],  # Top 5
                "top_finding": top_finding["finding"],
                "confidence": top_finding["confidence"],
                "needs_review": needs_review,
                "review_reason": review_reason,
                "all_scores": {f["finding"]: f["confidence"] for f in findings}
            }
            
        except Exception as e:
            logger.error(f"BiomedCLIP analysis error: {e}")
            return {
                "findings": [],
                "top_finding": "analysis_failed",
                "confidence": 0.0,
                "needs_review": True,
                "review_reason": f"Analysis failed: {str(e)}"
            }
    
    def get_embedding(self, image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Get image embedding for similarity search.
        
        Args:
            image: Input image
            
        Returns:
            Normalized embedding vector
        """
        self._load_model()
        
        if not self._loaded:
            return np.random.randn(512).astype(np.float32)
        
        pil_image = self._to_pil(image)
        image_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model.encode_image(image_tensor)
            features /= features.norm(dim=-1, keepdim=True)
        
        return features[0].cpu().numpy()
    
    def _to_pil(self, image: Union[str, np.ndarray, Image.Image]) -> Image.Image:
        """Convert various image formats to PIL Image, including DICOM"""
        if isinstance(image, str):
            # Check if it's a DICOM file
            if image.lower().endswith('.dcm') or image.lower().endswith('.dicom'):
                return self._load_dicom(image)
            return Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            # Handle float32 normalized images
            if image.dtype == np.float32 and image.max() <= 1:
                image = (image * 255).astype(np.uint8)
            # Handle grayscale
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)
            return Image.fromarray(image)
        elif isinstance(image, Image.Image):
            return image.convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def _load_dicom(self, dicom_path: str) -> Image.Image:
        """
        Load DICOM file and convert to PIL Image.
        
        Args:
            dicom_path: Path to DICOM file
            
        Returns:
            PIL Image
        """
        try:
            import pydicom
            from pydicom.pixel_data_handlers.util import apply_voi_lut
            
            # Read DICOM file
            ds = pydicom.dcmread(dicom_path)
            
            # Get pixel array
            pixel_array = ds.pixel_array
            
            # Apply VOI LUT (window/level) if available
            try:
                pixel_array = apply_voi_lut(pixel_array, ds)
            except Exception:
                pass
            
            # Normalize to 0-255
            pixel_array = pixel_array.astype(float)
            pixel_array = ((pixel_array - pixel_array.min()) / 
                          (pixel_array.max() - pixel_array.min() + 1e-8) * 255)
            pixel_array = pixel_array.astype(np.uint8)
            
            # Convert to RGB
            if len(pixel_array.shape) == 2:
                pixel_array = np.stack([pixel_array] * 3, axis=-1)
            
            logger.info(f"DICOM loaded: {ds.Modality if hasattr(ds, 'Modality') else 'Unknown'}")
            
            return Image.fromarray(pixel_array)
            
        except ImportError:
            logger.warning("pydicom not installed. Install with: pip install pydicom")
            # Return a placeholder image
            return Image.new('RGB', (512, 512), color=(128, 128, 128))
        except Exception as e:
            logger.error(f"Failed to load DICOM: {e}")
            return Image.new('RGB', (512, 512), color=(128, 128, 128))
    
    def _mock_analysis(self) -> Dict:
        """Return mock results when model not available"""
        logger.warning("Using mock BiomedCLIP analysis")
        
        # More deterministic mock for demo purposes
        findings = [
            {"finding": "normal chest radiograph", "confidence": 0.45},
            {"finding": "possible mild cardiomegaly", "confidence": 0.25},
            {"finding": "no acute infiltrate", "confidence": 0.15},
            {"finding": "clear lung fields", "confidence": 0.10},
            {"finding": "normal heart size", "confidence": 0.05}
        ]
        
        return {
            "findings": findings,
            "top_finding": findings[0]["finding"],
            "confidence": findings[0]["confidence"],
            "needs_review": True,
            "review_reason": "Mock analysis - model not loaded (install BiomedCLIP for real analysis)",
            "_mock": True
        }


class MockVisionAnalyzer:
    """
    Mock vision analyzer that works without models.
    Uses rule-based analysis for demonstration.
    """
    
    def __init__(self):
        logger.info("MockVisionAnalyzer initialized (no model required)")
        self._loaded = True
    
    def analyze_chest_xray(self, image, custom_prompts=None) -> Dict:
        """Mock X-ray analysis with plausible findings"""
        return {
            "findings": [
                {"finding": "normal chest radiograph", "confidence": 0.55},
                {"finding": "no acute cardiopulmonary abnormality", "confidence": 0.30},
                {"finding": "lung fields clear", "confidence": 0.10},
                {"finding": "heart size normal", "confidence": 0.05}
            ],
            "top_finding": "normal chest radiograph",
            "confidence": 0.55,
            "needs_review": True,
            "review_reason": "Mock analysis - radiologist review recommended",
            "_mock": True
        }
    
    def get_embedding(self, image) -> np.ndarray:
        """Return mock embedding vector"""
        return np.random.randn(512).astype(np.float32)


# Singleton instance
_analyzer = None

def get_analyzer(use_mock: bool = False) -> BiomedCLIPAnalyzer:
    """
    Get vision analyzer instance.
    
    Args:
        use_mock: If True, always use mock analyzer
        
    Returns:
        Vision analyzer (real or mock)
    """
    global _analyzer
    
    if use_mock:
        return MockVisionAnalyzer()
    
    if _analyzer is None:
        _analyzer = BiomedCLIPAnalyzer()
    
    return _analyzer


if __name__ == "__main__":
    # Test the analyzer
    analyzer = get_analyzer(use_mock=True)
    result = analyzer.analyze_chest_xray(None)
    print("Mock X-ray Analysis:")
    for finding in result["findings"]:
        print(f"  {finding['finding']}: {finding['confidence']:.0%}")

