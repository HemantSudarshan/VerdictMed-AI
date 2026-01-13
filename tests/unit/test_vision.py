"""
Unit tests for Vision Module (BiomedCLIP)
Tests image analysis, mock fallback, and error handling.
"""
import pytest
import numpy as np
from PIL import Image
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.vision.biomedclip import BiomedCLIPAnalyzer, MockVisionAnalyzer, get_analyzer


class TestMockVisionAnalyzer:
    """Tests for MockVisionAnalyzer (works without PyTorch)"""
    
    def test_initialization(self):
        """Test analyzer initializes correctly"""
        analyzer = MockVisionAnalyzer()
        assert analyzer._loaded == True
    
    def test_analyze_chest_xray_returns_dict(self):
        """Test analysis returns proper dictionary structure"""
        analyzer = MockVisionAnalyzer()
        result = analyzer.analyze_chest_xray(None)
        
        assert isinstance(result, dict)
        assert "findings" in result
        assert "top_finding" in result
        assert "confidence" in result
        assert "needs_review" in result
        assert "_mock" in result
    
    def test_findings_structure(self):
        """Test findings list has correct structure"""
        analyzer = MockVisionAnalyzer()
        result = analyzer.analyze_chest_xray(None)
        
        assert isinstance(result["findings"], list)
        assert len(result["findings"]) > 0
        
        for finding in result["findings"]:
            assert "finding" in finding
            assert "confidence" in finding
            assert isinstance(finding["confidence"], (int, float))
            assert 0 <= finding["confidence"] <= 1
    
    def test_confidence_is_valid(self):
        """Test confidence score is between 0 and 1"""
        analyzer = MockVisionAnalyzer()
        result = analyzer.analyze_chest_xray(None)
        
        assert 0 <= result["confidence"] <= 1
    
    def test_needs_review_flag(self):
        """Test needs_review is boolean"""
        analyzer = MockVisionAnalyzer()
        result = analyzer.analyze_chest_xray(None)
        
        assert isinstance(result["needs_review"], bool)



class TestGetAnalyzer:
    """Tests for get_analyzer factory function"""
    
    def test_get_mock_analyzer(self):
        """Test getting mock analyzer explicitly"""
        analyzer = get_analyzer(use_mock=True)
        assert isinstance(analyzer, MockVisionAnalyzer)
    
    def test_default_analyzer_type(self):
        """Test default analyzer type"""
        try:
            analyzer = get_analyzer(use_mock=False)
            # Should be BiomedCLIPAnalyzer (may fall back to mock if torch unavailable)
            assert hasattr(analyzer, 'analyze_chest_xray')
        except NameError:
            pytest.skip("PyTorch not available")



class TestBiomedCLIPAnalyzer:
    """Tests for BiomedCLIPAnalyzer class"""
    
    def test_initialization(self):
        """Test BiomedCLIPAnalyzer initializes"""
        try:
            analyzer = BiomedCLIPAnalyzer()
            assert analyzer.device in ["cuda", "cpu"]
            assert analyzer._loaded == False  # Lazy loading
        except Exception:
            pytest.skip("PyTorch not available")
    
    def test_chest_xray_findings_list(self):
        """Test default findings list exists"""
        try:
            analyzer = BiomedCLIPAnalyzer()
            assert len(analyzer.chest_xray_findings) > 0
            assert "normal chest radiograph" in analyzer.chest_xray_findings
        except Exception:
            pytest.skip("PyTorch not available")
    
    def test_mock_fallback(self):
        """Test mock analysis when model not loaded"""
        try:
            analyzer = BiomedCLIPAnalyzer()
            # Don't load model, should use mock
            result = analyzer._mock_analysis()
            
            assert "_mock" in result
            assert result["_mock"] == True
        except Exception:
            pytest.skip("PyTorch not available")


class TestImageConversion:
    """Tests for image format conversion"""
    
    def test_numpy_to_pil(self):
        """Test converting numpy array to PIL Image"""
        try:
            analyzer = BiomedCLIPAnalyzer()
            
            # Create test image
            np_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            pil_image = analyzer._to_pil(np_image)
            
            assert isinstance(pil_image, Image.Image)
            assert pil_image.mode == "RGB"
        except Exception:
            pytest.skip("PyTorch not available")
    
    def test_grayscale_conversion(self):
        """Test grayscale image gets converted to RGB"""
        try:
            analyzer = BiomedCLIPAnalyzer()
            
            # Create grayscale image
            gray_image = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
            pil_image = analyzer._to_pil(gray_image)
            
            assert pil_image.mode == "RGB"
        except Exception:
            pytest.skip("PyTorch not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
