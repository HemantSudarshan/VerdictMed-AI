"""
Explainability Module
Provides SHAP and GradCAM explanations for medical AI predictions.
"""

from .shap_explainer import SHAPExplainer, get_shap_explainer

__all__ = ["SHAPExplainer", "get_shap_explainer"]
