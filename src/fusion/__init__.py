"""
Multi-Modal Fusion Module

This module handles the fusion of multi-modal signals (NLP, Vision, Knowledge Graph)
for clinical decision support.

DESIGN NOTE:
------------
For the MVP, fusion logic is intentionally implemented inline within the
SimpleDiagnosticAgent (src/reasoning/simple_agent.py) rather than as a separate module.

This design choice offers several benefits:
1. Simpler execution flow - easier to debug and maintain
2. Reduced latency - no additional function call overhead
3. Context preservation - all state is in one place
4. Easier async optimization - all logic in one async workflow

The fusion approach combines:
- NLP symptom extraction (medical entities, negation handling)
- Vision analysis (BiomedCLIP chest X-ray interpretation)
- Knowledge graph queries (symptom-to-disease mapping)
- Confidence scoring across modalities

Future Enhancement:
-------------------
If fusion logic becomes more complex (e.g., learned fusion weights,
attention mechanisms, or separate fusion models), it can be extracted
into this module without changing the agent interface.

See SimpleDiagnosticAgent._calculate_confidence() for current fusion implementation.
"""

__version__ = "1.0.0"
__all__ = []

# Re-export fusion-related functionality if needed in the future
# from .multi_modal_fusion import FusionEngine
