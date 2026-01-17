# Multi-Modal Fusion Strategy

## Overview

The VerdictMed AI CDSS uses **multi-modal fusion** to combine evidence from three sources:
1. **NLP Analysis** - Symptom extraction from clinical notes
2. **Vision Analysis** - Chest X-ray interpretation via BiomedCLIP
3. **Knowledge Graph** - Symptom-to-disease reasoning via Neo4j

---

## Current Implementation

**Location**: [`src/reasoning/simple_agent.py`](../reasoning/simple_agent.py)

The fusion logic is intentionally implemented **inline** within the `SimpleDiagnosticAgent` for MVP simplicity.

### Why Inline Fusion?

✅ **Simpler execution flow** - All logic in one place, easier to debug  
✅ **Reduced latency** - No additional module loading overhead  
✅ **Async optimization** - Single async workflow without cross-module coordination  
✅ **Context preservation** - All diagnostic state accessible in one scope  

---

## Fusion Workflow

```python
# Parallel execution of modalities
async def run(self, patient_data):
    # 1. Parallel NLP and Image Analysis
    nlp_result = await run_nlp(symptoms_text)
    image_result = await run_image_analysis(image_path)
    
    # 2. Knowledge Graph Query (uses NLP results)
    kg_diseases = await query_kg(extracted_symptoms)
    
    # 3. Fusion: Calculate confidence for each disease
    for disease in kg_diseases:
        confidence = _calculate_confidence(
            disease=disease,
            image_findings=image_result,
            symptoms=nlp_result
        )
    
    # 4. Safety validation and ranking
    differential = rank_by_confidence(diseases)
    return validate_and_explain(differential)
```

---

## Confidence Calculation

**Method**: `SimpleDiagnosticAgent._calculate_confidence()`

Combines signals from all modalities:

```python
def _calculate_confidence(disease, image_findings, state):
    # Base score from KG symptom matching
    base_score = disease.get("match_ratio", 0.5)
    
    # Boost if image supports the diagnosis
    if image_findings.get("top_finding"):
        if disease_matches_image(disease, image_findings):
            base_score *= 1.2  # 20% boost for image confirmation
    
    # Adjust for data completeness
    completeness = calculate_completeness(state)
    # completeness = 0.4 (text only) to 1.0 (text + image + labs)
    
    final_confidence = base_score * completeness
    return min(final_confidence, 0.99)  # Cap at 99%
```

### Fusion Weights (Implicit)

| Modality | Weight | Rationale |
|----------|--------|-----------|
| KG Symptom Match | Base (50-90%) | Core diagnostic signal |
| Image Confirmation | +20% multiplier | Strong supporting evidence |
| Data Completeness | 40-100% multiplier | Uncertainty from missing data |

---

## Example Fusion

**Case**: Patient with fever, cough, dyspnea + chest X-ray showing consolidation

```
1. NLP extracts: ["fever", "cough", "dyspnea"]
2. KG matches: Pneumonia (match_ratio: 0.75)
3. Image finds: "Consolidation in right lower lobe" (confidence: 0.82)
4. Fusion:
   - Base: 0.75 (KG match)
   - Image boost: 0.75 * 1.2 = 0.90 (pneumonia aligns with consolidation)
   - Completeness: 0.90 * 0.9 = 0.81 (text + image, no labs)
   
Final: Pneumonia (81% confidence)
```

---

## Future Enhancements

When fusion complexity increases, extract into separate module:

### Learned Fusion Model
```python
# fusion/fusion_model.py
class LearnedFusionModel:
    """
    Learn optimal fusion weights from doctor feedback.
    Uses logistic regression or small neural network.
    """
    def fuse(self, nlp_score, vision_score, kg_score, metadata):
        # Learned weights instead of hard-coded multipliers
        return weighted_sum(scores, learned_weights)
```

### Attention-Based Fusion
```python
# For complex cases, use attention mechanism
def attention_fusion(modalities):
    # Determine which modality is most reliable for this case
    attention_weights = softmax(modality_confidences)
    return weighted_average(modalities, attention_weights)
```

---

## Monitoring Fusion Effectiveness

Track these metrics to guide fusion improvements:

| Metric | Target | Purpose |
|--------|--------|---------|
| Image-boosted accuracy | > NLP-only | Validate image adds value |
| Completeness correlation | High R² with accuracy | Justify completeness weighting |
| Modality agreement rate | > 80% | Detect conflicting signals |

Access in Grafana: `cdss-ops` dashboard → "Fusion Metrics" panel

---

## References

- Current implementation: [`simple_agent.py:236-255`](../reasoning/simple_agent.py#L236-L255)
- PRD specification: `CDSS_PRD_Part1.md` (Stage 3: Fusion Module)
- Related: [Safety Validator](../safety/validator.py) (checks signal conflicts)
