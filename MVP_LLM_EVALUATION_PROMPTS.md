# 🔍 MVP EVALUATION - LLM PROMPTS & TEMPLATES

**Quick Reference for Evaluating Your VerdictMed AI MVP**

Use these prompts to feed to Claude, Gemini, ChatGPT, or any LLM for detailed evaluation.

---

## 📌 PROMPT 1: COMPREHENSIVE MVP EVALUATION

```
You are an expert AI/ML architect evaluating a Clinical Decision Support System (CDSS).

I'm building "VerdictMed AI" - a multimodal CDSS combining:
1. Medical image analysis (MedSAM) 
2. Clinical text NLP (BioBERT)
3. Medical knowledge graph (Neo4j + UMLS)
4. AI reasoning engine (LangGraph + LLM)
5. Explainability (SHAP + reasoning chains)

Please evaluate my MVP against this framework: [PASTE THE FRAMEWORK FROM MVP_EVALUATION_FRAMEWORK.md]

My MVP code/architecture:
[PASTE YOUR CODE/ARCHITECTURE HERE]

Provide detailed analysis:
1. For each of the 10 sections, assess current implementation status
2. Identify gaps and missing features
3. Rate implementation quality (1-10)
4. Provide specific recommendations
5. Calculate overall MVP score (0-100)
6. Give final verdict: Ready for production or needs revision?

Focus on:
- Completeness of implementation
- Code quality and maintainability
- Alignment with expected architecture
- Safety and compliance
- Performance and accuracy
```

---

## 📌 PROMPT 2: ARCHITECTURE REVIEW

```
Review the architecture of my VerdictMed AI MVP.

Expected architecture:
- Layer 1: Vision (MedSAM) - medical image segmentation
- Layer 2: NLP (BioBERT) - entity extraction from text
- Layer 3: Knowledge Graph (Neo4j) - medical relationships
- Layer 4: Reasoning (LangGraph + LLM) - diagnostic reasoning
- Layer 5: Explainability (SHAP + chains) - explain outputs

My actual implementation:
[PASTE YOUR ARCHITECTURE CODE/DESCRIPTION]

Analyze:
1. Is each layer properly isolated and testable?
2. Are interfaces between layers clear?
3. Is data flow correct?
4. Are there any architectural issues or anti-patterns?
5. How modular and scalable is it?
6. Can it be deployed in containers?

Provide specific suggestions for improvement.
```

---

## 📌 PROMPT 3: TECH STACK VALIDATION

```
Validate my technology choices for VerdictMed AI MVP.

My tech stack:
- Vision: MedSAM
- NLP: BioBERT
- KG: Neo4j + UMLS
- Reasoning: LangGraph + Google Gemini API
- Explainability: SHAP + reasoning chains
- Infrastructure: Docker + PostgreSQL

My implementation details:
[PASTE YOUR IMPLEMENTATION]

Evaluate:
1. Is each technology choice appropriate for the use case?
2. Are there better alternatives?
3. Are technologies well-integrated?
4. Are there any compatibility issues?
5. What's the learning curve for maintenance?
6. Is stack production-ready?

Rate each component 1-10 and overall tech stack 1-10.
```

---

## 📌 PROMPT 4: ACCURACY & PERFORMANCE ASSESSMENT

```
Assess the accuracy and performance of my VerdictMed AI MVP.

Expected benchmarks:
- Vision: 85-92% Dice coefficient on pneumonia segmentation
- NLP: 92-95% F1 score on entity extraction
- Full system diagnosis: 80-88% top-1 accuracy
- Latency: <10 seconds total per diagnosis

My measurements:
[PASTE YOUR ACTUAL METRICS]

Analysis needed:
1. What are my actual accuracy/performance numbers?
2. How do they compare to expectations?
3. Where are bottlenecks?
4. Which module needs optimization?
5. What can I do to improve accuracy?
6. What can I do to improve latency?
7. Have I tested on real patient cases?

Provide prioritized list of optimizations with estimated effort.
```

---

## 📌 PROMPT 5: SAFETY & COMPLIANCE CHECK

```
Evaluate safety and regulatory compliance of my VerdictMed AI MVP.

My safety implementation:
[PASTE YOUR SAFETY CODE/DESCRIPTION]

Check for:
1. Confidence thresholds - do they exist and are they reasonable?
2. Conflict detection - can the system detect contradictory signals?
3. Human-in-loop - does it require doctor approval?
4. Audit logging - are all decisions logged?
5. Data privacy - is patient data encrypted and protected?
6. Error handling - does it fail gracefully?
7. Fallback mechanisms - what happens if a component fails?
8. HIPAA compliance - any violations or gaps?

For each item:
- Rate implementation completeness (1-10)
- Identify gaps
- Recommend specific improvements
- Flag any serious issues

Provide overall safety/compliance score.
```

---

## 📌 PROMPT 6: CODE QUALITY REVIEW

```
Perform a code quality review of my VerdictMed AI MVP.

Code structure:
[PASTE YOUR CODE OR DESCRIBE STRUCTURE]

Evaluate:
1. Code organization - are modules properly separated?
2. Naming conventions - are they consistent and clear?
3. Documentation - are functions/classes documented?
4. Error handling - comprehensive try-catch?
5. Testing - what's the test coverage?
6. Dependencies - are they pinned and minimal?
7. Style compliance - follows PEP8 or equivalent?
8. Type hints - are they used throughout?

For each item:
- Rate quality (1-10)
- Identify issues
- Provide specific examples of problems
- Recommend improvements

Suggest refactoring priorities with effort estimates.
```

---

## 📌 PROMPT 7: TESTING COMPLETENESS

```
Evaluate the testing coverage of my VerdictMed AI MVP.

Current test suite:
[PASTE YOUR TEST FILES/DESCRIBE TESTS]

Assess:
1. Unit tests - what's covered, what's missing?
2. Integration tests - are there end-to-end tests?
3. Test coverage - percentage of code covered?
4. Test quality - are tests meaningful or just for coverage?
5. Regression tests - baseline test cases?
6. Edge cases - are unusual scenarios tested?
7. Error cases - failure modes tested?
8. Performance tests - latency/throughput tested?

For each category:
- Rate completeness (1-10)
- Identify gaps
- Recommend specific test cases to add

Provide overall testing score and priority list of tests to write.
```

---

## 📌 PROMPT 8: DEPLOYMENT READINESS

```
Evaluate deployment readiness of my VerdictMed AI MVP.

Current deployment setup:
[PASTE YOUR DOCKERFILE/DOCKER-COMPOSE/DEPLOYMENT CONFIGS]

Check:
1. Containerization - does it have Dockerfile?
2. Docker Compose - all services configured?
3. API design - REST endpoints well-designed?
4. API documentation - Swagger/OpenAPI?
5. Health checks - can system status be monitored?
6. Logging - is logging comprehensive?
7. Metrics - are performance metrics tracked?
8. Configuration - are configs externalized?
9. Secrets management - API keys stored securely?
10. Scaling - can it scale horizontally?

For each item:
- Rate implementation (1-10)
- Identify issues
- Recommend improvements

Provide deployment readiness score and prioritized fixes.
```

---

## 📌 PROMPT 9: GAPS ANALYSIS

```
Identify all gaps between my current MVP and production-ready system.

Current MVP state:
[DESCRIBE YOUR MVP COMPLETION STATUS]

For each expected component:
1. Vision Module (MedSAM)
   - [ ] Implemented? Yes/Partial/No
   - [ ] Tested? Yes/No
   - [ ] Accurate? [Metric: ____%]
   - [ ] Gaps: [List]

2. NLP Module (BioBERT)
   - [ ] Implemented? Yes/Partial/No
   - [ ] Tested? Yes/No
   - [ ] Accurate? [Metric: ____%]
   - [ ] Gaps: [List]

3. Knowledge Graph
   - [ ] Implemented? Yes/Partial/No
   - [ ] Populated with UMLS? Yes/Partial/No
   - [ ] Query working? Yes/No
   - [ ] Gaps: [List]

4. Reasoning Engine
   - [ ] Implemented? Yes/Partial/No
   - [ ] LangGraph used? Yes/No
   - [ ] LLM integrated? Yes/No
   - [ ] Gaps: [List]

5. Explainability
   - [ ] SHAP implemented? Yes/No
   - [ ] Reasoning chains logged? Yes/No
   - [ ] Doctor-friendly output? Yes/No
   - [ ] Gaps: [List]

6. Safety & Compliance
   - [ ] Confidence thresholds? Yes/No
   - [ ] Audit logging? Yes/No
   - [ ] Error handling? Yes/No
   - [ ] Gaps: [List]

7. Testing
   - [ ] Unit tests? Yes/Partial/No
   - [ ] Integration tests? Yes/No
   - [ ] Coverage: _____%
   - [ ] Gaps: [List]

8. Deployment
   - [ ] Docker? Yes/No
   - [ ] API documented? Yes/No
   - [ ] Health checks? Yes/No
   - [ ] Gaps: [List]

For each gap identified:
- Describe what's missing
- Estimate effort to implement (hours)
- Assign priority (high/medium/low)
- Suggest implementation approach

Provide summary: What % of MVP is complete? What's critical vs nice-to-have?
```

---

## 📌 PROMPT 10: IMPROVEMENTS PRIORITIZATION

```
Help me prioritize improvements to my VerdictMed AI MVP.

Current state:
- Completion: ___%
- Test coverage: ___%
- Estimated time to production: ___ weeks

Gaps identified: [PASTE LIST OF GAPS FROM PROMPT 9]

Based on all gaps, create a prioritized improvement plan:

Priority 1 (CRITICAL - must fix before any deployment):
- Item 1: [Gap description]
  Effort: ___ hours
  Impact: Blocks deployment / High impact
  Recommendation: [How to fix]

- Item 2: [Gap description]
  Effort: ___ hours
  Impact: Blocks deployment / High impact
  Recommendation: [How to fix]

Priority 2 (HIGH - should fix before production):
- Item 1: [Gap description]
  Effort: ___ hours
  Impact: Affects reliability/accuracy
  Recommendation: [How to fix]

Priority 3 (MEDIUM - nice to have):
- Item 1: [Gap description]
  Effort: ___ hours
  Impact: Improves user experience
  Recommendation: [How to fix]

For the entire improvement plan:
1. Total effort needed (hours)
2. Estimated timeline (weeks at 20 hrs/week)
3. Which improvements can be parallelized
4. Critical path items
5. Contingency planning

What's the realistic timeline to production?
```

---

## 📌 PROMPT 11: FINAL MVP VERDICT

```
Provide a final verdict on my VerdictMed AI MVP.

Current state:
- Architecture: [Quality 1-10]
- Tech stack: [Quality 1-10]
- Code quality: [Quality 1-10]
- Testing: [Quality 1-10]
- Safety: [Quality 1-10]
- Deployment: [Quality 1-10]
- Documentation: [Quality 1-10]

Overall completion: ___%
Overall quality score: __/100

Based on comprehensive evaluation, answer:

1. IS IT READY FOR NEXT PHASE? (Yes/No/With conditions)
   - If no, what must be fixed?
   - If with conditions, what conditions?

2. WHAT'S THE MAIN STRENGTH?
   - Single most impressive aspect

3. WHAT'S THE MAIN WEAKNESS?
   - Single most critical issue

4. CAN IT HANDLE REAL PATIENTS? (Yes/No)
   - What safety issues remain?
   - What would need to happen for Yes?

5. WHAT'S MISSING FOR PRODUCTION?
   - List of must-have items
   - Estimated effort for each

6. CONFIDENCE LEVEL
   - How confident am I that this MVP can become production system?
   - 0-100%: ___%

7. RECOMMENDATION
   Choose one:
   - ✅ APPROVED - Ready to scale/enhance
   - ✅ APPROVED WITH CONDITIONS - Fix X, Y, Z then deploy
   - ⚠️ NEEDS REVISION - Significant work needed
   - ❌ NOT APPROVED - Too many critical gaps

8. NEXT STEPS
   - What should I do immediately?
   - What should I do in next 2 weeks?
   - What can wait?

9. ESTIMATED TIMELINE TO DEPLOYMENT
   - Current state: MVP
   - Next milestone: Alpha ready (weeks: ___)
   - Production ready (weeks: ___)

10. CAREER IMPACT
    - Will this MVP be impressive for ₹25-35 LPA jobs? Yes/No
    - What needs to be highlighted?
    - What needs improvement?
```

---

## 📌 QUICK EVALUATION CHECKLIST

Use this simple checklist for quick assessment:

```
ARCHITECTURE
- [ ] 5 layers properly separated
- [ ] Clear interfaces between layers
- [ ] Modular and testable
- [ ] Data flow correct
- [ ] Scalable design

TECH STACK
- [ ] MedSAM integrated
- [ ] BioBERT integrated
- [ ] Neo4j + UMLS configured
- [ ] LangGraph + LLM working
- [ ] SHAP for explanations

INPUT/OUTPUT
- [ ] Accepts images (DICOM/PNG/JPG)
- [ ] Accepts clinical text
- [ ] Accepts lab values
- [ ] Outputs diagnosis + confidence
- [ ] Outputs differentials
- [ ] Outputs recommendations
- [ ] Outputs explanations

ACCURACY & PERFORMANCE
- [ ] Vision accuracy measured
- [ ] NLP accuracy measured
- [ ] End-to-end accuracy measured
- [ ] Latency measured (< 10 sec target)
- [ ] Meets or exceeds expectations

SAFETY & COMPLIANCE
- [ ] Confidence thresholds
- [ ] Conflict detection
- [ ] Human-in-loop
- [ ] Audit logging
- [ ] Data encryption
- [ ] Error handling

TESTING
- [ ] Unit tests (>80% coverage)
- [ ] Integration tests
- [ ] End-to-end tests
- [ ] Accuracy verified
- [ ] Performance verified

DEPLOYMENT
- [ ] Dockerfile exists and works
- [ ] Docker Compose configured
- [ ] API endpoints documented
- [ ] Health checks implemented
- [ ] Logging implemented

CODE QUALITY
- [ ] Follows style guide
- [ ] Well documented
- [ ] Type hints used
- [ ] Dependencies pinned
- [ ] No major warnings

DOCUMENTATION
- [ ] README comprehensive
- [ ] Architecture documented
- [ ] API documented
- [ ] Deployment guide written

TOTAL CHECKS: ___/50
PERCENTAGE: ___%

Score:
- 50/50 (100%) = Production-ready ✅
- 45-49/50 = Ready with minor fixes ✅
- 40-44/50 = Needs revision before deployment ⚠️
- <40/50 = Major work needed ❌
```

---

## 💡 TIPS FOR BEST LLM EVALUATION

1. **Be specific** - Include actual code snippets, not just descriptions
2. **Provide context** - Explain your design decisions and constraints
3. **Share metrics** - Include actual accuracy/latency numbers
4. **Ask follow-up questions** - Don't settle for surface-level answers
5. **Request examples** - Ask for specific code improvements
6. **Get prioritization** - Ask what to fix first vs what can wait
7. **Clarify scope** - Define "production-ready" for your context
8. **Check assumptions** - Verify LLM understood your architecture
9. **Iterate** - Ask for deeper analysis on critical areas
10. **Document feedback** - Save LLM's recommendations for reference

---

## 🎯 SAMPLE OUTPUT TEMPLATE

When using these prompts, the LLM should produce output like:

```
# VerdictMed AI MVP Evaluation Report

## Executive Summary
- Overall Score: ___/100
- Status: [Production-ready / Needs revision / Not approved]
- Key Finding: [1-2 sentence summary]

## Section-by-Section Analysis

### 1. Architecture (Score: __/10)
Strengths:
- [List 2-3 key strengths]

Gaps:
- [List 2-3 main issues]

Recommendations:
- [Specific improvement 1]
- [Specific improvement 2]

[Repeat for all sections]

## Strengths Summary
1. [Top strength 1]
2. [Top strength 2]
3. [Top strength 3]

## Critical Gaps
1. [Must fix 1] - Effort: __ hours
2. [Must fix 2] - Effort: __ hours
3. [Must fix 3] - Effort: __ hours

## Improvement Priority Plan
[Detailed timeline with tasks, effort, and priority]

## Final Verdict
[Recommendation: Approved / Approved with conditions / Needs revision / Not approved]
[Rationale: 3-4 sentence explanation]

## Next Steps
1. [Immediate action]
2. [Near-term (1 week)]
3. [Medium-term (2-4 weeks)]
4. [Production deployment (4+ weeks)]
```

---

## 📝 HOW TO USE THIS DOCUMENT

1. **Choose a prompt** from above based on what you want to evaluate
2. **Fill in your details** where indicated with [PASTE YOUR X]
3. **Feed entire prompt to LLM** (Claude, Gemini, ChatGPT)
4. **Get detailed evaluation** with specific recommendations
5. **Document feedback** for future reference
6. **Create action items** from recommendations
7. **Re-evaluate** after implementing improvements

**Use multiple prompts for comprehensive evaluation** - don't just use one!

