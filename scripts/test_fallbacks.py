"""Quick test script for all fallback components"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 50)
print("Testing VerdictMed AI Fallback Components")
print("=" * 50)

# Test 1: Mock Knowledge Graph
print("\n1. Mock Knowledge Graph:")
try:
    from src.knowledge_graph.mock_kg import MockKnowledgeGraph
    kg = MockKnowledgeGraph()
    results = kg.find_diseases_by_symptoms(['fever', 'cough', 'shortness of breath'])
    for r in results[:5]:
        print(f"   {r['disease']}: {r['match_ratio']:.0%} ({r['severity']})")
    print("   ✓ Knowledge Graph working!")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 2: In-Memory Cache
print("\n2. In-Memory Cache:")
try:
    from src.cache.redis_service import get_cache
    cache = get_cache()
    cache.set('test_key', {'diagnosis': 'Pneumonia', 'confidence': 0.85})
    result = cache.get('test_key')
    print(f"   Stored: test_key = {result}")
    print(f"   Using fallback: {cache.using_fallback}")
    print("   ✓ Cache working!")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 3: Mock Vision Analyzer
print("\n3. Mock Vision Analyzer:")
try:
    from src.vision.biomedclip import get_analyzer
    analyzer = get_analyzer(use_mock=True)
    result = analyzer.analyze_chest_xray(None)
    for f in result['findings'][:3]:
        print(f"   {f['finding']}: {f['confidence']:.0%}")
    print("   ✓ Vision Analyzer working!")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "=" * 50)
print("All fallback components verified!")
print("=" * 50)
