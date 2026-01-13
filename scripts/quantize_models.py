"""
Model Quantization Scripts
Reduce model size and improve inference speed through quantization.
"""

import torch
from pathlib import Path
from loguru import logger
from typing import Optional


def quantize_model_dynamic(
    model: torch.nn.Module,
    output_path: Optional[str] = None
) -> torch.nn.Module:
    """
    Apply dynamic quantization to a PyTorch model.
    Best for CPU inference on models with Linear layers.
    
    Args:
        model: PyTorch model to quantize
        output_path: Optional path to save quantized model
        
    Returns:
        Quantized model
    """
    logger.info("Applying dynamic quantization...")
    
    # Dynamic quantization - quantizes weights at load time
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU},
        dtype=torch.qint8
    )
    
    if output_path:
        torch.save(quantized_model.state_dict(), output_path)
        logger.info(f"Saved quantized model to {output_path}")
    
    # Log size reduction
    original_size = sum(p.numel() * p.element_size() for p in model.parameters())
    quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters())
    reduction = (1 - quantized_size / original_size) * 100
    
    logger.info(f"Size reduction: {reduction:.1f}%")
    logger.info(f"Original: {original_size / 1e6:.2f} MB -> Quantized: {quantized_size / 1e6:.2f} MB")
    
    return quantized_model


def quantize_to_onnx(
    model: torch.nn.Module,
    input_shape: tuple,
    output_path: str,
    opset_version: int = 14
):
    """
    Export model to ONNX format for optimized inference.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (e.g., (1, 3, 224, 224))
        output_path: Path to save ONNX model
        opset_version: ONNX opset version
    """
    logger.info("Exporting model to ONNX...")
    
    model.eval()
    dummy_input = torch.randn(*input_shape)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    logger.info(f"Exported ONNX model to {output_path}")
    
    # Verify
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model validation passed")
    except ImportError:
        logger.warning("onnx package not installed, skipping validation")


def quantize_biobert(output_dir: str = "data/models"):
    """
    Quantize BioBERT model for faster NLP inference.
    """
    try:
        from transformers import AutoModel, AutoTokenizer
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Loading BioBERT model...")
        model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
        tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
        
        # Dynamic quantization
        quantized = quantize_model_dynamic(
            model,
            str(output_dir / "biobert_quantized.pt")
        )
        
        # Save tokenizer
        tokenizer.save_pretrained(str(output_dir / "biobert_tokenizer"))
        
        logger.info("BioBERT quantization complete!")
        return quantized
        
    except ImportError:
        logger.error("transformers package required for BioBERT quantization")
        return None


def quantize_vision_model(output_dir: str = "data/models"):
    """
    Quantize vision model (BiomedCLIP backbone) for faster inference.
    """
    try:
        import open_clip
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Loading BiomedCLIP model...")
        model, _, _ = open_clip.create_model_and_transforms(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        
        # Export to ONNX
        quantize_to_onnx(
            model.visual,
            (1, 3, 224, 224),
            str(output_dir / "biomedclip_visual.onnx")
        )
        
        logger.info("Vision model quantization complete!")
        return True
        
    except ImportError:
        logger.error("open_clip package required for vision quantization")
        return False


def benchmark_inference(
    original_model: torch.nn.Module,
    quantized_model: torch.nn.Module,
    input_shape: tuple,
    num_runs: int = 100
) -> dict:
    """
    Benchmark original vs quantized model performance.
    
    Args:
        original_model: Original PyTorch model
        quantized_model: Quantized model
        input_shape: Input tensor shape
        num_runs: Number of inference runs
        
    Returns:
        Dict with timing results
    """
    import time
    
    dummy_input = torch.randn(*input_shape)
    
    # Warm up
    for _ in range(10):
        original_model(dummy_input)
        quantized_model(dummy_input)
    
    # Benchmark original
    start = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            original_model(dummy_input)
    original_time = (time.time() - start) / num_runs
    
    # Benchmark quantized
    start = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            quantized_model(dummy_input)
    quantized_time = (time.time() - start) / num_runs
    
    speedup = original_time / quantized_time
    
    results = {
        "original_ms": original_time * 1000,
        "quantized_ms": quantized_time * 1000,
        "speedup": speedup,
        "num_runs": num_runs
    }
    
    logger.info(f"Original: {results['original_ms']:.2f}ms")
    logger.info(f"Quantized: {results['quantized_ms']:.2f}ms")
    logger.info(f"Speedup: {speedup:.2f}x")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quantize CDSS models")
    parser.add_argument("--model", choices=["biobert", "vision", "all"], default="all")
    parser.add_argument("--output-dir", default="data/models")
    args = parser.parse_args()
    
    if args.model in ["biobert", "all"]:
        quantize_biobert(args.output_dir)
    
    if args.model in ["vision", "all"]:
        quantize_vision_model(args.output_dir)
    
    logger.info("Quantization complete!")
