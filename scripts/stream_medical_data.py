"""
Medical Data Streamer
Stream medical datasets from HuggingFace without downloading.
Uses < 5 GB storage via API-based approach.
"""

from datasets import load_dataset
from typing import Iterator, Dict, List, Optional
import numpy as np
from loguru import logger


class MedicalDataStreamer:
    """Stream medical datasets from HuggingFace Hub"""
    
    def __init__(self):
        self.xray_dataset = None
        self.text_dataset = None
        logger.info("MedicalDataStreamer initialized")
    
    def stream_xrays(self, num_samples: int = 1000) -> Iterator[Dict]:
        """
        Stream chest X-rays without downloading full dataset.
        
        Uses the alkzar90/NIH-Chest-X-ray-dataset on HuggingFace.
        Streaming mode uses approximately 0 local storage.
        
        Args:
            num_samples: Maximum number of samples to yield
            
        Yields:
            Dict with image array, labels, and index
        """
        logger.info(f"Starting X-ray stream (max {num_samples} samples)")
        
        dataset = load_dataset(
            "alkzar90/NIH-Chest-X-ray-dataset",
            split="train",
            streaming=True  # KEY: Stream mode uses ~0 storage
        )
        
        for i, sample in enumerate(dataset):
            if i >= num_samples:
                break
            
            yield {
                "image": np.array(sample["image"]),
                "labels": sample["labels"],
                "index": i
            }
        
        logger.info(f"Streamed {min(i+1, num_samples)} X-ray samples")
    
    def stream_clinical_notes(self, num_samples: int = 1000) -> Iterator[Dict]:
        """
        Stream clinical text data for NLP training.
        
        Args:
            num_samples: Maximum samples to yield
            
        Yields:
            Dict with clinical note text
        """
        logger.info(f"Starting clinical notes stream (max {num_samples} samples)")
        
        try:
            dataset = load_dataset(
                "medical_dialog",
                split="train",
                streaming=True
            )
            
            for i, sample in enumerate(dataset):
                if i >= num_samples:
                    break
                yield sample
                
        except Exception as e:
            logger.warning(f"medical_dialog dataset unavailable: {e}")
            logger.info("Using fallback synthetic data")
            
            # Fallback: Yield synthetic samples
            for i in range(min(num_samples, 100)):
                yield {
                    "text": f"Sample clinical note {i}",
                    "index": i
                }
    
    def get_sample_batch(
        self, 
        batch_size: int = 32,
        include_labels: bool = True
    ) -> tuple:
        """
        Get a batch of samples for training/testing.
        
        Args:
            batch_size: Number of samples in batch
            include_labels: Whether to include label array
            
        Returns:
            Tuple of (images_array, labels_list) if include_labels
            Otherwise just images_array
        """
        samples = list(self.stream_xrays(batch_size))
        images = np.stack([s["image"] for s in samples])
        
        if include_labels:
            labels = [s["labels"] for s in samples]
            return images, labels
        
        return images
    
    def get_disease_distribution(self, num_samples: int = 5000) -> Dict[str, int]:
        """
        Calculate disease label distribution from streamed data.
        
        Args:
            num_samples: Number of samples to analyze
            
        Returns:
            Dict mapping disease names to counts
        """
        distribution = {}
        
        for sample in self.stream_xrays(num_samples):
            for label in sample["labels"]:
                distribution[label] = distribution.get(label, 0) + 1
        
        return dict(sorted(distribution.items(), key=lambda x: x[1], reverse=True))


# Convenience functions
def stream_xray_samples(num_samples: int = 100) -> Iterator[Dict]:
    """Quick access to X-ray streaming"""
    streamer = MedicalDataStreamer()
    return streamer.stream_xrays(num_samples)


def get_training_batch(batch_size: int = 32):
    """Get a batch for training"""
    streamer = MedicalDataStreamer()
    return streamer.get_sample_batch(batch_size)


if __name__ == "__main__":
    # Test streaming
    streamer = MedicalDataStreamer()
    
    print("Testing X-ray streaming...")
    for i, sample in enumerate(streamer.stream_xrays(5)):
        print(f"Sample {i}: shape={sample['image'].shape}, labels={sample['labels']}")
    
    print("\nCalculating disease distribution (100 samples)...")
    dist = streamer.get_disease_distribution(100)
    for disease, count in list(dist.items())[:10]:
        print(f"  {disease}: {count}")
