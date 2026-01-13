"""
X-Ray Data Processor
Process raw X-ray images and metadata into training-ready format.
"""

import cv2
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger
from tqdm import tqdm
from dataclasses import dataclass


@dataclass
class ProcessingConfig:
    """Configuration for image processing"""
    target_size: tuple = (224, 224)
    apply_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: tuple = (8, 8)
    normalize: bool = True
    validate_quality: bool = True
    min_sharpness: float = 50.0


class XRayDataProcessor:
    """Process raw X-ray images for the CDSS"""
    
    def __init__(
        self, 
        raw_dir: str, 
        processed_dir: str,
        config: ProcessingConfig = None
    ):
        """
        Initialize processor.
        
        Args:
            raw_dir: Path to raw data
            processed_dir: Path to save processed data
            config: Processing configuration
        """
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.config = config or ProcessingConfig()
        
        # Create output directories
        (self.processed_dir / "images").mkdir(parents=True, exist_ok=True)
        
        # Disease label mapping
        self.disease_labels = [
            "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
            "Mass", "Nodule", "Pneumonia", "Pneumothorax",
            "Consolidation", "Edema", "Emphysema", "Fibrosis",
            "Pleural_Thickening", "Hernia"
        ]
    
    def load_metadata(self) -> pd.DataFrame:
        """Load and validate NIH metadata"""
        metadata_path = self.raw_dir / "Data_Entry_2017_v2020.csv"
        
        if not metadata_path.exists():
            # Try alternative name
            metadata_path = self.raw_dir / "Data_Entry_2017.csv"
        
        if not metadata_path.exists():
            logger.error(f"Metadata file not found in {self.raw_dir}")
            raise FileNotFoundError(f"Metadata not found")
        
        df = pd.read_csv(metadata_path)
        logger.info(f"Loaded {len(df)} records from metadata")
        
        # Validate required columns
        required = ["Image Index", "Finding Labels", "Patient Age", "Patient Gender"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        return df
    
    def preprocess_image(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Preprocess single X-ray image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Processed image array or None if failed
        """
        try:
            # Load grayscale
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                logger.warning(f"Failed to load: {image_path}")
                return None
            
            # Quality check
            if self.config.validate_quality:
                sharpness = cv2.Laplacian(img, cv2.CV_64F).var()
                if sharpness < self.config.min_sharpness:
                    logger.debug(f"Low sharpness ({sharpness:.1f}): {image_path}")
            
            # Resize
            img = cv2.resize(img, self.config.target_size)
            
            # Apply CLAHE for contrast enhancement
            if self.config.apply_clahe:
                clahe = cv2.createCLAHE(
                    clipLimit=self.config.clahe_clip_limit,
                    tileGridSize=self.config.clahe_tile_grid
                )
                img = clahe.apply(img)
            
            # Normalize to 0-1
            if self.config.normalize:
                img = img.astype(np.float32) / 255.0
            
            return img
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return None
    
    def parse_labels(self, label_string: str) -> List[str]:
        """Parse multi-label string to list"""
        if pd.isna(label_string) or label_string == "No Finding":
            return []
        return [label.strip() for label in label_string.split("|")]
    
    def process_dataset(self, sample_size: Optional[int] = None) -> Dict:
        """
        Process full dataset into training-ready format.
        
        Args:
            sample_size: Optional limit on number of samples
            
        Returns:
            Dict with processing statistics
        """
        df = self.load_metadata()
        
        if sample_size:
            df = df.sample(n=min(sample_size, len(df)), random_state=42)
            logger.info(f"Sampled {len(df)} records for processing")
        
        processed_records = []
        failed = 0
        skipped = 0
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
            image_name = row["Image Index"]
            
            # Find image (might be in subdirectory)
            image_path = self._find_image(image_name)
            
            if image_path is None:
                skipped += 1
                continue
            
            # Process image
            img = self.preprocess_image(image_path)
            
            if img is None:
                failed += 1
                continue
            
            # Save processed image
            output_path = self.processed_dir / "images" / image_name.replace(".png", ".npy")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, img)
            
            # Record metadata
            processed_records.append({
                "image_id": image_name,
                "image_path": str(output_path),
                "labels": self.parse_labels(row["Finding Labels"]),
                "age": int(row["Patient Age"]) if pd.notna(row["Patient Age"]) else None,
                "sex": row["Patient Gender"],
                "view_position": row.get("View Position", "PA")
            })
        
        # Save metadata
        with open(self.processed_dir / "metadata.json", "w") as f:
            json.dump(processed_records, f, indent=2)
        
        # Create train/val/test splits
        self._create_splits(processed_records)
        
        stats = {
            "total": len(df),
            "processed": len(processed_records),
            "failed": failed,
            "skipped": skipped,
            "success_rate": len(processed_records) / len(df) * 100
        }
        
        logger.info(f"Processing complete: {stats}")
        return stats
    
    def _find_image(self, image_name: str) -> Optional[Path]:
        """Find image file in raw directory"""
        # Direct path
        direct = self.raw_dir / image_name
        if direct.exists():
            return direct
        
        # In images subdirectory
        images_dir = self.raw_dir / "images" / image_name
        if images_dir.exists():
            return images_dir
        
        # Search in numbered directories
        for subdir in self.raw_dir.glob("images_*"):
            path = subdir / image_name
            if path.exists():
                return path
        
        return None
    
    def _create_splits(self, records: List[Dict], ratios: tuple = (0.7, 0.15, 0.15)):
        """
        Create stratified train/val/test splits.
        
        Args:
            records: Processed records
            ratios: (train, val, test) proportions
        """
        np.random.seed(42)
        indices = np.random.permutation(len(records))
        
        n = len(records)
        train_end = int(ratios[0] * n)
        val_end = int((ratios[0] + ratios[1]) * n)
        
        splits = {
            "train": [records[i] for i in indices[:train_end]],
            "val": [records[i] for i in indices[train_end:val_end]],
            "test": [records[i] for i in indices[val_end:]]
        }
        
        for split_name, split_data in splits.items():
            path = self.processed_dir / f"{split_name}.json"
            with open(path, "w") as f:
                json.dump(split_data, f, indent=2)
            logger.info(f"{split_name}: {len(split_data)} samples")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process X-ray data")
    parser.add_argument("--raw-dir", default="data/raw/nih_chestxray")
    parser.add_argument("--processed-dir", default="data/processed/nih_chestxray")
    parser.add_argument("--sample-size", type=int, default=None)
    args = parser.parse_args()
    
    processor = XRayDataProcessor(args.raw_dir, args.processed_dir)
    stats = processor.process_dataset(sample_size=args.sample_size)
    
    print("\n=== Processing Complete ===")
    for key, value in stats.items():
        print(f"  {key}: {value}")
