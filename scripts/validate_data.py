"""
Data Validation Pipeline
Validate processed data quality before training or deployment.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ValidationResult:
    """Result of a validation check"""
    name: str
    passed: bool
    message: str
    details: Dict = None


class DataValidator:
    """Validate processed medical data quality"""
    
    def __init__(self, processed_dir: str):
        """
        Initialize validator.
        
        Args:
            processed_dir: Path to processed data directory
        """
        self.processed_dir = Path(processed_dir)
        self.results: List[ValidationResult] = []
    
    def validate_all(self) -> Dict:
        """
        Run all validation checks.
        
        Returns:
            Dict with validation results
        """
        self.results = []
        
        checks = [
            self._validate_metadata,
            self._validate_images,
            self._validate_labels,
            self._validate_splits,
            self._check_data_quality,
            self._check_label_distribution
        ]
        
        for check in checks:
            try:
                result = check()
                self.results.append(result)
            except Exception as e:
                self.results.append(ValidationResult(
                    name=check.__name__,
                    passed=False,
                    message=f"Check failed with error: {e}"
                ))
        
        # Aggregate
        all_passed = all(r.passed for r in self.results)
        
        return {
            "overall_passed": all_passed,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "message": r.message,
                    "details": r.details
                }
                for r in self.results
            ],
            "summary": f"{sum(r.passed for r in self.results)}/{len(self.results)} checks passed"
        }
    
    def _load_metadata(self) -> List[Dict]:
        """Load metadata file"""
        metadata_path = self.processed_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError("metadata.json not found")
        
        with open(metadata_path) as f:
            return json.load(f)
    
    def _validate_metadata(self) -> ValidationResult:
        """Validate metadata file structure"""
        try:
            data = self._load_metadata()
            
            required_fields = ["image_id", "image_path", "labels", "age", "sex"]
            
            issues = []
            for i, record in enumerate(data[:100]):  # Check first 100
                missing = [f for f in required_fields if f not in record]
                if missing:
                    issues.append(f"Record {i}: missing {missing}")
            
            if issues:
                return ValidationResult(
                    name="metadata_validation",
                    passed=False,
                    message=f"Found {len(issues)} records with missing fields",
                    details={"issues": issues[:10]}
                )
            
            return ValidationResult(
                name="metadata_validation",
                passed=True,
                message=f"Validated {len(data)} records",
                details={"record_count": len(data)}
            )
            
        except Exception as e:
            return ValidationResult(
                name="metadata_validation",
                passed=False,
                message=str(e)
            )
    
    def _validate_images(self) -> ValidationResult:
        """Validate image files exist and have correct format"""
        try:
            data = self._load_metadata()
            sample = data[:100]  # Check 100 random
            
            missing = 0
            wrong_shape = 0
            corrupt = 0
            
            for record in sample:
                img_path = Path(record["image_path"])
                
                if not img_path.exists():
                    missing += 1
                    continue
                
                try:
                    img = np.load(img_path)
                    if img.shape != (224, 224):
                        wrong_shape += 1
                except Exception:
                    corrupt += 1
            
            passed = missing == 0 and wrong_shape == 0 and corrupt == 0
            
            return ValidationResult(
                name="image_validation",
                passed=passed,
                message=f"Checked {len(sample)} images",
                details={
                    "checked": len(sample),
                    "missing": missing,
                    "wrong_shape": wrong_shape,
                    "corrupt": corrupt
                }
            )
            
        except Exception as e:
            return ValidationResult(
                name="image_validation",
                passed=False,
                message=str(e)
            )
    
    def _validate_labels(self) -> ValidationResult:
        """Validate label consistency"""
        try:
            data = self._load_metadata()
            
            # Valid disease labels
            valid_labels = {
                "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
                "Mass", "Nodule", "Pneumonia", "Pneumothorax",
                "Consolidation", "Edema", "Emphysema", "Fibrosis",
                "Pleural_Thickening", "Hernia", "No Finding"
            }
            
            invalid_labels = set()
            for record in data:
                for label in record.get("labels", []):
                    if label not in valid_labels:
                        invalid_labels.add(label)
            
            passed = len(invalid_labels) == 0
            
            return ValidationResult(
                name="label_validation",
                passed=passed,
                message=f"Found {len(invalid_labels)} invalid labels" if invalid_labels else "All labels valid",
                details={
                    "invalid_labels": list(invalid_labels)[:10],
                    "valid_labels": list(valid_labels)
                }
            )
            
        except Exception as e:
            return ValidationResult(
                name="label_validation",
                passed=False,
                message=str(e)
            )
    
    def _validate_splits(self) -> ValidationResult:
        """Validate train/val/test splits exist and don't overlap"""
        try:
            splits = {}
            
            for split in ["train", "val", "test"]:
                path = self.processed_dir / f"{split}.json"
                if path.exists():
                    with open(path) as f:
                        splits[split] = json.load(f)
            
            if len(splits) < 3:
                return ValidationResult(
                    name="split_validation",
                    passed=False,
                    message=f"Missing splits: {set(['train', 'val', 'test']) - set(splits.keys())}"
                )
            
            # Check for overlap
            train_ids = set(r["image_id"] for r in splits["train"])
            val_ids = set(r["image_id"] for r in splits["val"])
            test_ids = set(r["image_id"] for r in splits["test"])
            
            overlap_train_val = train_ids & val_ids
            overlap_train_test = train_ids & test_ids
            overlap_val_test = val_ids & test_ids
            
            has_overlap = any([overlap_train_val, overlap_train_test, overlap_val_test])
            
            return ValidationResult(
                name="split_validation",
                passed=not has_overlap,
                message="Splits are valid and non-overlapping" if not has_overlap else "Found overlapping records",
                details={
                    "train_count": len(train_ids),
                    "val_count": len(val_ids),
                    "test_count": len(test_ids),
                    "overlap_train_val": len(overlap_train_val),
                    "overlap_train_test": len(overlap_train_test),
                    "overlap_val_test": len(overlap_val_test)
                }
            )
            
        except Exception as e:
            return ValidationResult(
                name="split_validation",
                passed=False,
                message=str(e)
            )
    
    def _check_data_quality(self) -> ValidationResult:
        """Check for data quality issues"""
        try:
            data = self._load_metadata()
            
            issues = []
            
            # Check age range
            ages = [r["age"] for r in data if r.get("age") is not None]
            if ages:
                if min(ages) < 0:
                    issues.append("Negative age values found")
                if max(ages) > 120:
                    issues.append("Age values > 120 found")
            
            # Check sex values
            sexes = set(r["sex"] for r in data if r.get("sex"))
            valid_sexes = {"M", "F", "Male", "Female"}
            invalid_sexes = sexes - valid_sexes
            if invalid_sexes:
                issues.append(f"Invalid sex values: {invalid_sexes}")
            
            # Check for empty labels
            empty_labels = sum(1 for r in data if not r.get("labels"))
            if empty_labels > len(data) * 0.5:  # More than 50% empty
                issues.append(f"{empty_labels} records have no labels")
            
            return ValidationResult(
                name="data_quality",
                passed=len(issues) == 0,
                message=f"Found {len(issues)} quality issues" if issues else "Data quality OK",
                details={
                    "issues": issues,
                    "age_range": [min(ages), max(ages)] if ages else None,
                    "sexes": list(sexes),
                    "total_records": len(data)
                }
            )
            
        except Exception as e:
            return ValidationResult(
                name="data_quality",
                passed=False,
                message=str(e)
            )
    
    def _check_label_distribution(self) -> ValidationResult:
        """Check for class imbalance"""
        try:
            data = self._load_metadata()
            
            label_counts = {}
            for record in data:
                for label in record.get("labels", []):
                    label_counts[label] = label_counts.get(label, 0) + 1
            
            if not label_counts:
                return ValidationResult(
                    name="label_distribution",
                    passed=True,
                    message="No labels found (might be normal for some datasets)"
                )
            
            max_count = max(label_counts.values())
            min_count = min(label_counts.values())
            imbalance_ratio = max_count / max(min_count, 1)
            
            # Warn if ratio > 100
            passed = imbalance_ratio < 100
            
            return ValidationResult(
                name="label_distribution",
                passed=passed,
                message=f"Imbalance ratio: {imbalance_ratio:.1f}" + (" (severe imbalance)" if not passed else ""),
                details={
                    "label_counts": label_counts,
                    "imbalance_ratio": imbalance_ratio,
                    "most_common": max(label_counts, key=label_counts.get),
                    "least_common": min(label_counts, key=label_counts.get)
                }
            )
            
        except Exception as e:
            return ValidationResult(
                name="label_distribution",
                passed=False,
                message=str(e)
            )


def print_validation_report(results: Dict):
    """Print formatted validation report"""
    print("\n" + "=" * 50)
    print("DATA VALIDATION REPORT")
    print("=" * 50)
    print(f"Timestamp: {results['timestamp']}")
    print(f"Overall: {'✅ PASSED' if results['overall_passed'] else '❌ FAILED'}")
    print(f"Summary: {results['summary']}")
    print("-" * 50)
    
    for check in results['checks']:
        status = "✅" if check['passed'] else "❌"
        print(f"{status} {check['name']}: {check['message']}")
    
    print("=" * 50)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate processed data")
    parser.add_argument("--data-dir", default="data/processed/nih_chestxray")
    args = parser.parse_args()
    
    validator = DataValidator(args.data_dir)
    results = validator.validate_all()
    print_validation_report(results)
