#!/usr/bin/env python3
"""
Data Download Script
Downloads public medical datasets for training and testing.
"""

import os
import sys
import requests
import zipfile
import tarfile
import gzip
import shutil
from pathlib import Path
from loguru import logger
from tqdm import tqdm


# Dataset configurations
DATASETS = {
    "nih_chest_xray_small": {
        "url": "https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz",
        "description": "NIH Chest X-ray14 sample (100 images for testing)",
        "size_mb": 50,
        "type": "images",
        "extract": True,
        "dest": "data/images/nih_sample"
    },
    "mimic_cxr_meta": {
        "url": "https://physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv.gz",
        "description": "MIMIC-CXR metadata (requires PhysioNet credentials)",
        "size_mb": 5,
        "type": "metadata",
        "extract": True,
        "dest": "data/metadata",
        "requires_auth": True
    },
    "icd10_codes": {
        "url": "https://www.cms.gov/files/zip/2024-code-descriptions-tabular-order.zip",
        "description": "ICD-10-CM Code Descriptions",
        "size_mb": 2,
        "type": "reference",
        "extract": True,
        "dest": "data/reference/icd10"
    },
    "snomed_sample": {
        "url": "https://download.nlm.nih.gov/umls/kss/SNOMED_CT_US_Edition/SNOMED_CT_US_Edition.zip",
        "description": "SNOMED CT US Edition sample (requires UMLS license)",
        "size_mb": 500,
        "type": "ontology",
        "extract": False,
        "dest": "data/ontology",
        "requires_auth": True
    }
}


def download_file(url: str, dest_path: Path, chunk_size: int = 8192) -> bool:
    """
    Download a file with progress bar.
    
    Args:
        url: URL to download
        dest_path: Destination file path
        chunk_size: Download chunk size
        
    Returns:
        True if successful
    """
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(dest_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=dest_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        return True
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Download failed: {e}")
        return False


def extract_archive(file_path: Path, dest_dir: Path) -> bool:
    """
    Extract compressed archive.
    
    Args:
        file_path: Path to archive
        dest_dir: Extraction destination
        
    Returns:
        True if successful
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if file_path.suffix == '.zip':
            with zipfile.ZipFile(file_path, 'r') as zf:
                zf.extractall(dest_dir)
                
        elif file_path.suffix == '.gz':
            if file_path.stem.endswith('.tar'):
                with tarfile.open(file_path, 'r:gz') as tf:
                    tf.extractall(dest_dir)
            else:
                # Single gzipped file
                output_path = dest_dir / file_path.stem
                with gzip.open(file_path, 'rb') as f_in:
                    with open(output_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                        
        elif file_path.suffix == '.tar':
            with tarfile.open(file_path, 'r') as tf:
                tf.extractall(dest_dir)
                
        else:
            logger.warning(f"Unknown archive format: {file_path.suffix}")
            return False
        
        logger.info(f"Extracted to {dest_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return False


def download_dataset(name: str, config: dict, base_dir: Path) -> bool:
    """
    Download and optionally extract a dataset.
    
    Args:
        name: Dataset name
        config: Dataset configuration
        base_dir: Base directory for downloads
        
    Returns:
        True if successful
    """
    logger.info(f"Downloading {name}: {config['description']}")
    
    if config.get("requires_auth"):
        logger.warning(f"Dataset {name} requires authentication. Skipping automatic download.")
        logger.info(f"Please download manually from: {config['url']}")
        return False
    
    dest_dir = base_dir / config["dest"]
    
    # Determine filename from URL
    filename = config["url"].split("/")[-1]
    download_path = base_dir / "downloads" / filename
    
    # Download
    if not download_path.exists():
        if not download_file(config["url"], download_path):
            return False
    else:
        logger.info(f"File already exists: {download_path}")
    
    # Extract if needed
    if config.get("extract"):
        if not extract_archive(download_path, dest_dir):
            return False
    else:
        # Just copy to destination
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(download_path, dest_dir / filename)
    
    logger.info(f"✓ {name} ready at {dest_dir}")
    return True


def create_sample_data(base_dir: Path):
    """
    Create sample/mock data for development.
    
    Args:
        base_dir: Base directory for data
    """
    logger.info("Creating sample data for development...")
    
    # Sample symptoms data
    symptoms_file = base_dir / "data" / "samples" / "sample_symptoms.json"
    symptoms_file.parent.mkdir(parents=True, exist_ok=True)
    
    import json
    sample_symptoms = [
        {
            "id": "SAMPLE001",
            "symptoms": "Patient presents with fever, dry cough, and shortness of breath for 3 days",
            "age": 45,
            "sex": "M",
            "expected_diagnosis": "COVID-19 or Pneumonia"
        },
        {
            "id": "SAMPLE002", 
            "symptoms": "Chest pain radiating to left arm, diaphoresis, nausea",
            "age": 62,
            "sex": "M",
            "expected_diagnosis": "Acute Myocardial Infarction"
        },
        {
            "id": "SAMPLE003",
            "symptoms": "Severe headache, neck stiffness, photophobia, fever",
            "age": 28,
            "sex": "F",
            "expected_diagnosis": "Meningitis"
        },
        {
            "id": "SAMPLE004",
            "symptoms": "Right lower quadrant abdominal pain, nausea, fever. Pain started periumbilical.",
            "age": 19,
            "sex": "M",
            "expected_diagnosis": "Appendicitis"
        },
        {
            "id": "SAMPLE005",
            "symptoms": "Productive cough with yellow sputum, fever, pleuritic chest pain",
            "age": 55,
            "sex": "F",
            "expected_diagnosis": "Bacterial Pneumonia"
        }
    ]
    
    with open(symptoms_file, 'w') as f:
        json.dump(sample_symptoms, f, indent=2)
    
    logger.info(f"✓ Sample symptoms created: {symptoms_file}")
    
    # Sample lab values
    labs_file = base_dir / "data" / "samples" / "sample_labs.json"
    
    sample_labs = [
        {
            "patient_id": "SAMPLE001",
            "labs": {"wbc": 15.2, "crp": 45.0, "procalcitonin": 0.8}
        },
        {
            "patient_id": "SAMPLE002",
            "labs": {"troponin": 0.5, "ck_mb": 18.0, "bnp": 450}
        },
        {
            "patient_id": "SAMPLE003",
            "labs": {"wbc": 18.0, "glucose": 85, "lactate": 2.5}
        }
    ]
    
    with open(labs_file, 'w') as f:
        json.dump(sample_labs, f, indent=2)
    
    logger.info(f"✓ Sample labs created: {labs_file}")
    
    # Create placeholder directories
    for dir_name in ["images", "metadata", "reference", "ontology", "embeddings"]:
        (base_dir / "data" / dir_name).mkdir(parents=True, exist_ok=True)
    
    logger.info("✓ Sample data creation complete")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download medical datasets for CDSS")
    parser.add_argument("--dataset", type=str, help="Specific dataset to download (or 'all')")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--sample", action="store_true", help="Create sample data only")
    parser.add_argument("--base-dir", type=str, default=".", help="Base directory")
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir).resolve()
    
    if args.list:
        print("\n📊 Available Datasets:\n")
        for name, config in DATASETS.items():
            auth = "🔒" if config.get("requires_auth") else "✓"
            print(f"  {auth} {name}")
            print(f"     {config['description']}")
            print(f"     Size: ~{config['size_mb']} MB | Type: {config['type']}")
            print()
        return
    
    if args.sample:
        create_sample_data(base_dir)
        return
    
    if args.dataset:
        if args.dataset == "all":
            for name, config in DATASETS.items():
                download_dataset(name, config, base_dir)
        elif args.dataset in DATASETS:
            download_dataset(args.dataset, DATASETS[args.dataset], base_dir)
        else:
            logger.error(f"Unknown dataset: {args.dataset}")
            sys.exit(1)
    else:
        # Default: create sample data
        logger.info("No dataset specified. Creating sample data...")
        create_sample_data(base_dir)
        print("\nTo download a specific dataset, run:")
        print("  python scripts/download_data.py --dataset <name>")
        print("\nTo see available datasets, run:")
        print("  python scripts/download_data.py --list")


if __name__ == "__main__":
    main()
