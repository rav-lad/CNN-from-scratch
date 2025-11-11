"""
Script to download dermatology datasets

Supports:
- HAM10000 dataset (recommended)
- ISIC datasets

Run: python -m dermascan.scripts.download_data --dataset ham10000
"""

import argparse
from pathlib import Path
import sys


def download_ham10000(data_dir: Path):
    """
    Download HAM10000 dataset

    The HAM10000 dataset contains 10,015 dermatoscopic images of pigmented lesions.

    Manual download instructions:
    1. Visit: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
    2. Download the dataset
    3. Extract to: data/dermatology/raw/HAM10000/

    Args:
        data_dir: Directory to save the dataset
    """
    print("=" * 70)
    print("HAM10000 Dataset Download Instructions")
    print("=" * 70)
    print()
    print("The HAM10000 dataset must be downloaded manually from Kaggle.")
    print()
    print("Steps:")
    print("1. Create a Kaggle account at https://www.kaggle.com")
    print("2. Visit: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000")
    print("3. Click 'Download' button")
    print("4. Extract the downloaded files to:")
    print(f"   {data_dir / 'raw' / 'HAM10000'}")
    print()
    print("Expected structure:")
    print("   data/dermatology/raw/HAM10000/")
    print("   ├── HAM10000_images_part_1/")
    print("   ├── HAM10000_images_part_2/")
    print("   └── HAM10000_metadata.csv")
    print()
    print("Alternative: Use Kaggle API")
    print("   pip install kaggle")
    print("   kaggle datasets download -d kmader/skin-cancer-mnist-ham10000")
    print("   unzip skin-cancer-mnist-ham10000.zip -d data/dermatology/raw/HAM10000")
    print()


def download_isic(data_dir: Path):
    """
    Download ISIC dataset

    Instructions for downloading ISIC dataset.

    Args:
        data_dir: Directory to save the dataset
    """
    print("=" * 70)
    print("ISIC Dataset Download Instructions")
    print("=" * 70)
    print()
    print("Visit: https://www.isic-archive.com/")
    print("1. Create an account")
    print("2. Navigate to 'Gallery'")
    print("3. Apply filters for desired lesion types")
    print("4. Download images and metadata")
    print("5. Extract to:")
    print(f"   {data_dir / 'raw' / 'ISIC'}")
    print()


def check_dataset(data_dir: Path, dataset_name: str) -> bool:
    """
    Check if dataset already exists

    Args:
        data_dir: Data directory
        dataset_name: Name of the dataset

    Returns:
        True if dataset exists
    """
    dataset_path = data_dir / 'raw' / dataset_name.upper()
    if dataset_path.exists():
        print(f"✓ Dataset found at: {dataset_path}")
        print(f"  Files: {len(list(dataset_path.rglob('*')))} items")
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Download dermatology datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["ham10000", "isic"],
        default="ham10000",
        help="Dataset to download (default: ham10000)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/dermatology",
        help="Data directory (default: data/dermatology)"
    )

    args = parser.parse_args()

    # Create data directory
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / 'raw').mkdir(exist_ok=True)
    (data_dir / 'processed').mkdir(exist_ok=True)

    print(f"\nData directory: {data_dir.absolute()}\n")

    # Check if dataset exists
    if check_dataset(data_dir, args.dataset):
        print("\nDataset already downloaded!")
        return

    # Download dataset
    if args.dataset == "ham10000":
        download_ham10000(data_dir)
    elif args.dataset == "isic":
        download_isic(data_dir)


if __name__ == "__main__":
    main()
