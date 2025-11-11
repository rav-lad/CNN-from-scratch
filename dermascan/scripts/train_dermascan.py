"""
Training Script for DermaScan Model

Train the CNN model on dermatology dataset (HAM10000).

Usage:
    python -m dermascan.scripts.train_dermascan --config dermascan/configs/dermascan_model.yaml
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Train DermaScan model")
    parser.add_argument(
        "--config",
        type=str,
        default="dermascan/configs/dermascan_model.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/dermatology",
        help="Path to data directory"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("DermaScan Model Training")
    print("=" * 70)
    print()
    print(f"Config: {args.config}")
    print(f"Data directory: {args.data_dir}")
    print()

    # Check if data exists
    data_path = Path(args.data_dir) / "raw" / "HAM10000"
    if not data_path.exists():
        print("❌ Error: HAM10000 dataset not found!")
        print()
        print("Please download the dataset first:")
        print("  python -m dermascan.scripts.download_data --dataset ham10000")
        print()
        return

    # Check if config exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Error: Config file not found: {args.config}")
        return

    print("✓ Dataset found")
    print("✓ Config found")
    print()

    # TODO: Implement actual training
    # This would use the existing src.cli.train infrastructure
    # with dermatology-specific data loader

    print("🚀 Training not yet implemented")
    print()
    print("Next steps:")
    print("1. Create data loader for HAM10000 in dermascan/data/ham10000.py")
    print("2. Preprocess images to 224x224")
    print("3. Use src.train.loop for training")
    print("4. Save best model to data/dermatology/models/")
    print()
    print("For now, you can use the generic training command:")
    print(f"  python -m src.cli.train --config {args.config}")
    print()


if __name__ == "__main__":
    main()
