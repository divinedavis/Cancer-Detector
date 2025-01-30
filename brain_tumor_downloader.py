import os
import kaggle
import logging
from pathlib import Path
import shutil

class BrainTumorDataDownloader:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "raw_data" / "brain_tumor"

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def setup_kaggle_credentials(self):
        """Setup Kaggle API credentials"""
        try:
            kaggle_dir = Path.home() / '.kaggle'
            kaggle_dir.mkdir(exist_ok=True)

            # Look for kaggle.json in common locations
            possible_locations = [
                self.base_dir / 'kaggle.json',
                Path.home() / 'Downloads' / 'kaggle.json'
            ]

            credential_found = False
            for cred_path in possible_locations:
                if cred_path.exists():
                    dest_path = kaggle_dir / 'kaggle.json'
                    shutil.copy(cred_path, dest_path)
                    os.chmod(dest_path, 0o600)
                    credential_found = True
                    self.logger.info(f"Kaggle credentials copied from {cred_path}")
                    break

            if not credential_found:
                self.logger.error("Kaggle credentials not found. Please download kaggle.json from kaggle.com.")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error setting up Kaggle credentials: {str(e)}")
            return False

    def download_dataset(self):
        """Download brain tumor dataset from Kaggle"""
        try:
            self.logger.info("Starting dataset download...")

            # Create data directory
            self.data_dir.mkdir(parents=True, exist_ok=True)

            # Download and unzip dataset
            kaggle.api.dataset_download_files(
                'sartajbhuvaji/brain-tumor-classification-mri',
                path=str(self.data_dir),
                unzip=True
            )

            self.logger.info(f"Dataset downloaded to {self.data_dir}")
            return True

        except Exception as e:
            self.logger.error(f"Error downloading dataset: {str(e)}")
            return False

    def organize_dataset(self):
        """Organize downloaded dataset into proper structure"""
        try:
            self.logger.info("Organizing dataset...")

            # Define categories and create destination directories
            category_mapping = {
                'glioma_tumor': 'glioma',
                'meningioma_tumor': 'meningioma',
                'pituitary_tumor': 'pituitary',
                'no_tumor': 'no_tumor'
            }

            for target_category in category_mapping.values():
                (self.data_dir / target_category).mkdir(exist_ok=True)

            # Search for images in the Training directory
            training_dir = self.data_dir / 'Training'
            if not training_dir.exists():
                self.logger.error("Training directory not found. Cannot organize dataset.")
                return False

            # Move images to their respective target directories
            for source_category, target_category in category_mapping.items():
                source_dir = training_dir / source_category
                if source_dir.exists():
                    self.logger.info(f"Moving images from {source_dir} to {target_category} directory...")
                    for img_file in source_dir.glob('*.jpg'):
                        shutil.move(str(img_file), str(self.data_dir / target_category / img_file.name))
                else:
                    self.logger.warning(f"Source category folder not found: {source_dir}")

            self.logger.info("Dataset organized successfully.")
            return True

        except Exception as e:
            self.logger.error(f"Error organizing dataset: {str(e)}")
            return False

def main():
    base_dir = Path("C:/Users/divin/Documents/Cancer Detector")

    # Initialize downloader
    downloader = BrainTumorDataDownloader(base_dir)

    # Setup credentials
    if not downloader.setup_kaggle_credentials():
        print("Failed to setup Kaggle credentials. Please ensure kaggle.json is available.")
        return

    # Download dataset
    if not downloader.download_dataset():
        print("Failed to download dataset.")
        return

    # Organize dataset
    if not downloader.organize_dataset():
        print("Failed to organize dataset.")
        return

    print("\nDataset setup completed successfully!")
    print(f"Data is organized in: {downloader.data_dir}")

if __name__ == "__main__":
    main()
