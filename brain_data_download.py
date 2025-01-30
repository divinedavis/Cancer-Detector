import os
import kaggle
import zipfile
from pathlib import Path
import shutil
import logging

class BrainDataDownloader:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "raw_data" / "brain_tumor"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def download_dataset(self):
        """Download brain tumor dataset from Kaggle"""
        try:
            # Create data directory
            self.data_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info("Downloading dataset from Kaggle...")
            
            # Download dataset
            kaggle.api.dataset_download_files(
                'sartajbhuvaji/brain-tumor-classification-mri',
                path=str(self.data_dir),
                unzip=True
            )
            
            self.logger.info("Dataset downloaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error downloading dataset: {str(e)}")
            return False

    def organize_files(self):
        """Organize downloaded files into proper structure"""
        try:
            # Expected directory structure after download
            training_dir = self.data_dir / "Training"
            
            if not training_dir.exists():
                self.logger.error(f"Training directory not found at {training_dir}")
                return False
            
            # Move files from Training directory
            categories = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
            
            for category in categories:
                source_dir = training_dir / category
                dest_dir = self.data_dir / category
                
                if not source_dir.exists():
                    self.logger.error(f"Source directory not found: {source_dir}")
                    continue
                
                # Create destination directory
                dest_dir.mkdir(exist_ok=True)
                
                # Move files
                for img_file in source_dir.glob('*.jpg'):
                    shutil.copy2(img_file, dest_dir / img_file.name)
                    self.logger.info(f"Moved {img_file.name} to {category}")
            
            # Clean up original directories
            if training_dir.exists():
                shutil.rmtree(training_dir)
            
            self.logger.info("Files organized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error organizing files: {str(e)}")
            return False

    def verify_download(self):
        """Verify the download and organization"""
        categories = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
        
        self.logger.info("\nVerifying dataset:")
        for category in categories:
            category_dir = self.data_dir / category
            if category_dir.exists():
                image_count = len(list(category_dir.glob('*.jpg')))
                self.logger.info(f"{category}: {image_count} images")
            else:
                self.logger.error(f"Missing category directory: {category}")

def main():
    base_dir = Path("C:/Users/divin/Documents/Cancer Detector")
    downloader = BrainDataDownloader(base_dir)
    
    print("Starting dataset download...")
    if downloader.download_dataset():
        print("\nOrganizing files...")
        if downloader.organize_files():
            print("\nVerifying download...")
            downloader.verify_download()
            print("\nSetup completed successfully!")
        else:
            print("\nFailed to organize files.")
    else:
        print("\nFailed to download dataset.")

if __name__ == "__main__":
    main()