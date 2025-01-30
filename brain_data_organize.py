import os
from pathlib import Path
import shutil
import logging
import zipfile

class DatasetOrganizer:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "raw_data" / "brain_tumor"
        self.download_dir = self.base_dir / "raw_data" / "brain_tumor" / "archive"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def organize_dataset(self):
        """Reorganize the dataset into proper structure"""
        try:
            # First, check if we need to extract archive
            zip_files = list(self.data_dir.glob('*.zip'))
            if zip_files:
                self.logger.info("Found zip file, extracting...")
                for zip_file in zip_files:
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        zip_ref.extractall(self.download_dir)

            # Look for source directories
            source_dirs = [
                self.download_dir / "Training",
                self.data_dir / "Training",
                self.download_dir,
                self.data_dir
            ]

            categories = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
            
            # Find and move files for each category
            for category in categories:
                self.logger.info(f"\nProcessing {category}...")
                dest_dir = self.data_dir / category
                dest_dir.mkdir(exist_ok=True)
                
                # Look for category files in all possible locations
                for source_dir in source_dirs:
                    category_dir = source_dir / category
                    if category_dir.exists():
                        self.logger.info(f"Found source: {category_dir}")
                        for img_file in category_dir.glob('*.jpg'):
                            dest_file = dest_dir / img_file.name
                            if not dest_file.exists():  # Avoid duplicates
                                shutil.copy2(img_file, dest_file)
                                self.logger.info(f"Copied: {img_file.name}")

            # Clean up extraction directory if it exists
            if self.download_dir.exists():
                shutil.rmtree(self.download_dir)
                self.logger.info("\nCleaned up temporary files")

            return True

        except Exception as e:
            self.logger.error(f"Error organizing dataset: {str(e)}")
            return False

    def verify_organization(self):
        """Verify the organization was successful"""
        categories = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
        
        self.logger.info("\nVerifying organization:")
        for category in categories:
            category_dir = self.data_dir / category
            if category_dir.exists():
                image_count = len(list(category_dir.glob('*.jpg')))
                self.logger.info(f"{category}: {image_count} images")
            else:
                self.logger.error(f"Missing category directory: {category}")

def main():
    base_dir = Path("C:/Users/divin/Documents/Cancer Detector")
    organizer = DatasetOrganizer(base_dir)
    
    print("Starting dataset organization...")
    if organizer.organize_dataset():
        organizer.verify_organization()
        print("\nDataset organization completed!")
    else:
        print("\nFailed to organize dataset.")

if __name__ == "__main__":
    main()