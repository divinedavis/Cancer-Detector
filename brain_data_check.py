import os
from pathlib import Path
import logging

class DatasetChecker:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "raw_data" / "brain_tumor"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def list_directory_contents(self):
        """List all contents of the data directory"""
        self.logger.info(f"\nExploring: {self.data_dir}")
        
        def explore_dir(directory, indent=0):
            try:
                for item in directory.iterdir():
                    if item.is_file():
                        self.logger.info(f"{'  ' * indent}📄 {item.name}")
                    else:
                        self.logger.info(f"{'  ' * indent}📁 {item.name}")
                        explore_dir(item, indent + 1)
            except Exception as e:
                self.logger.error(f"Error exploring {directory}: {str(e)}")
        
        explore_dir(self.data_dir)

    def find_all_images(self):
        """Find all image files recursively"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif'}
        image_files = []
        
        for ext in image_extensions:
            files = list(self.data_dir.rglob(f"*{ext}"))
            image_files.extend(files)
        
        self.logger.info("\nImage files found:")
        for img in image_files[:10]:  # Show first 10 files
            self.logger.info(f"Found: {img.relative_to(self.data_dir)}")
        
        if len(image_files) > 10:
            self.logger.info(f"... and {len(image_files) - 10} more files")
        
        return image_files

    def check_dataset_structure(self):
        """Check the current structure of the dataset"""
        expected_categories = {'glioma', 'meningioma', 'no_tumor', 'pituitary'}
        found_categories = set()
        category_counts = {}
        
        # Check immediate subdirectories
        for item in self.data_dir.iterdir():
            if item.is_dir():
                found_categories.add(item.name.lower())
                image_count = len(list(item.glob('*.jpg')))
                category_counts[item.name] = image_count
        
        self.logger.info("\nCategory Analysis:")
        self.logger.info(f"Expected categories: {expected_categories}")
        self.logger.info(f"Found categories: {found_categories}")
        self.logger.info("\nImage counts by category:")
        for category, count in category_counts.items():
            self.logger.info(f"{category}: {count} images")
        
        # Check for possible alternative locations
        archive_dir = self.data_dir / "archive"
        training_dir = self.data_dir / "Training"
        
        if archive_dir.exists():
            self.logger.info(f"\nFound archive directory: {archive_dir}")
            self.count_images_in_dir(archive_dir)
            
        if training_dir.exists():
            self.logger.info(f"\nFound training directory: {training_dir}")
            self.count_images_in_dir(training_dir)

    def count_images_in_dir(self, directory):
        """Count images in a directory and its subdirectories"""
        for item in directory.iterdir():
            if item.is_dir():
                image_count = len(list(item.glob('*.jpg')))
                if image_count > 0:
                    self.logger.info(f"  {item.name}: {image_count} images")

def main():
    base_dir = Path("C:/Users/divin/Documents/Cancer Detector")
    checker = DatasetChecker(base_dir)
    
    print("\nListing directory contents:")
    checker.list_directory_contents()
    
    print("\nChecking dataset structure:")
    checker.check_dataset_structure()
    
    print("\nFinding all images:")
    checker.find_all_images()

if __name__ == "__main__":
    main()