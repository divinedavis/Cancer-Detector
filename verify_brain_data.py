import os
from pathlib import Path
import shutil
import logging

class DatasetFixer:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "raw_data" / "brain_tumor"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def fix_organization(self):
        """Fix dataset organization by checking all possible locations"""
        try:
            # Check common directory names after unzip
            possible_dirs = [
                self.data_dir / "Training",
                self.data_dir / "Training Data",
                self.data_dir / "archive" / "Training",
                self.data_dir
            ]

            source_dir = None
            for dir_path in possible_dirs:
                if dir_path.exists():
                    source_dir = dir_path
                    self.logger.info(f"Found source directory: {dir_path}")
                    break

            if not source_dir:
                self.logger.error("Could not find source directory")
                return False

            # Categories to look for
            categories = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
            
            for category in categories:
                # Look for category directory
                category_source = source_dir / category
                if not category_source.exists():
                    self.logger.warning(f"Could not find source for {category}")
                    continue

                # Create destination directory
                category_dest = self.data_dir / category
                category_dest.mkdir(exist_ok=True)

                # Move files
                file_count = 0
                for img_file in category_source.glob('*.*'):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        shutil.copy2(img_file, category_dest / img_file.name)
                        file_count += 1

                self.logger.info(f"Moved {file_count} files to {category}")

            # Clean up temporary directories
            self.cleanup_temp_dirs()
            
            return True

        except Exception as e:
            self.logger.error(f"Error fixing dataset: {str(e)}")
            return False

    def cleanup_temp_dirs(self):
        """Clean up temporary directories after organizing"""
        try:
            temp_dirs = ['Training', 'Training Data', 'archive']
            for temp_dir in temp_dirs:
                temp_path = self.data_dir / temp_dir
                if temp_path.exists():
                    shutil.rmtree(temp_path)
                    self.logger.info(f"Cleaned up {temp_dir}")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")

    def verify_fix(self):
        """Verify the fix was successful"""
        categories = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
        all_good = True
        
        for category in categories:
            category_dir = self.data_dir / category
            if not category_dir.exists():
                self.logger.error(f"Missing category directory: {category}")
                all_good = False
                continue
            
            file_count = len(list(category_dir.glob('*.*')))
            self.logger.info(f"{category}: {file_count} images")
            
            if file_count == 0:
                self.logger.error(f"No images in {category}")
                all_good = False

        return all_good

def main():
    base_dir = Path("C:/Users/divin/Documents/Cancer Detector")
    fixer = DatasetFixer(base_dir)
    
    print("Fixing dataset organization...")
    if fixer.fix_organization():
        print("\nVerifying fix...")
        if fixer.verify_fix():
            print("\nDataset organization fixed successfully!")
        else:
            print("\nFix verification failed. Please check the logs.")
    else:
        print("\nFailed to fix dataset organization.")

if __name__ == "__main__":
    main()