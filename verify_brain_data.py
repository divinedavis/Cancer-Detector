import os
from pathlib import Path
import json
import logging
from PIL import Image
import numpy as np

class DatasetVerifier:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "raw_data" / "brain_tumor"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def verify_structure(self):
        """Verify dataset directory structure and count images"""
        categories = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
        dataset_stats = {}
        
        for category in categories:
            category_dir = self.data_dir / category
            if not category_dir.exists():
                self.logger.error(f"Missing category directory: {category}")
                continue
            
            images = list(category_dir.glob('*.jpg'))
            dataset_stats[category] = {
                'count': len(images),
                'sample_resolution': self.get_image_stats(images[0]) if images else None
            }
        
        self.logger.info("\nDataset Statistics:")
        for category, stats in dataset_stats.items():
            self.logger.info(f"\n{category}:")
            self.logger.info(f"  - Images: {stats['count']}")
            if stats['sample_resolution']:
                self.logger.info(f"  - Sample Resolution: {stats['sample_resolution']}")
        
        return dataset_stats

    def get_image_stats(self, image_path):
        """Get image resolution and basic stats"""
        try:
            with Image.open(image_path) as img:
                return f"{img.size[0]}x{img.size[1]}"
        except Exception as e:
            self.logger.error(f"Error reading image {image_path}: {str(e)}")
            return None

    def verify_image_integrity(self, sample_size=5):
        """Verify a sample of images from each category can be opened"""
        categories = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
        
        for category in categories:
            category_dir = self.data_dir / category
            if not category_dir.exists():
                continue
            
            images = list(category_dir.glob('*.jpg'))
            sample = np.random.choice(images, min(sample_size, len(images)), replace=False)
            
            self.logger.info(f"\nVerifying {category} images:")
            for img_path in sample:
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                        self.logger.info(f"  ✓ {img_path.name} - OK")
                except Exception as e:
                    self.logger.error(f"  ✗ {img_path.name} - Failed: {str(e)}")

def main():
    base_dir = Path("C:/Users/divin/Documents/Cancer Detector")
    verifier = DatasetVerifier(base_dir)
    
    print("\nVerifying dataset structure...")
    stats = verifier.verify_structure()
    
    print("\nVerifying image integrity...")
    verifier.verify_image_integrity()

if __name__ == "__main__":
    main()