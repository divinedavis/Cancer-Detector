import os
from pathlib import Path
from PIL import Image, ImageStat
import logging
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def analyze_class_data(class_dir):
    """
    Analyze images in a given class directory.
    Returns:
      count: Number of images
      avg_resolution: Tuple (avg_width, avg_height)
      avg_brightness: Average brightness (0-255 scale)
    """
    image_files = list(class_dir.glob('*.jpg'))
    count = len(image_files)
    resolutions = []
    brightness_values = []
    
    for image_file in image_files:
        try:
            with Image.open(image_file) as img:
                resolutions.append(img.size)
                # Convert image to grayscale and compute average brightness
                grayscale = img.convert('L')
                stat = ImageStat.Stat(grayscale)
                brightness_values.append(stat.mean[0])
        except Exception as e:
            logger.error(f"Error processing {image_file}: {e}")
    
    if resolutions:
        avg_width = sum(w for w, h in resolutions) / len(resolutions)
        avg_height = sum(h for w, h in resolutions) / len(resolutions)
    else:
        avg_width, avg_height = None, None
        
    if brightness_values:
        avg_brightness = sum(brightness_values) / len(brightness_values)
    else:
        avg_brightness = None
    
    return count, (avg_width, avg_height), avg_brightness

def investigate_data_quality(base_dir, dataset_type='brain_tumor'):
    """
    Investigate class imbalance and image quality statistics for the dataset.
    Scans category directories and logs image counts, average resolution,
    and average brightness. Saves a CSV summary for further review.
    """
    data_dir = Path(base_dir) / "raw_data" / dataset_type
    categories = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
    
    summary = []
    
    for category in categories:
        class_dir = data_dir / category
        if not class_dir.exists():
            logger.error(f"Directory for category '{category}' not found.")
            continue
        count, avg_res, avg_brightness = analyze_class_data(class_dir)
        
        if avg_res[0] is not None:
            logger.info(f"Category: {category}")
            logger.info(f" - Image Count      : {count}")
            logger.info(f" - Avg Resolution   : {avg_res[0]:.1f} x {avg_res[1]:.1f}")
            logger.info(f" - Avg Brightness   : {avg_brightness:.1f}")
        else:
            logger.info(f"Category: {category} - No valid images found.")
        
        summary.append({
            'category': category,
            'image_count': count,
            'avg_width': avg_res[0] if avg_res[0] is not None else 0,
            'avg_height': avg_res[1] if avg_res[1] is not None else 0,
            'avg_brightness': avg_brightness if avg_brightness is not None else 0
        })
    
    # Save summary as CSV for further analysis
    df = pd.DataFrame(summary)
    csv_path = Path(base_dir) / f"{dataset_type}_data_quality_summary.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Data quality summary saved to: {csv_path}")

def main():
    # Set the base directory (adjust this path as needed)
    base_dir = "C:/Users/divin/Documents/Cancer Detector"
    investigate_data_quality(base_dir, dataset_type='brain_tumor')

if __name__ == "__main__":
    main()
