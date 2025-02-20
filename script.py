from pathlib import Path

data_dir = Path("C:/Users/divin/Documents/Cancer Detector/data")
for subdir in data_dir.iterdir():
    if subdir.is_dir():
        image_files = list(subdir.glob('*'))
        print(f"Class: {subdir.name} - Number of images: {len(image_files)}")
