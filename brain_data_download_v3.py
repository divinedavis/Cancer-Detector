import os
import kaggle
from pathlib import Path
import logging
import subprocess
import shutil

class BrainTumorNotebookDownloader:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "raw_data" / "brain_tumor"
        self.temp_dir = self.base_dir / "temp_download"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def download_notebook_and_data(self):
        """Download notebook and associated data from Kaggle"""
        try:
            # Create temporary directory for download
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info("Downloading notebook and data from Kaggle...")
            
            # Use quotes around path to handle spaces
            command = f'kaggle kernels output huthayfahodeb/brain-tumor-detection-99-8-accuracy -p "{str(self.temp_dir)}"'
            
            # Run kaggle command
            self.logger.info(f"Running command: {command}")
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info("Download completed successfully")
                
                # Move files to final location
                self.data_dir.mkdir(parents=True, exist_ok=True)
                for file in self.temp_dir.glob('*'):
                    shutil.move(str(file), str(self.data_dir / file.name))
                
                # Clean up temp directory
                shutil.rmtree(self.temp_dir)
                return True
            else:
                self.logger.error("Download failed")
                self.logger.error(f"Error output: {result.stderr}")
                return False
            
        except Exception as e:
            self.logger.error(f"Error during download: {str(e)}")
            return False

    def verify_download(self):
        """Verify the downloaded content"""
        try:
            self.logger.info("\nVerifying downloaded content:")
            
            # List all files in the directory
            files = list(self.data_dir.rglob("*.*"))
            
            if not files:
                self.logger.error("No files found in download directory")
                return False
            
            # Print file information
            file_count = 0
            for file in files:
                self.logger.info(f"Found: {file.relative_to(self.data_dir)}")
                file_count += 1
            
            self.logger.info(f"\nTotal files found: {file_count}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during verification: {str(e)}")
            return False

def main():
    base_dir = Path("C:/Users/divin/Documents/Cancer Detector")
    downloader = BrainTumorNotebookDownloader(base_dir)
    
    print("Starting notebook and data download...")
    if downloader.download_notebook_and_data():
        print("\nVerifying downloaded content...")
        if downloader.verify_download():
            print("\nDownload completed successfully!")
        else:
            print("\nVerification failed. Please check the logs.")
    else:
        print("\nDownload failed. Please check the logs.")

if __name__ == "__main__":
    main()