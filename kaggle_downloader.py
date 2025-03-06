import os
import subprocess
from pathlib import Path

def download_kaggle_notebook(notebook_slug, output_dir):
    """
    Downloads a Kaggle notebook using the Kaggle API.
    
    Args:
        notebook_slug (str): The notebook slug (e.g., 'sartajbhuvaji/brain-tumor-classification-mri').
        output_dir (str): Directory to save the downloaded notebook.
    """
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Check if Kaggle API token is set up
    kaggle_json_path = Path.home() / '.kaggle' / 'kaggle.json'
    if not kaggle_json_path.exists():
        raise FileNotFoundError(
            "Kaggle API token not found. Please place 'kaggle.json' in " +
            f"{kaggle_json_path.parent}. See Kaggle API setup instructions."
        )
    
    # Command to download the notebook
    # Using 'kaggle kernels pull' to fetch the notebook source
    cmd = f"kaggle kernels pull {notebook_slug} -p {output_dir}"
    
    try:
        # Run the command
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"Download successful! Output:\n{result.stdout}")
        
        # Check if the file exists and rename it for clarity
        downloaded_file = output_path / f"{notebook_slug.split('/')[-1]}.ipynb"
        if downloaded_file.exists():
            final_file = output_path / "brain_tumor_classification_mri.ipynb"
            downloaded_file.rename(final_file)
            print(f"Notebook saved as: {final_file}")
        else:
            print(f"Warning: Expected file {downloaded_file} not found. Check Kaggle API output.")
    
    except subprocess.CalledProcessError as e:
        print(f"Error downloading notebook: {e}")
        print(f"Command output: {e.output}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    # Specify the notebook slug and output directory
    notebook_slug = "sartajbhuvaji/brain-tumor-classification-mri"
    output_dir = r"C:\Users\divin\Documents\Cancer Detector"
    
    # Run the download
    download_kaggle_notebook(notebook_slug, output_dir)