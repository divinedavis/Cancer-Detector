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
    
    # Command to download the notebook.
    # Using 'kaggle kernels pull' to fetch the notebook source.
    #
    # Passed as an argv list rather than a shell string: with shell=True, any
    # shell metacharacter in notebook_slug or output_dir (`;`, `&&`, backticks,
    # `$(...)`) would be interpreted by /bin/sh instead of treated as part of
    # the argument. Both are callable parameters, so a caller passing a path
    # with a space would already break this, and one passing untrusted input
    # would get command execution. An argv list removes the shell entirely.
    cmd = ["kaggle", "kernels", "pull", str(notebook_slug), "-p", str(output_dir)]

    try:
        # Run the command
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
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