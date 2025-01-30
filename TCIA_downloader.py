import os
import requests
import pandas as pd
from pathlib import Path
import json
import time
import logging
from urllib.parse import urljoin

class TCIADownloader:
    def __init__(self, output_dir):
        self.base_url = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
        self.output_dir = Path(output_dir)
        self.collection = "CBIS-DDSM"

        # Setup logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            force=True
        )
        self.logger = logging.getLogger(__name__)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_patient_studies(self):
        """Get list of patient studies from CBIS-DDSM."""
        endpoint = f"{self.base_url}/getPatientStudy"
        params = {"Collection": self.collection}
        headers = {"Accept": "application/json"}

        try:
            self.logger.info(f"Requesting patient studies from {endpoint}")
            response = requests.get(endpoint, params=params, headers=headers)

            if response.status_code == 200:
                self.logger.info("Successfully retrieved patient studies.")
                return response.json()
            else:
                self.logger.error(f"API returned status code {response.status_code}")
                self.logger.error(f"Response content: {response.text}")
                return None

        except Exception as e:
            self.logger.error(f"Error fetching patient studies: {str(e)}")
            return None

    def get_series_info(self, study_instance_uid):
        """Get series information for a specific study."""
        endpoint = f"{self.base_url}/getSeries"
        params = {"StudyInstanceUID": study_instance_uid}
        headers = {"Accept": "application/json"}

        try:
            self.logger.info(f"Requesting series info for StudyInstanceUID: {study_instance_uid}")
            response = requests.get(endpoint, params=params, headers=headers)

            if response.status_code == 200:
                self.logger.info("Successfully retrieved series info.")
                return response.json()
            else:
                self.logger.error(f"Failed to retrieve series info. Status code: {response.status_code}")
                return None

        except Exception as e:
            self.logger.error(f"Error fetching series info: {str(e)}")
            return None

    def download_series_instance(self, series_instance_uid, case_type):
        """Download images for a specific series."""
        endpoint = f"{self.base_url}/getImage"
        params = {"SeriesInstanceUID": series_instance_uid}

        try:
            self.logger.info(f"Downloading series {series_instance_uid}")
            response = requests.get(endpoint, params=params, stream=True)

            if response.status_code == 200:
                save_path = self.output_dir / case_type / f"{series_instance_uid}.dcm"
                save_path.parent.mkdir(parents=True, exist_ok=True)

                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                self.logger.info(f"Successfully saved series to {save_path}")
                return str(save_path)
            else:
                self.logger.error(f"Download failed with status {response.status_code}")
                self.logger.error(f"Response content: {response.text}")
                return None

        except Exception as e:
            self.logger.error(f"Error downloading series: {str(e)}")
            return None

    def create_labels_csv(self, downloaded_cases):
        """Create labels.csv from downloaded cases."""
        try:
            labels_data = [
                {
                    'filename': f"{case['series_uid']}.dcm",
                    'label': 1 if case['type'] == 'malignant' else 0,
                    'case_type': case['type']
                }
                for case in downloaded_cases
            ]

            labels_df = pd.DataFrame(labels_data)
            labels_path = self.output_dir.parent / 'labels.csv'
            labels_df.to_csv(labels_path, index=False)
            self.logger.info(f"Labels saved to {labels_path}")

        except Exception as e:
            self.logger.error(f"Error creating labels CSV: {str(e)}")

    def download_dataset(self, num_cases=10):
        """Download a subset of CBIS-DDSM mammograms."""
        self.logger.info(f"Starting download of {num_cases} cases from {self.collection}")

        # Get patient studies
        studies = self.get_patient_studies()
        if not studies:
            self.logger.error("Failed to get patient studies.")
            return False

        self.logger.info(f"Retrieved {len(studies)} studies.")
        downloaded_cases = []
        count = 0

        try:
            for study in studies:
                if count >= num_cases:
                    break

                study_uid = study.get('StudyInstanceUID')
                if not study_uid:
                    continue

                # Get series for this study
                series_list = self.get_series_info(study_uid)
                if not series_list:
                    continue

                for series in series_list:
                    series_uid = series.get('SeriesInstanceUID')
                    if not series_uid:
                        continue

                    description = series.get('SeriesDescription', '').lower()
                    case_type = 'malignant' if 'malignant' in description else 'benign' if 'benign' in description else None
                    if not case_type:
                        continue

                    path = self.download_series_instance(series_uid, case_type)
                    if path:
                        downloaded_cases.append({'series_uid': series_uid, 'type': case_type, 'path': path})
                        count += 1

                    if count >= num_cases:
                        break

                    time.sleep(1)  # Rate limiting

        except Exception as e:
            self.logger.error(f"Error in download process: {str(e)}")
            return False

        if downloaded_cases:
            self.create_labels_csv(downloaded_cases)
            self.logger.info(f"Successfully downloaded {len(downloaded_cases)} cases.")
            return True
        else:
            self.logger.error("No cases were downloaded successfully.")
            return False

def main():
    current_dir = Path(os.getcwd())
    raw_data_dir = current_dir / "raw_data" / "mammogram"

    raw_data_dir.mkdir(parents=True, exist_ok=True)

    # Initialize downloader
    downloader = TCIADownloader(raw_data_dir)

    # Download a smaller test set first (10 images)
    success = downloader.download_dataset(num_cases=10)

    if success:
        print("\nDataset downloaded successfully!")
        print(f"Images are stored in: {raw_data_dir}")
        print(f"Labels file is stored in: {raw_data_dir.parent / 'labels.csv'}")
    else:
        print("\nFailed to download dataset. Check the logs above for errors.")

if __name__ == "__main__":
    main()
