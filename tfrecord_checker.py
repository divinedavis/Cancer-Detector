import tensorflow as tf
import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def inspect_tfrecord(file_path):
    """Inspect the structure of a TFRecord file"""
    logger.info(f"Inspecting file: {file_path}")
    
    try:
        # Create dataset
        dataset = tf.data.TFRecordDataset(file_path)
        
        # Try different feature descriptions
        feature_descriptions = [
            {
                'image': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.int64)
            },
            {
                'feature': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.int64)
            },
            {
                'data': tf.io.FixedLenFeature([], tf.string),
                'target': tf.io.FixedLenFeature([], tf.int64)
            }
        ]
        
        # Try each feature description
        for desc in feature_descriptions:
            logger.info(f"Trying feature description: {desc}")
            try:
                def parse_fn(example_proto):
                    return tf.io.parse_single_example(example_proto, desc)
                
                parsed_dataset = dataset.map(parse_fn)
                first_element = next(iter(parsed_dataset))
                
                logger.info("Successfully parsed with features:")
                for key, value in first_element.items():
                    logger.info(f"- {key}: shape={value.shape if hasattr(value, 'shape') else 'scalar'}, dtype={value.dtype}")
                
                return desc
                
            except Exception as e:
                logger.info(f"Failed with this description: {str(e)}")
                continue
        
        logger.error("Could not determine the correct feature description")
        return None
        
    except Exception as e:
        logger.error(f"Error inspecting file: {str(e)}")
        return None

def main():
    # Path to your TFRecord file
    data_dir = Path("C:/Users/divin/Documents/raw_data/Kaggle dataset")
    tfrecord_path = str(next(data_dir.glob('training10_0/*.tfrecords')))
    
    logger.info("Starting TFRecord inspection...")
    
    # Inspect the file
    feature_desc = inspect_tfrecord(tfrecord_path)
    
    if feature_desc:
        logger.info(f"Found correct feature description: {feature_desc}")
    else:
        logger.error("Could not determine the file structure")

if __name__ == "__main__":
    main()