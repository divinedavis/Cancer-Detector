import tensorflow as tf
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_first_record(file_path):
    try:
        # Read the TFRecord file
        dataset = tf.data.TFRecordDataset(file_path)
        
        # Get the first record
        for record in dataset.take(1):
            # Try basic feature description
            features = {
                'image': tf.io.VarLenFeature(tf.float32),
                'label': tf.io.FixedLenFeature([], tf.int64)
            }
            
            try:
                example = tf.io.parse_single_example(record, features)
                logger.info("Parsed with VarLenFeature")
                logger.info(f"Image shape: {example['image'].dense_shape}")
                logger.info(f"Label: {example['label']}")
                return
            except:
                logger.info("Failed VarLenFeature parsing, trying different format...")
            
            # Try parsing as raw bytes
            features = {
                'image': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.int64)
            }
            
            try:
                example = tf.io.parse_single_example(record, features)
                image_raw = example['image'].numpy()
                logger.info(f"Raw image size: {len(image_raw)} bytes")
                logger.info(f"Label: {example['label'].numpy()}")
                return
            except:
                logger.info("Failed string parsing...")
            
            # Print raw record info
            logger.info(f"Raw record size: {len(record.numpy())} bytes")
            logger.info(f"First few bytes: {record.numpy()[:20]}")
            
    except Exception as e:
        logger.error(f"Error reading file: {str(e)}")

def main():
    data_dir = Path("C:/Users/divin/Documents/raw_data/Kaggle dataset")
    tfrecord_path = str(next(data_dir.glob('training10_0/*.tfrecords')))
    
    logger.info(f"Checking file: {tfrecord_path}")
    check_first_record(tfrecord_path)

if __name__ == "__main__":
    main()