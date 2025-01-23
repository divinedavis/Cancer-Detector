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

def inspect_tfrecord_raw(file_path):
    """Inspect raw content of TFRecord file"""
    logger.info(f"Inspecting file: {file_path}")
    
    try:
        dataset = tf.data.TFRecordDataset(file_path)
        
        # Take first record
        for raw_record in dataset.take(1):
            # Print raw bytes info
            logger.info(f"Raw record size: {len(raw_record.numpy())} bytes")
            logger.info(f"First 100 bytes: {raw_record.numpy()[:100]}")
            
            try:
                # Try parsing as Example
                example = tf.train.Example()
                example.ParseFromString(raw_record.numpy())
                logger.info("Successfully parsed as tf.train.Example")
                logger.info(f"Features: {example.features.feature.keys()}")
                return True
            except Exception as e:
                logger.error(f"Failed to parse as Example: {str(e)}")
                
            try:
                # Try parsing as SequenceExample
                sequence_example = tf.train.SequenceExample()
                sequence_example.ParseFromString(raw_record.numpy())
                logger.info("Successfully parsed as tf.train.SequenceExample")
                logger.info(f"Context features: {sequence_example.context.feature.keys()}")
                logger.info(f"Sequence features: {sequence_example.feature_lists.feature_list.keys()}")
                return True
            except Exception as e:
                logger.error(f"Failed to parse as SequenceExample: {str(e)}")
            
            # Try reading as numpy array
            try:
                array_data = np.frombuffer(raw_record.numpy(), dtype=np.float32)
                logger.info(f"Successfully read as numpy array")
                logger.info(f"Array shape: {array_data.shape}")
                logger.info(f"Array type: {array_data.dtype}")
                return True
            except Exception as e:
                logger.error(f"Failed to read as numpy array: {str(e)}")
            
            return False
            
    except Exception as e:
        logger.error(f"Error reading file: {str(e)}")
        return False

def main():
    # Path to your TFRecord file
    data_dir = Path("C:/Users/divin/Documents/raw_data/Kaggle dataset")
    tfrecord_path = str(next(data_dir.glob('training10_0/*.tfrecords')))
    
    logger.info("Starting detailed TFRecord inspection...")
    success = inspect_tfrecord_raw(tfrecord_path)
    
    if not success:
        logger.error("Could not determine the file format")
    
if __name__ == "__main__":
    main()