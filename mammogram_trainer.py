import tensorflow as tf
import numpy as np
from pathlib import Path
import logging
import os

class MammogramTrainer:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _parse_tfrecord(self, example_proto):
        """Parse TFRecord data with correct format"""
        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }
        
        parsed_features = tf.io.parse_single_example(example_proto, feature_description)
        
        # Convert raw bytes to uint8 tensor first
        image = tf.io.decode_raw(parsed_features['image'], tf.uint8)
        image = tf.reshape(image, [299, 299])  # Reshape to 2D
        image = tf.expand_dims(image, -1)  # Add channel dimension
        
        # Convert to float32 and normalize
        image = tf.cast(image, tf.float32) / 255.0
        
        # Get label
        label = tf.cast(parsed_features['label'], tf.float32)
        
        return image, label

    def create_dataset(self, tfrecord_path, batch_size=8, max_samples=1000):
        """Create a TensorFlow dataset from TFRecord file"""
        self.logger.info(f"Creating dataset from: {tfrecord_path}")
        
        # Create dataset with proper parsing
        dataset = tf.data.TFRecordDataset(tfrecord_path)
        dataset = dataset.take(max_samples)  # Limit dataset size
        dataset = dataset.map(self._parse_tfrecord, 
                            num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(1000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        # Count samples
        count = sum(1 for _ in dataset)
        self.logger.info(f"Dataset contains {count} batches")
        
        return dataset, count

    def build_model(self):
        """Create the CNN model"""
        model = tf.keras.Sequential([
            # Input layer
            tf.keras.layers.Input(shape=(299, 299, 1)),
            
            # First conv block
            tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),
            
            # Second conv block
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),
            
            # Third conv block
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),
            
            # Dense layers
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        return model

    def train(self, epochs=10, batch_size=8, max_time_hours=2):
        """Train the model with time limit"""
        self.logger.info("Starting training process...")
        
        # Load training data
        train_files = list(self.data_dir.glob('training10_*/training10_*.tfrecords'))
        self.logger.info(f"Found {len(train_files)} training files")
        
        if not train_files:
            raise ValueError("No training files found!")
        
        # Create training dataset
        train_dataset, train_count = self.create_dataset(
            str(train_files[0]), 
            batch_size=batch_size,
            max_samples=1000  # Limit dataset size
        )
        
        # Create validation dataset
        cv_data_path = self.data_dir / 'cv10_data' / 'cv10_data.tfrecords'
        if cv_data_path.exists():
            val_dataset, val_count = self.create_dataset(
                str(cv_data_path), 
                batch_size=batch_size,
                max_samples=200  # Limit validation size
            )
        else:
            self.logger.warning("No validation data found, using training data split")
            val_dataset = train_dataset.take(train_count // 5)
            train_dataset = train_dataset.skip(train_count // 5)
        
        # Build and compile model
        model = self.build_model()
        model.summary()
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        # Create directories
        model_dir = Path('models')
        logs_dir = Path('logs')
        model_dir.mkdir(exist_ok=True)
        logs_dir.mkdir(exist_ok=True)
        
        # Training callbacks with time limit
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                model_dir / 'best_model.h5',
                save_best_only=True,
                monitor='val_accuracy'
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=3,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=0.0001
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=str(logs_dir),
                histogram_freq=1
            ),
            tf.keras.callbacks.TimeLimit(
                max_time_hours * 3600,  # Convert hours to seconds
                verbose=1
            )
        ]
        
        # Train model
        try:
            history = model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
            
            self.logger.info("Training completed successfully!")
            return history, model
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise

def main():
    # Set up base directory and data directory
    base_dir = Path("C:/Users/divin/Documents/Cancer Detector")
    data_dir = base_dir / "raw_data/Kaggle dataset"
    
    # Initialize trainer
    trainer = MammogramTrainer(data_dir)
    
    try:
        # Train model with 2-hour time limit
        history, model = trainer.train(epochs=10, batch_size=8, max_time_hours=2)
        
        # Save final model
        model.save('models/final_mammogram_model.h5')
        
    except Exception as e:
        print(f"Error occurred during training: {str(e)}")

if __name__ == "__main__":
    main()