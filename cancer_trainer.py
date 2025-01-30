import tensorflow as tf
import numpy as np
from pathlib import Path
import logging
import argparse

class CancerTrainer:
    def __init__(self, dataset_type, data_dir):
        self.dataset_type = dataset_type
        self.data_dir = Path(data_dir)

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def create_dataset(self, batch_size=16):
        """Create a TensorFlow dataset depending on the dataset type."""
        self.logger.info("Creating dataset...")

        if self.dataset_type == 'mammogram':
            image_size = 299
            categories = ['malignant', 'benign']
        elif self.dataset_type == 'brain_tumor':
            image_size = 150
            categories = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
        else:
            raise ValueError("Invalid dataset type. Use 'mammogram' or 'brain_tumor'.")

        # Prepare file paths and labels
        image_paths = []
        labels = []

        for category in categories:
            category_path = self.data_dir / category
            label = categories.index(category)

            # Collect all images and corresponding labels
            for img_path in category_path.glob('*.jpg'):
                image_paths.append(str(img_path))  # Ensure paths are strings
                labels.append(label)

        # Convert lists to TensorFlow-compatible format
        image_paths = tf.constant(image_paths, dtype=tf.string)
        labels = tf.constant(labels, dtype=tf.int32)

        # Create a TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

        def load_image(img_path, label):
            img = tf.io.read_file(img_path)              # Read file from path
            img = tf.io.decode_jpeg(img, channels=1)      # Decode JPEG to grayscale
            img = tf.image.resize(img, [image_size, image_size])  # Resize image
            img = img / 255.0                             # Normalize image
            return img, label

        dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        return dataset, image_size

    def build_model(self, image_size, num_classes):
        """Build a CNN model."""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(image_size, image_size, 1)),

            # Convolutional layers
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),

            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),

            tf.keras.layers.Conv2D(128, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),

            # Fully connected layers
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1 if num_classes == 2 else num_classes, 
                                  activation='sigmoid' if num_classes == 2 else 'softmax')
        ])

        # Select appropriate loss function
        if num_classes == 2:
            loss = 'binary_crossentropy'
        else:
            loss = 'sparse_categorical_crossentropy'

        model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

        return model

    def train(self, epochs=10, batch_size=16):
        """Train the model."""
        self.logger.info("Starting training...")

        dataset, image_size = self.create_dataset(batch_size=batch_size)
        val_dataset = dataset.take(100)
        train_dataset = dataset.skip(100)

        num_classes = 4 if self.dataset_type == 'brain_tumor' else 2

        # Build and train the model
        model = self.build_model(image_size, num_classes)
        model.summary()

        model_dir = Path('models')
        model_dir.mkdir(exist_ok=True)

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                model_dir / f'best_{self.dataset_type}_model.h5',
                save_best_only=True,
                monitor='val_accuracy'
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            )
        ]

        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks
        )

        self.logger.info("Training completed.")
        model.save(model_dir / f'final_{self.dataset_type}_model.h5')
        return history

def main():
    parser = argparse.ArgumentParser(description='Train a cancer detection model.')
    parser.add_argument('--dataset', type=str, required=True, help="Dataset type: 'mammogram' or 'brain_tumor'")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the dataset directory")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs to train")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training")

    args = parser.parse_args()

    # Initialize and train the model
    trainer = CancerTrainer(args.dataset, args.data_dir)
    trainer.train(epochs=args.epochs, batch_size=args.batch_size)

if __name__ == "__main__":
    main()
