import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import logging
import argparse
from pathlib import Path

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

    def create_augmented_dataset(self, batch_size=16):
        """Create a TensorFlow dataset with data augmentation."""
        self.logger.info("Creating augmented dataset...")

        image_size = 150 if self.dataset_type == 'brain_tumor' else 299
        categories = ['glioma', 'meningioma', 'no_tumor', 'pituitary', 'class_5', 'class_6']

        datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.2
        )

        train_dataset = datagen.flow_from_directory(
            self.data_dir,
            target_size=(image_size, image_size),
            color_mode='rgb',  # Convert images to RGB
            batch_size=batch_size,
            class_mode='categorical',
            subset='training'
        )

        val_dataset = datagen.flow_from_directory(
            self.data_dir,
            target_size=(image_size, image_size),
            color_mode='rgb',
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation'
        )

        return train_dataset, val_dataset, image_size

    def build_model(self, image_size, num_classes):
        """Build a CNN model using transfer learning."""
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
        base_model.trainable = False

        model = Sequential([
            base_model,
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        return model

    def train(self, epochs=10, batch_size=16):
        """Train the model."""
        self.logger.info("Starting training...")

        train_dataset, val_dataset, image_size = self.create_augmented_dataset(batch_size=batch_size)
        num_classes = 6  # Update number of classes here

        model = self.build_model(image_size, num_classes)
        model.summary()

        callbacks = [
            ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy'),
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        ]

        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks
        )

        self.logger.info("Training completed.")
        model.save('final_model.h5')

        self.evaluate_model(model, val_dataset)

    def evaluate_model(self, model, val_dataset):
        """Evaluate model and display metrics."""
        val_images, val_labels = next(iter(val_dataset))
        predictions = model.predict(val_images)

        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(val_labels, axis=1)

        print(confusion_matrix(true_classes, predicted_classes))
        print(classification_report(true_classes, predicted_classes))

def main():
    parser = argparse.ArgumentParser(description='Train a cancer detection model.')
    parser.add_argument('--dataset', type=str, required=True, help="Dataset type: 'mammogram' or 'brain_tumor'")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the dataset directory")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs to train")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training")

    args = parser.parse_args()

    trainer = CancerTrainer(args.dataset, args.data_dir)
    trainer.train(epochs=args.epochs, batch_size=args.batch_size)

if __name__ == "__main__":
    main()
