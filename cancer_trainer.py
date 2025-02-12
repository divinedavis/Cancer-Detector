import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers.schedules import CosineDecay
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
        """Create a TensorFlow dataset with enhanced data augmentation."""
        self.logger.info("Creating augmented dataset...")

        image_size = 150 if self.dataset_type == 'brain_tumor' else 299

        datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=40,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.3,
            zoom_range=0.4,
            horizontal_flip=True,
            brightness_range=[0.7, 1.3],
            fill_mode='nearest',
            validation_split=0.2
        )

        train_dataset = datagen.flow_from_directory(
            self.data_dir,
            target_size=(image_size, image_size),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training'
        )

        val_dataset = datagen.flow_from_directory(
            self.data_dir,
            target_size=(image_size, image_size),
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation'
        )

        return train_dataset, val_dataset, image_size

    def build_model(self, image_size, num_classes):
        """Build an optimized CNN model using ResNet50 with fine-tuning."""
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
        for layer in base_model.layers[:-5]:  # Unfreeze last 5 layers for fine-tuning
            layer.trainable = True

        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.4),
            Dense(num_classes, activation='softmax')
        ])

        lr_schedule = CosineDecay(initial_learning_rate=0.0005, decay_steps=1000, alpha=0.0001)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        return model

    def train(self, epochs=30, batch_size=16):
        """Train the model with additional callbacks for better performance."""
        self.logger.info("Starting training...")

        train_dataset, val_dataset, image_size = self.create_augmented_dataset(batch_size=batch_size)
        num_classes = len(train_dataset.class_indices)
        model = self.build_model(image_size, num_classes)
        model.summary()

        class_weights = {i: 1.0 / count for i, count in enumerate(np.bincount(train_dataset.classes))}

        callbacks = [
            ModelCheckpoint('best_model_resnet.h5', save_best_only=True, monitor='val_accuracy', mode='max'),
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
        ]

        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )

        self.logger.info("Training completed.")
        model.save('final_model_resnet.keras')
        self.evaluate_model(model, val_dataset)

    def evaluate_model(self, model, val_dataset):
        """Evaluate model and display performance metrics."""
        val_images, val_labels = next(iter(val_dataset))
        predictions = model.predict(val_images)

        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(val_labels, axis=1)

        print(confusion_matrix(true_classes, predicted_classes))
        print(classification_report(true_classes, predicted_classes, zero_division=1))


def main():
    parser = argparse.ArgumentParser(description='Train a cancer detection model.')
    parser.add_argument('--dataset', type=str, required=True, help="Dataset type: 'mammogram' or 'brain_tumor'")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the dataset directory")
    parser.add_argument('--epochs', type=int, default=30, help="Number of epochs to train")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training")

    args = parser.parse_args()

    trainer = CancerTrainer(args.dataset, args.data_dir)
    trainer.train(epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
