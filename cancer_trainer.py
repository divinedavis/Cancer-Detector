import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import Sequential
<<<<<<< HEAD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, Callback
=======
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
>>>>>>> 688e34beec54fcc961753e1bdadb225b2c6c971b
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.regularizers import l2
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
import logging
import argparse
from pathlib import Path
import albumentations as A
<<<<<<< HEAD
import os

tf.keras.mixed_precision.set_global_policy('mixed_float16')

class ValidationLogger(Callback):
    def __init__(self, val_dataset, log_freq=50):
        super().__init__()
        self.val_dataset = val_dataset
        self.log_freq = log_freq

    def on_train_batch_end(self, batch, logs=None):
        if (batch + 1) % self.log_freq == 0:
            val_loss, val_accuracy, val_auc, val_precision, val_recall = self.model.evaluate(self.val_dataset, verbose=0)
            logging.info(f"Step {batch + 1}: val_loss: {val_loss:.4f}, val_accuracy: {val_accuracy:.4f}, val_auc: {val_auc:.4f}, val_precision: {val_precision:.4f}, val_recall: {val_recall:.4f}")
=======

tf.keras.mixed_precision.set_global_policy('mixed_float16')
>>>>>>> 688e34beec54fcc961753e1bdadb225b2c6c971b

class CancerTrainer:
    def __init__(self, dataset_type, data_dir):
        self.dataset_type = dataset_type
        self.data_dir = Path(data_dir)
        self.image_size = 224
<<<<<<< HEAD
        self.expected_classes = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
=======
        self.expected_classes = ['glioma', 'meningioma', 'no_tumor', 'pituitary']  # Define expected classes
>>>>>>> 688e34beec54fcc961753e1bdadb225b2c6c971b
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        self.train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(translate_percent=0.1, scale=(0.9, 1.1), rotate=(-30, 30), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
<<<<<<< HEAD
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
=======
            A.GaussNoise(noise_scale=(10.0, 50.0), p=0.3),  # Updated for 2.0.5+
>>>>>>> 688e34beec54fcc961753e1bdadb225b2c6c971b
        ])

    def preprocess_image(self, image):
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)
        return cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB) / 255.0

    def preprocess_train(self, image):
        image = self.preprocess_image(image)
        image_uint8 = (image * 255).astype(np.uint8)
        augmented = self.train_transform(image=image_uint8)['image']
        return augmented / 255.0

<<<<<<< HEAD
    def create_augmented_dataset(self, batch_size=16):
=======
    def create_augmented_dataset(self, batch_size=32):
>>>>>>> 688e34beec54fcc961753e1bdadb225b2c6c971b
        self.logger.info("Creating augmented dataset...")
        
        train_datagen = ImageDataGenerator(
            preprocessing_function=self.preprocess_train, 
<<<<<<< HEAD
            validation_split=0.2
        )
        val_datagen = ImageDataGenerator(
            preprocessing_function=self.preprocess_image, 
            validation_split=0.2
        )

        train_dataset = train_datagen.flow_from_directory(
            self.data_dir, 
            target_size=(self.image_size, self.image_size), 
            batch_size=batch_size,
            class_mode='categorical', 
            subset='training', 
            shuffle=True,
            classes=self.expected_classes
        )
        val_dataset = val_datagen.flow_from_directory(
            self.data_dir, 
            target_size=(self.image_size, self.image_size), 
            batch_size=batch_size,
            class_mode='categorical', 
            subset='validation', 
            shuffle=False,
            classes=self.expected_classes
        )

=======
            validation_split=0.2,
            classes=self.expected_classes  # Restrict to 4 classes
        )
        val_datagen = ImageDataGenerator(
            preprocessing_function=self.preprocess_image, 
            validation_split=0.2,
            classes=self.expected_classes  # Restrict to 4 classes
        )

        train_dataset = train_datagen.flow_from_directory(
            self.data_dir, target_size=(self.image_size, self.image_size), batch_size=batch_size,
            class_mode='categorical', subset='training', shuffle=True
        )
        val_dataset = val_datagen.flow_from_directory(
            self.data_dir, target_size=(self.image_size, self.image_size), batch_size=batch_size,
            class_mode='categorical', subset='validation', shuffle=False
        )

        # Verify classes
>>>>>>> 688e34beec54fcc961753e1bdadb225b2c6c971b
        detected_classes = list(train_dataset.class_indices.keys())
        if set(detected_classes) != set(self.expected_classes):
            self.logger.error(f"Class mismatch! Expected {self.expected_classes}, but found {detected_classes}")
            raise ValueError("Class mismatch detected. Check dataset directory.")

<<<<<<< HEAD
        class_counts = {cls: len(os.listdir(self.data_dir / cls)) for cls in self.expected_classes}
        self.logger.info(f"Class distribution: {class_counts}")

=======
>>>>>>> 688e34beec54fcc961753e1bdadb225b2c6c971b
        return train_dataset, val_dataset

    def build_model(self, num_classes):
        base_model = EfficientNetV2S(weights='imagenet', include_top=False, input_shape=(self.image_size, self.image_size, 3))
<<<<<<< HEAD
        for layer in base_model.layers[:-100]:  # Adjusted to 100 trainable layers
=======
        for layer in base_model.layers[:-30]:
>>>>>>> 688e34beec54fcc961753e1bdadb225b2c6c971b
            layer.trainable = False

        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            BatchNormalization(),
            Dense(512, kernel_initializer='he_uniform', kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Activation('relu'),
<<<<<<< HEAD
            Dropout(0.6),
=======
            Dropout(0.5),
>>>>>>> 688e34beec54fcc961753e1bdadb225b2c6c971b
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
<<<<<<< HEAD
            optimizer=AdamW(learning_rate=5e-4, weight_decay=0.01),  # Back to 5e-4
=======
            optimizer=AdamW(learning_rate=1e-4, weight_decay=0.01),
>>>>>>> 688e34beec54fcc961753e1bdadb225b2c6c971b
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), 'recall', 'precision']
        )
        return model

<<<<<<< HEAD
    def train(self, epochs=50, batch_size=16):
=======
    def train(self, epochs=50, batch_size=32):
>>>>>>> 688e34beec54fcc961753e1bdadb225b2c6c971b
        self.logger.info("Starting training...")
        train_dataset, val_dataset = self.create_augmented_dataset(batch_size=batch_size)
        num_classes = len(train_dataset.class_indices)
        model = self.build_model(num_classes)
        model.summary()

        class_weights = compute_class_weight('balanced', classes=np.unique(train_dataset.classes), y=train_dataset.classes)
<<<<<<< HEAD
        self.logger.info(f"Class weights: {class_weights}")
        class_weights = dict(enumerate(class_weights))

        callbacks = [
            ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_auc', mode='max'),
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7),
            TensorBoard(log_dir='./logs'),
            ValidationLogger(val_dataset, log_freq=50)
=======
        class_weights = dict(enumerate(class_weights))

        callbacks = [
            ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_auc', mode='max'),
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7),
            TensorBoard(log_dir='./logs')
>>>>>>> 688e34beec54fcc961753e1bdadb225b2c6c971b
        ]

        history = model.fit(
            train_dataset, validation_data=val_dataset, epochs=epochs,
<<<<<<< HEAD
            class_weight=class_weights, callbacks=callbacks, verbose=1,
            validation_freq=1
=======
            class_weight=class_weights, callbacks=callbacks, verbose=1
>>>>>>> 688e34beec54fcc961753e1bdadb225b2c6c971b
        )
        model.save('final_model.keras')
        self.evaluate_model(model, val_dataset)

    def evaluate_model(self, model, val_dataset):
        self.logger.info("Running evaluation...")
        y_true, y_pred = [], []
        for i in range(len(val_dataset)):
            X_batch, y_batch = val_dataset[i]
            y_pred_batch = model.predict(X_batch, verbose=0)
            y_true.extend(np.argmax(y_batch, axis=1))
            y_pred.extend(np.argmax(y_pred_batch, axis=1))

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=list(val_dataset.class_indices.keys())))
        print("\nF1-Score per class:")
        print(f1_score(y_true, y_pred, average=None))

def main():
    parser = argparse.ArgumentParser(description='Train a cancer detection model.')
    parser.add_argument('--dataset', type=str, required=True, help="Dataset type: 'mammogram' or 'brain_tumor'")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to dataset directory")
    parser.add_argument('--epochs', type=int, default=50)
<<<<<<< HEAD
    parser.add_argument('--batch_size', type=int, default=16)
=======
    parser.add_argument('--batch_size', type=int, default=32)
>>>>>>> 688e34beec54fcc961753e1bdadb225b2c6c971b

    args = parser.parse_args()
    trainer = CancerTrainer(args.dataset, args.data_dir)
    trainer.train(epochs=args.epochs, batch_size=args.batch_size)

if __name__ == "__main__":
    main()