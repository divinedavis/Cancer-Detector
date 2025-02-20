import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.regularizers import l2
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import logging
import argparse
from pathlib import Path
import albumentations as A

# Enable mixed precision early so that the model builds with the mixed_float16 policy.
tf.keras.mixed_precision.set_global_policy('mixed_float16')

class CancerTrainer:
    def __init__(self, dataset_type, data_dir):
        self.dataset_type = dataset_type
        self.data_dir = Path(data_dir)
        self.image_size = 224  # Standard size for ResNet50
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Define Albumentations augmentation pipeline (used only for training)
        self.train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
            # Removed alpha_affine to avoid warning from ElasticTransform.
            A.ElasticTransform(alpha=1, sigma=50, p=0.3),
            A.GridDistortion(p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.3),
        ])

    def preprocess_image(self, image):
        """Enhanced preprocessing for medical images using CLAHE"""
        # Ensure the image is in uint8 format (values 0-255)
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        # Convert to grayscale and apply CLAHE
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)
        # Merge back to 3 channels and normalize to [0, 1]
        return cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB) / 255.0

    def preprocess_train(self, image):
        """Preprocess and apply Albumentations augmentations for training data"""
        # First, apply CLAHE-based preprocessing
        image = self.preprocess_image(image)
        # Albumentations expects uint8 images, so convert back temporarily
        image_uint8 = (image * 255).astype(np.uint8)
        augmented = self.train_transform(image=image_uint8)['image']
        # Normalize the augmented image to [0, 1]
        return augmented / 255.0

    def create_augmented_dataset(self, batch_size=32):
        """Create datasets with Albumentations augmentations applied to training images."""
        self.logger.info("Creating augmented dataset with medical-specific augmentations...")
        
        # For training: use the combined preprocessing and augmentation function.
        train_datagen = ImageDataGenerator(
            preprocessing_function=self.preprocess_train,
            validation_split=0.2,
            dtype=np.float32
        )
        
        # For validation: use only the CLAHE-based preprocessing.
        val_datagen = ImageDataGenerator(
            preprocessing_function=self.preprocess_image,
            validation_split=0.2,
            dtype=np.float32
        )

        train_dataset = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=(self.image_size, self.image_size),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )

        val_dataset = val_datagen.flow_from_directory(
            self.data_dir,
            target_size=(self.image_size, self.image_size),
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )

        return train_dataset, val_dataset

    def build_model(self, num_classes):
        """Build optimized model with fine-tuning capabilities"""
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(self.image_size, self.image_size, 3)
        )

        # Fine-tune last 50 layers
        for layer in base_model.layers[:-50]:
            layer.trainable = False
        for layer in base_model.layers[-50:]:
            layer.trainable = True

        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            BatchNormalization(),
            Dense(512, kernel_initializer='he_uniform', kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.5),
            Dense(256, kernel_initializer='he_uniform', kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Dense(num_classes, activation='softmax')
        ])

        # Optimizer with cosine decay learning rate schedule
        initial_learning_rate = 0.0001
        lr_schedule = CosineDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=1000,
            alpha=0.00001
        )
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss='categorical_crossentropy',
            metrics=['accuracy', 
                     tf.keras.metrics.AUC(name='auc'),
                     tf.keras.metrics.Recall(name='recall'),
                     tf.keras.metrics.Precision(name='precision')]
        )
        
        return model

    def train(self, epochs=50, batch_size=32):
        """Enhanced training process with comprehensive monitoring"""
        self.logger.info("Starting optimized training process...")

        train_dataset, val_dataset = self.create_augmented_dataset(batch_size=batch_size)
        num_classes = len(train_dataset.class_indices)
        model = self.build_model(num_classes)
        model.summary()

        # Compute class weights to handle any class imbalance.
        classes = np.unique(train_dataset.classes).astype(np.int64)
        y_train = train_dataset.classes.astype(np.int64)
        class_weights = compute_class_weight(
            'balanced',
            classes=classes,
            y=y_train
        )
        class_weights = dict(enumerate(class_weights))

        # Define callbacks for training.
        callbacks = [
            ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_auc', mode='max'),
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7),
            TensorBoard(log_dir='./logs', histogram_freq=1)
        ]

        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )

        self.logger.info("Training completed. Saving final model...")
        model.save('final_model.keras')
        self.evaluate_model(model, val_dataset)

    def evaluate_model(self, model, val_dataset):
        """Comprehensive evaluation on full validation set"""
        self.logger.info("Running full evaluation...")
        
        val_dataset.reset()
        y_true = []
        y_pred = []

        for i in range(len(val_dataset)):
            X_batch, y_batch = val_dataset[i]
            y_pred_batch = model.predict(X_batch, verbose=0)
            y_true.extend(np.argmax(y_batch, axis=1))
            y_pred.extend(np.argmax(y_pred_batch, axis=1))

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=list(val_dataset.class_indices.keys())))

def main():
    parser = argparse.ArgumentParser(description='Train a cancer detection model.')
    parser.add_argument('--dataset', type=str, required=True, help="Dataset type: 'mammogram' or 'brain_tumor'")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the dataset directory")
    parser.add_argument('--epochs', type=int, default=50, help="Number of epochs to train")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")

    args = parser.parse_args()

    trainer = CancerTrainer(args.dataset, args.data_dir)
    trainer.train(epochs=args.epochs, batch_size=args.batch_size)

if __name__ == "__main__":
    main()
