import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from src.data_loader import DataLoader
from src.model import build_cnn_model
import matplotlib.pyplot as plt # For plotting training history
import numpy as np # For numpy operations

# --- Configuration ---
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32
EPOCHS = 50 # Adjust as needed. Early stopping will likely stop it sooner.
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
MODEL_FILENAME = os.path.join(MODELS_DIR, 'cnn_xray_classifier.keras')
HISTORY_PLOT_FILENAME = os.path.join(MODELS_DIR, 'training_history.png')


os.makedirs(MODELS_DIR, exist_ok=True)

def plot_training_history(history, filename):
    """
    Plots training and validation accuracy and loss.
    """
    plt.figure(figsize=(12, 5))

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Training history plot saved to {filename}")
    # plt.show() # Uncomment to display plot during execution

def train_model():
    print("--- Starting Model Training ---")
    print(f"Data root: {DATA_ROOT}")
    print(f"Models directory: {MODELS_DIR}")

    # 1. Load and Preprocess Data
    data_loader = DataLoader(IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE, DATA_ROOT)
    try:
        train_generator, val_generator, test_generator = data_loader.create_generators(augmentation=True)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please ensure your data is correctly organized in 'data/raw/{train,val,test}/{class1,class2,...}'")
        return

    # Determine number of classes
    num_classes = len(data_loader.class_names)
    if train_generator.class_mode == 'categorical': # ImageDataGenerator outputs one-hot encoded for 'categorical'
        num_classes_model = num_classes
    else: # Binary classification ('binary' class_mode), ImageDataGenerator outputs scalar labels
        num_classes_model = 1

    input_shape = (IMG_HEIGHT, IMG_WIDTH, 3) # Assuming RGB images

    # 2. Build CNN Model
    model = build_cnn_model(input_shape, num_classes_model)
    print("\n--- Model Summary ---")
    model.summary()

    # 3. Define Callbacks
    checkpoint = ModelCheckpoint(
        MODEL_FILENAME,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10, # Number of epochs with no improvement after which training will be stopped.
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2, # Factor by which the learning rate will be reduced. new_lr = lr * factor
        patience=5, # Number of epochs with no improvement after which learning rate will be reduced.
        min_lr=0.00001,
        verbose=1
    )

    callbacks = [checkpoint, early_stopping, reduce_lr]

    # 4. Train Model
    print("\n--- Training Model ---")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks
    )

    print("\n--- Training Finished ---")
    print(f"Best model saved to {MODEL_FILENAME}")

    # Plot training history
    plot_training_history(history, HISTORY_PLOT_FILENAME)

    # 5. Evaluate Model on Test Set
    print("\n--- Evaluating Model on Test Set ---")
    # Load the best model found during training for final evaluation
    if os.path.exists(MODEL_FILENAME):
        best_model = tf.keras.models.load_model(MODEL_FILENAME)
        test_loss, test_accuracy, test_auc = best_model.evaluate(test_generator, verbose=1)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test AUC: {test_auc:.4f}")
    else:
        print("No best model saved to evaluate. Training might have failed or been interrupted.")


    print("\n--- Training Script Finished ---")

if __name__ == "__main__":
    train_model()