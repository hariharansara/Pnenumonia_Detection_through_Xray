import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

def build_cnn_model(input_shape, num_classes):
    """
    Builds a Convolutional Neural Network (CNN) model.

    Args:
        input_shape (tuple): Shape of the input images (height, width, channels).
        num_classes (int): Number of classes for classification.
                           Use 1 for binary classification (sigmoid activation),
                           >1 for multi-class classification (softmax activation).

    Returns:
        tf.keras.Model: Compiled Keras CNN model.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(), # Added BatchNormalization
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(), # Added BatchNormalization
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(), # Added BatchNormalization
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='sigmoid' if num_classes == 1 else 'softmax')
    ])

    # For binary classification, use binary_crossentropy
    # For multi-class classification, use categorical_crossentropy
    loss_function = 'binary_crossentropy' if num_classes == 1 else 'categorical_crossentropy'
    metrics = ['accuracy', tf.keras.metrics.AUC(name='auc')] # AUC is very useful, especially for imbalanced datasets

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss=loss_function,
                  metrics=metrics)

    return model

if __name__ == "__main__":
    # Example usage for model building
    print("Running model.py test block...")
    IMG_HEIGHT = 150
    IMG_WIDTH = 150
    INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3) # 3 for RGB images (X-rays are often grayscale, but DataGenerator defaults to 3 channels)
    NUM_CLASSES = 1 # Example: Binary classification (e.g., normal vs. condition)

    model = build_cnn_model(INPUT_SHAPE, NUM_CLASSES)
    model.summary()

    # Example for multi-class
    # NUM_CLASSES_MULTICLASS = 3
    # model_multi = build_cnn_model(INPUT_SHAPE, NUM_CLASSES_MULTICLASS)
    # model_multi.summary()