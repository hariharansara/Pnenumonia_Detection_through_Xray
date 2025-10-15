import os
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import DataLoader # To get the test generator

# --- Configuration ---
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'cnn_xray_classifier.h5'))
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
EVAL_RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'evaluation_results'))

os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)

def evaluate_model():
    print("--- Starting Model Evaluation ---")
    print(f"Loading model from: {MODEL_PATH}")
    print(f"Loading data from: {DATA_ROOT}")

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}. Please train the model first using `python src/train.py`.")
        return

    # Load the trained model
    model = tf.keras.models.load_model(MODEL_PATH)

    # Load data for evaluation (only test set needed)
    data_loader = DataLoader(IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE, DATA_ROOT)
    try:
        # We only need the test generator here; augmentation should be False for evaluation
        _, _, test_generator = data_loader.create_generators(augmentation=False)
    except FileNotFoundError as e:
        print(f"Error loading test data: {e}")
        print("Please ensure your test data is correctly organized in 'data/raw/test/{class1,class2,...}'")
        return

    CLASS_NAMES = data_loader.class_names
    num_classes = len(CLASS_NAMES)

    print(f"\nEvaluating model on {test_generator.n} test images...")
    loss, accuracy, auc_metric = model.evaluate(test_generator, verbose=1)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test AUC: {auc_metric:.4f}")

    print("\n--- Generating detailed metrics and plots ---")

    # Get true labels and predictions
    test_generator.reset() # Reset generator to ensure order
    y_true = test_generator.classes
    y_pred_proba = model.predict(test_generator, verbose=1)

    if test_generator.class_mode == 'binary':
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        y_score = y_pred_proba.flatten()
    else: # 'categorical'
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_score = y_pred_proba # Keep probabilities for each class

    # Classification Report
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    with open(os.path.join(EVAL_RESULTS_DIR, 'classification_report.txt'), 'w') as f:
        f.write(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    # Confusion Matrix
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(EVAL_RESULTS_DIR, 'confusion_matrix.png'))
    plt.show()
    print(f"Confusion matrix saved to {os.path.join(EVAL_RESULTS_DIR, 'confusion_matrix.png')}")

    # ROC Curve and AUC (for binary classification)
    if test_generator.class_mode == 'binary':
        print(f"\n--- ROC Curve and AUC ({CLASS_NAMES[0]} vs {CLASS_NAMES[1]}) ---")
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(EVAL_RESULTS_DIR, 'roc_curve.png'))
        plt.show()
        print(f"ROC curve plot saved to {os.path.join(EVAL_RESULTS_DIR, 'roc_curve.png')}")
        print(f"ROC AUC Score: {roc_auc:.4f}")
    elif test_generator.class_mode == 'categorical':
        print("\n--- ROC Curve and AUC (Multi-class) ---")
        # For multi-class, you can plot one-vs-rest ROC curves
        from sklearn.preprocessing import LabelBinarizer
        lb = LabelBinarizer()
        y_true_binarized = lb.fit_transform(y_true)

        plt.figure(figsize=(10, 8))
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'ROC curve of class {CLASS_NAMES[i]} (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve for Multi-class')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(EVAL_RESULTS_DIR, 'roc_curve_multiclass.png'))
        plt.show()
        print(f"Multi-class ROC curve plot saved to {os.path.join(EVAL_RESULTS_DIR, 'roc_curve_multiclass.png')}")

    print("\n--- Evaluation Script Finished ---")


if __name__ == "__main__":
    evaluate_model()


