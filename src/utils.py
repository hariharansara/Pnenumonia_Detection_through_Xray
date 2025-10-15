import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_model_history(history, save_dir=None):
    """
    Plots training & validation accuracy and loss from a Keras history object.

    Args:
        history (keras.callbacks.History): The history object returned by model.fit().
        save_dir (str, optional): Directory to save the plots. If None, plots are displayed.
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
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'training_history.png'))
        print(f"Training history plot saved to {os.path.join(save_dir, 'training_history.png')}")
    # plt.show() # Uncomment to display plots during execution

def plot_confusion_matrix(cm, class_names, title='Confusion Matrix', save_dir=None):
    """
    Plots a confusion matrix using seaborn.

    Args:
        cm (np.array): The confusion matrix.
        class_names (list): List of class names.
        title (str): Title of the plot.
        save_dir (str, optional): Directory to save the plot. If None, plot is displayed.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
        print(f"Confusion matrix plot saved to {os.path.join(save_dir, 'confusion_matrix.png')}")
    # plt.show() # Uncomment to display plots during execution

def plot_roc_curve(fpr, tpr, roc_auc, class_name=None, title='ROC Curve', save_dir=None):
    """
    Plots an ROC curve.

    Args:
        fpr (np.array): False Positive Rate.
        tpr (np.array): True Positive Rate.
        roc_auc (float): Area Under the Curve (AUC).
        class_name (str, optional): Name of the class for binary ROC.
        title (str): Title of the plot.
        save_dir (str, optional): Directory to save the plot. If None, plot is displayed.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plot_title = f'{title} - {class_name}' if class_name else title
    plt.title(plot_title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"roc_curve_{class_name.lower().replace(' ', '_')}.png" if class_name else "roc_curve.png"
        plt.savefig(os.path.join(save_dir, filename))
        print(f"ROC curve plot saved to {os.path.join(save_dir, filename)}")
    # plt.show() # Uncomment to display plots during execution

if __name__ == "__main__":
    print("This is a utility file. It contains helper functions.")
    print("Run `src/train.py` or `src/evaluate.py` to see these utilities in action.")

    # Example of how you might use plot_model_history if you had a history object
    # from tensorflow.keras.callbacks import History
    # dummy_history = History()
    # dummy_history.history = {
    #     'accuracy': [0.7, 0.75, 0.8, 0.82],
    #     'val_accuracy': [0.65, 0.7, 0.78, 0.80],
    #     'loss': [0.5, 0.45, 0.4, 0.38],
    #     'val_loss': [0.55, 0.5, 0.42, 0.40]
    # }
    # plot_model_history(dummy_history, save_dir='../models/evaluation_results')