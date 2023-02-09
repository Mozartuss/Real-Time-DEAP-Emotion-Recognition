from pathlib import Path

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from Utils.Constants import ROOT_PATH


def plot_confusion_matrix(x_val, y_val, model_name):
    y_pred_classes = np.argmax(x_val, axis=1)
    y_test_classes = np.argmax(y_val, axis=1)
    cm = confusion_matrix(y_test_classes, y_pred_classes)
    ax = sns.heatmap(cm / np.sum(cm), annot=True, fmt='.2%', cmap='Blues')
    ax.set_title('Confusion Matrix\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values\n')
    # Ticket labels - List must be in order
    ax.xaxis.set_ticklabels(['HAHV', 'HALV', 'LAHV', 'LALV'])
    ax.yaxis.set_ticklabels(['HAHV', 'HALV', 'LAHV', 'LALV'])
    plt.savefig(Path(ROOT_PATH, f"Real_Time_Model_{model_name}", "Confusion_Matrix.svg"), format="svg")
    plt.clf()


def plot_train_history(history, model_name, accuracy, loss):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Accuracy: {:.2%}, Loss {}.".format(accuracy, loss))
    ax1.plot(history.history['accuracy'], label='Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Accuracy and Validation Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax2.plot(history.history['loss'], label='Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Loss and Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(Path(ROOT_PATH, f"Real_Time_Model_{model_name}", "graph.svg"), format="svg")
    plt.clf()
