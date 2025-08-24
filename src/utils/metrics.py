import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

def plot_confusion_matrix(y_true, y_pred, labels, out_path, normalize=True):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)
    fig = plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

def save_classification_report(y_true, y_pred, labels, out_path):
    report = classification_report(y_true, y_pred, labels=labels, digits=4)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    return report
