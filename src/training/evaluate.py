import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

from src.utils.dataset import build_datasets_from_directory

def main():
    # Argument parser
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=64)
    args = p.parse_args()

    # Load model
    print("Loading model:", args.model_path)
    model = tf.keras.models.load_model(args.model_path, compile=False)

    # Load datasets
    train_ds, val_ds, test_ds, class_names = build_datasets_from_directory(
        args.data_dir, args.img_size, args.batch_size, augment=False
    )

    # Optional: Compute class weights (not used in evaluation, for reference)
    labels = []
    for idx, cls in enumerate(class_names):
        cls_path = os.path.join(args.data_dir, "train", cls)
        labels += [idx] * len(os.listdir(cls_path))
    labels = np.array(labels)

    class_weights = compute_class_weight(class_weight='balanced', classes=np.arange(len(class_names)), y=labels)
    class_weights = dict(enumerate(class_weights))
    print("Class weights (reference):", class_weights)

    # Choose dataset to evaluate
    ds = test_ds if test_ds is not None else val_ds

    # Make predictions
    y_true, y_pred = [], []
    for X, y in ds:
        probs = model.predict(X, verbose=0)
        y_true.extend(y.numpy())
        y_pred.extend(np.argmax(probs, axis=1))

    # Classification report
    print("\nClassification report:\n")
    print(classification_report(y_true, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap="Blues", xticklabels=False, yticklabels=False)
    plt.title("Confusion Matrix")
    os.makedirs("outputs/confusion_matrices", exist_ok=True)
    out = f"outputs/confusion_matrices/{os.path.basename(args.model_path)}.png"
    plt.savefig(out)
    print("Saved confusion matrix to", out)

if __name__ == "__main__":
    main()
