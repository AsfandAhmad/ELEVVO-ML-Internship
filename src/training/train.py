import os
import argparse
import datetime
import tensorflow as tf
from tensorflow.keras import optimizers, callbacks
from sklearn.metrics import accuracy_score
from src.utils.dataset import build_datasets_from_directory
from src.utils.metrics import plot_confusion_matrix, save_classification_report
from src.utils.plots import plot_history
from src.models.custom_cnn import build_custom_cnn
from src.models.mobilenet import build_mobilenet_v2

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed", help="Processed data root (with train/val/test)")
    ap.add_argument("--model", choices=["custom", "mobilenet"], default="custom")
    ap.add_argument("--img_size", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=1e-3)
    ap.add_argument("--freeze_base", type=lambda x: x.lower() in ['true','1','yes'], default=True)
    return ap.parse_args()

def main():
    args = parse_args()

    # Make output folders
    os.makedirs("outputs/checkpoints", exist_ok=True)
    os.makedirs("outputs/history", exist_ok=True)
    os.makedirs("outputs/confusion_matrices", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)

    # Load datasets
    train_ds, val_ds, test_ds, class_names = build_datasets_from_directory(
        args.data_dir, args.img_size, args.batch_size, augment=True
    )
    num_classes = len(class_names)
    print(f"[INFO] Number of classes detected: {num_classes}")
    input_shape = (args.img_size, args.img_size, 3)

    # Build model
    if args.model == "custom":
        model = build_custom_cnn(input_shape, num_classes)
    else:
        model = build_mobilenet_v2(input_shape, num_classes, freeze_base=args.freeze_base)

    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=args.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Setup callbacks
    run_tag = f"{args.model}_sz{args.img_size}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    ckpt_path = f"outputs/checkpoints/best_model.keras"
    cbs = [
        callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", mode="max", save_best_only=True, verbose=1),
        callbacks.EarlyStopping(patience=7, restore_best_weights=True, monitor="val_accuracy", mode="max"),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6, verbose=1, monitor="val_loss"),
        callbacks.TensorBoard(log_dir=os.path.join("outputs/logs", run_tag))
    ]

    # Train
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=cbs)

    # Plot training history
    plot_history(history, f"outputs/history/{run_tag}.png")

    # Evaluation helper
    def eval_and_report(ds, split_name):
        if ds is None:
            return
        y_true, y_pred = [], []
        for xb, yb in ds:
            probs = model.predict(xb, verbose=0)
            y_true.extend(list(yb.numpy()))
            y_pred.extend(list(probs.argmax(axis=1)))
        acc = accuracy_score(y_true, y_pred)
        print(f"{split_name} accuracy: {acc:.4f}")
        plot_confusion_matrix(y_true, y_pred, labels=list(range(num_classes)),
                              out_path=f"outputs/confusion_matrices/{run_tag}_{split_name}.png",
                              normalize=True)
        save_classification_report(y_true, y_pred, labels=list(range(num_classes)),
                                  out_path=f"outputs/reports/{run_tag}_{split_name}.txt")

    # Evaluate
    eval_and_report(val_ds, "val")
    eval_and_report(test_ds, "test")

    # Save final model
    model.save(f"outputs/checkpoints/{run_tag}.keras")
    print("Training complete. Best model saved to:", ckpt_path)

if __name__ == "__main__":
    main()
