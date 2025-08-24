import tensorflow as tf
import os

AUTOTUNE = tf.data.AUTOTUNE

def build_datasets_from_directory(data_dir, img_size, batch_size, augment=True, shuffle=True):
    train_dir = os.path.join(data_dir, "train")
    val_dir   = os.path.join(data_dir, "val")
    test_dir  = os.path.join(data_dir, "test")

    # Load datasets first
    raw_train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=(img_size, img_size), batch_size=batch_size, shuffle=shuffle
    )
    raw_val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir, image_size=(img_size, img_size), batch_size=batch_size, shuffle=False
    )
    raw_test_ds = None
    if os.path.isdir(test_dir) and len(os.listdir(test_dir)) > 0:
        raw_test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir, image_size=(img_size, img_size), batch_size=batch_size, shuffle=False
        )

    # Capture class_names BEFORE mapping
    class_names = raw_train_ds.class_names
    print(f"[INFO] Number of classes detected: {len(class_names)}")
    print(f"[INFO] Classes: {class_names}")

    # Normalization layer
    norm = tf.keras.layers.Rescaling(1./255)

    # Optional augmentation
    aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ]) if augment else None

    def prep(ds, training=False):
        ds = ds.map(lambda x, y: (norm(x), y), num_parallel_calls=AUTOTUNE)
        if training and aug is not None:
            ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
        return ds.cache().prefetch(AUTOTUNE)

    train_ds = prep(raw_train_ds, training=True)
    val_ds   = prep(raw_val_ds, training=False)
    test_ds  = prep(raw_test_ds, training=False) if raw_test_ds is not None else None

    return train_ds, val_ds, test_ds, class_names
