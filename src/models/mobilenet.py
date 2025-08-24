import tensorflow as tf
from tensorflow.keras import layers, models

def build_mobilenet_v2(input_shape, num_classes, freeze_base=True):
    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights='imagenet'
    )
    if freeze_base:
        base.trainable = False
    else:
        base.trainable = True

    inputs = layers.Input(shape=input_shape)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=not freeze_base)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, outputs, name="mobilenet_v2")
    return model
