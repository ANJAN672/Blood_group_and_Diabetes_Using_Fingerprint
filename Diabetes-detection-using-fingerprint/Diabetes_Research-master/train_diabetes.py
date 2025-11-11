"""Train a diabetes-vs-healthy classifier on fingerprint images.

Expect dataset directory structure:
  dataset/
    diabetic/
      img1.png
      ...
    non_diabetic/
      imgA.png
      ...

The script will save a Keras model and a class mapping under ./models.

Usage examples:
  # dry-run (no dataset required) - builds the model and trains on synthetic data
  python train_diabetes.py --dry-run

  # train using dataset directory
  set DIABETES_DATA_PATH=F:\\path\\to\\dataset
  python train_diabetes.py --epochs 10 --batch-size 16

"""
import os
import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


def build_model(input_shape=(224,224,3), dropout=0.3):
    base = MobileNetV2(include_top=False, input_shape=input_shape, weights='imagenet')
    base.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs=base.input, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def get_datasets(data_dir, img_size=(224,224), batch_size=16, val_split=0.2, seed=1337):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='binary',
        image_size=img_size,
        batch_size=batch_size,
        validation_split=val_split,
        subset='training',
        seed=seed,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='binary',
        image_size=img_size,
        batch_size=batch_size,
        validation_split=val_split,
        subset='validation',
        seed=seed,
    )
    class_names = train_ds.class_names
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds, class_names


def synthetic_dataset(samples=80, img_size=(224,224,3), batch_size=16):
    # Create small synthetic tf.data datasets for dry-run
    X = np.random.rand(samples, *img_size).astype('float32')
    y = np.random.randint(0, 2, size=(samples,)).astype('float32')
    ds = tf.data.Dataset.from_tensor_slices((X, y)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    # split
    split = int(samples * 0.8)
    train = tf.data.Dataset.from_tensor_slices((X[:split], y[:split])).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val = tf.data.Dataset.from_tensor_slices((X[split:], y[split:])).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train, val, ['non_diabetic', 'diabetic']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default=None, help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--dry-run', action='store_true', help='Run using synthetic data')
    parser.add_argument('--output-dir', type=str, default='models')
    args = parser.parse_args()

    data_dir = args.data_dir or os.environ.get('DIABETES_DATA_PATH')

    if args.dry_run or not data_dir or not os.path.exists(data_dir):
        print('Running dry-run training with synthetic data.')
        train_ds, val_ds, class_names = synthetic_dataset(samples=160, img_size=(224,224,3), batch_size=args.batch_size)
    else:
        print('Loading dataset from', data_dir)
        train_ds, val_ds, class_names = get_datasets(data_dir, img_size=(224,224), batch_size=args.batch_size)

    print('Class names:', class_names)
    model = build_model(input_shape=(224,224,3))
    model.summary()

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, 'diabetes_model.h5')

    callbacks = [
        ModelCheckpoint(ckpt_path, save_best_only=True, monitor='val_loss'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    # save class mapping
    with open(os.path.join(args.output_dir, 'class_names.json'), 'w') as f:
        json.dump(class_names, f)

    print('Training finished. Model saved to', ckpt_path)


if __name__ == '__main__':
    main()
