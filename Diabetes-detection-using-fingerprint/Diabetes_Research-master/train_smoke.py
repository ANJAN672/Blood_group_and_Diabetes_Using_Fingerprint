"""Smoke test: build the DocumentModel from the notebooks and train on synthetic data.
This verifies the model code runs with modern packages and without access to the original dataset.
"""
import os
import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv1D, MaxPooling1D, Dropout
from tensorflow.keras.models import Model


def DocumentModel(input_shape):
    X_input = Input(input_shape)

    X = Conv1D(32, 7, strides=1, name='conv1')(X_input)
    X = BatchNormalization(axis=2, name='bn1')(X)
    X = Activation('relu')(X)
    X = Dropout(0.2)(X)
    X = MaxPooling1D(2, name='max_pool1')(X)

    X = Conv1D(64, 5, strides=1, name='conv2')(X)
    X = BatchNormalization(axis=2, name='bn2')(X)
    X = Activation('relu')(X)
    X = Dropout(0.2)(X)
    X = MaxPooling1D(2, name='max_pool2')(X)

    X = Conv1D(128, 3, strides=1, name='conv3')(X)
    X = BatchNormalization(axis=2, name='bn3')(X)
    X = Activation('relu')(X)
    X = Dropout(0.2)(X)
    X = MaxPooling1D(2, name='max_pool3')(X)

    X = Conv1D(64, 1, strides=1, name='conv4')(X)
    X = BatchNormalization(axis=2, name='bn4')(X)
    X = Activation('relu')(X)
    X = Dropout(0.2)(X)

    X = Conv1D(32, 3, strides=1, name='conv5')(X)
    X = BatchNormalization(axis=2, name='bn5')(X)
    X = Activation('relu')(X)
    X = Dropout(0.2)(X)
    X = MaxPooling1D(2, name='max_pool5')(X)

    X = Flatten()(X)
    X = Dense(128, activation='sigmoid', name='fc1')(X)
    X = Dense(32, activation='sigmoid', name='fc2')(X)
    X = Dense(2, activation='sigmoid', name='fc3')(X)

    model = Model(inputs=X_input, outputs=X, name='DocumentModel')
    return model


def load_images_from_folder(root_dir, target_size=(200, 250), max_images=None):
    """Load images from a root directory. If subfolders are present, each subfolder is a class.
    Returns (X, y, class_names). X has shape (N, target_size[0], target_size[1]).
    """
    images = []
    labels = []
    class_names = []
    # detect class subfolders
    entries = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    if entries:
        class_names = sorted(entries)
        for idx, cls in enumerate(class_names):
            cls_dir = os.path.join(root_dir, cls)
            for root, _, files in os.walk(cls_dir):
                for fname in files:
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                        path = os.path.join(root, fname)
                        try:
                            img = Image.open(path).convert('L')
                            img = img.resize((target_size[1], target_size[0]), Image.Resampling.LANCZOS)
                            arr = np.asarray(img, dtype=np.float32) / 255.0
                            images.append(arr)
                            labels.append(idx)
                        except Exception:
                            continue
                        if max_images and len(images) >= max_images:
                            break
                if max_images and len(images) >= max_images:
                    break
            if max_images and len(images) >= max_images:
                break
    else:
        # no subfolders, load images from root as single-class
        class_names = ['all']
        for root, _, files in os.walk(root_dir):
            for fname in files:
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                    path = os.path.join(root, fname)
                    try:
                        img = Image.open(path).convert('L')
                        img = img.resize((target_size[1], target_size[0]), Image.Resampling.LANCZOS)
                        arr = np.asarray(img, dtype=np.float32) / 255.0
                        images.append(arr)
                        labels.append(0)
                    except Exception:
                        continue
                    if max_images and len(images) >= max_images:
                        break
            if max_images and len(images) >= max_images:
                break

    if not images:
        return None, None, None

    X = np.stack(images, axis=0)
    y = np.array(labels, dtype=np.int32)
    return X, y, class_names


def make_synthetic_dataset(samples=40, height=200, width=250):
    X = np.random.rand(samples, height, width).astype(np.float32)
    y = np.random.randint(0, 2, size=(samples,))
    return X, y


def main():
    # dataset path can be provided via env var or CLI argument. Default: './dataset'
    data_path = os.environ.get('DIABETES_DATA_PATH', None)
    if data_path is None:
        data_path = os.path.join(os.path.dirname(__file__), 'dataset')

    if os.path.exists(data_path):
        print('Found dataset at', data_path, ' — attempting to load images')
        X, y, classes = load_images_from_folder(data_path, target_size=(200, 250), max_images=2000)
        if X is None:
            print('No images found in dataset path, falling back to synthetic data')
            X, y = make_synthetic_dataset()
        else:
            print('Loaded dataset:', X.shape, 'classes:', classes)
    else:
        print('Dataset path not found:', data_path)
        print('Falling back to synthetic data')
        X, y = make_synthetic_dataset()

    # DocumentModel expects input shape (steps, features) i.e., (height, width)
    input_shape = (X.shape[1], X.shape[2])
    print('Building model with input shape', input_shape)
    model = DocumentModel(input_shape)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print('Dataset shapes:', X.shape, y.shape)
    # quick train
    model.fit(X, y, epochs=2, batch_size=8)

    preds = model.predict(X[:8])
    print('Preds shape:', preds.shape)


if __name__ == '__main__':
    main()
