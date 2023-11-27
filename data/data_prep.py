# does not access the data, just provides label_images to aid in extraction of the data

"""Prepare the x-train, x-test, y-train, y-test data."""
import glob
import numpy as np
from sklearn import preprocessing
from keras.utils import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input


def label_images(parent_directory: str) -> tuple:
    """Return an array of images, and an array of labels in a two-tuple.
    Input is a directory path containing the classification folders for
    either training/testing. Walks through the directory, prepares the images
    and stores them."""

    # labeling training data

    images = []
    labels = []

    # 'maize_split/train/*'
    # 'maize_split/val/*'
    for directory_path in glob.glob(parent_directory + '/*'):

        # get the disease name
        classification = directory_path.split('\\')[-1]

        for image_path in glob.glob(directory_path + '/*'):
            img = load_img(image_path, target_size=(256, 256))
            arrayed_img = img_to_array(img)
            proc_img = preprocess_input(arrayed_img)

            labels.append(classification)
            images.append(proc_img)

    # convert to numpy array
    images = np.array(images)
    labels = np.array(labels)

    return images, labels


def encode_labels(labels):
    """convert the labels to encoded values instead of str"""
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(labels)
    labels_encoded = label_encoder.transform(labels)
    return labels_encoded


def decode_labels(labels: int):
    pass
