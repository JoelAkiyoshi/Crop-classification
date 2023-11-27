# does not access the data, just provides label_images to aid in extraction of the data

"""Prepare the x-train, x-test, y-train, y-test data."""
import glob
import numpy as np
from sklearn import preprocessing
from keras.utils import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input


class DataHandler:
    def __init__(self, parent_directory):
        self.parent_directory = parent_directory
        self.label_encoder = preprocessing.LabelEncoder()

    def get_images_and_labels(self) -> tuple:
        """Return an array of images, and an array of labels in a two-tuple.
        Input is a directory path containing the classification folders for
        either training/testing. Walks through the directory, prepares the images
        and stores them."""

        # labeling training data

        images = []
        labels = []

        # 'maize_split/train/*'
        # 'maize_split/val/*'
        for directory_path in glob.glob(self.parent_directory + '/*'):

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

    def get_encoded_labels(self, original_labels):
        """convert the labels to encoded values (0, 1, 2, 3)"""
        encoded_labels = self.label_encoder.fit_transform(original_labels)
        return encoded_labels

    def decode_labels(self, encoded_labels):
        """convert the labels to original values (Blight, Rust, Spot, Normal)"""
        original_labels = self.label_encoder.inverse_transform(encoded_labels)
        return original_labels
