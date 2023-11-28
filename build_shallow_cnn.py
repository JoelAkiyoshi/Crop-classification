from keras import Sequential
from keras.layers import GlobalAveragePooling2D
from keras.applications import vgg16
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

import numpy as np

from data.data_prep import DataHandler

import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt


def _build_shallow_cnn_vgg(size=256, color_channels=3) -> Sequential:
    """Return the SCNN model using the first six layers of the vgg16 architecture, trained on imagenet weights.
    Model output is a 128 dimensional feature vector."""

    # accessing the vgg16 architecture, using imagenet weights.
    # include_top refers to dense layers, input_shape is the image size (256x256 pixels, 3 color channels).

    vgg = vgg16.VGG16(include_top=False, weights="imagenet", input_shape=(size, size, color_channels))

    # "chopping" the model's input layer, conv1-1, conv1-2, (max)pool, conv2-1, conv2-2
    shallow_cnn_layers = vgg.layers[:6]

    # global average pool to be added
    global_average_pool = GlobalAveragePooling2D()

    # adding a global avg pooling layer so the model outputs 128-dimensional feature vector
    shallow_cnn_layers.append(global_average_pool)

    # constructing the Sequential model from the layers.
    shallow_model = Sequential(shallow_cnn_layers)

    # model weights should be frozen.
    shallow_model.trainable = False

    return shallow_model


def build_svm_pipeline(svc_c, svc_gamma) -> Pipeline:
    scnn = KerasClassifier(_build_shallow_cnn_vgg())

    pipe = Pipeline([
        ("scnn", scnn),
        ("pca", PCA(n_components=21)),
        ("svc", SVC(C=svc_c, gamma=svc_gamma))
    ])
    return pipe



def get_best_rf_params():  # temporary function, should implement pipeline later
    scnn = _build_shallow_cnn_vgg()

    train_dh = DataHandler(parent_directory="data/maize_split/train")
    X_train, y_train = train_dh.get_images_and_labels()
    y_train = train_dh.get_encoded_labels(y_train)

    scnn_output = scnn.predict(X_train)

    pca = PCA(n_components=21)
    pca_output = pca.fit_transform(scnn_output)

    param_grid = {
        'n_estimators': [50, 100, 200, 1000],
        'max_depth': [None, 10, 20, 30],
        'max_features': ['auto', 'sqrt', 'log2', None]
    }

    # Create a Random Forest Classifier
    rf = RandomForestClassifier()

    # Create a GridSearchCV object
    grid_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, cv=5, scoring='accuracy', verbose=3)

    # Fit the model with Grid Search
    print("searching...")
    grid_search.fit(pca_output, y_train)

    # Print the best hyperparameters and corresponding score
    print("Best hyperparameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)

    return grid_search.best_estimator_


def get_best_svm_params():  # temporary function, should implement pipeline later
    scnn = _build_shallow_cnn_vgg()
    print("scnn built")

    train_dh = DataHandler(parent_directory="data/maize_split/train")
    X_train, y_train = train_dh.get_images_and_labels()
    y_train = train_dh.get_encoded_labels(y_train)

    scnn_output = scnn.predict(X_train)
    print("scnn_output calculated (128 dim vectors)")

    pca = PCA(n_components=21)
    pca_output = pca.fit_transform(scnn_output)

    param_grid = {
        'C': [9, 10, 11],
        'gamma': ["scale", 0.000001]
    }

    # Create a Support Vector Machine Classifier
    svm = SVC()

    # Create a RandomizedSearchCV object
    grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, scoring='accuracy', verbose=3)

    # Fit the model with Grid Search
    print("searching...")
    grid_search.fit(pca_output, y_train)

    # Print the best hyperparameters and corresponding score
    print("Best hyperparameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)

    return grid_search.best_estimator_




if __name__ == "__main__":
    best_rf = get_best_rf_params()

    scnn = _build_shallow_cnn_vgg()

    test_dh = DataHandler(parent_directory="data/maize_split/test")
    X_test, y_test = test_dh.get_images_and_labels()
    y_test = test_dh.get_encoded_labels(y_test)

    scnn_output = scnn.predict(X_test)

    pca = PCA(n_components=21)
    pca_output = pca.fit_transform(scnn_output)

    predictions = best_rf.predict(pca_output)
    print(predictions)

    print(y_test)

    cm = metrics.confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True)
    plt.show()

# Issue may be in encoding of the values. Massive overfitting.


