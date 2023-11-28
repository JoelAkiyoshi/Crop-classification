# construct and train the full-size CNN. Then use the resulting model to form the basis for the rf & svm pipelines.
# 3 main functions in total, each saving a model. Pipeline constructors will NOT call the first function.

from keras import Sequential
from keras.applications import vgg16
from keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping

from data.data_prep import DataHandler

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

from sklearn.svm import SVC

import numpy as np

# function: build a full-size CNN and train it (might need helper func), save the best result.


def _build_full_cnn_structure(dense_neurons: int, dropout_rate: float, is_trainable) -> Sequential:
    """Return the full vgg16 architecture, trained on imagenet weights. Model output is an array of shape (x, 4)."""
    size = 256
    color_channels = 3
    vgg = vgg16.VGG16(include_top=False, weights="imagenet", input_shape=(size, size, color_channels))

    vgg.trainable = is_trainable

    full_model = Sequential(
        [vgg,
         Flatten(),
         Dense(dense_neurons, activation="relu"),
         Dropout(dropout_rate),
         Dense(dense_neurons, activation="relu"),
         Dense(4, activation="softmax")]
    )

    return full_model


def create_full_cnn_model(epochs=10, batch_size=16, dense_neurons=128, dropout_rate=0.4, is_trainable=False):
    """Prepare train and test data, train the full_cnn."""
    model = _build_full_cnn_structure(dense_neurons, dropout_rate, is_trainable)

    train_dh = DataHandler(parent_directory="data/maize_split/train")
    train_images, train_labels = train_dh.get_images_and_labels()
    train_labels = train_dh.get_encoded_labels(train_labels)

    val_dh = DataHandler(parent_directory="data/maize_split/val")
    val_images, val_labels = val_dh.get_images_and_labels()
    val_labels = val_dh.get_encoded_labels(val_labels)

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    history = model.fit(train_images, train_labels,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(val_images, val_labels),
                        callbacks=es, verbose=1)

    return model
    # use GridSearchCV, possibly in conjunction with EarlyStopping, save best model for use with other funcs
    # this might not be necessary, because we only care about the scratch model and the trainable one


def _build_shallow_cnn() -> Sequential:
    """Return the SCNN model using the first six layers of the vgg16 architecture, trained on imagenet weights.
    Model output is a 128 dimensional feature vector."""

    # accessing the vgg16 architecture, using imagenet weights.
    # include_top refers to dense layers, input_shape is the image size (256x256 pixels, 3 color channels).
    size = 256
    color_channels = 3
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


def get_best_rf_params():  # temporary function, should implement pipeline later
    scnn = _build_shallow_cnn()
    print("scnn built")

    train_dh = DataHandler(parent_directory="data/maize_split/train")
    X_train, y_train = train_dh.get_images_and_labels()
    y_train = train_dh.get_encoded_labels(y_train)

    scnn_output = scnn.predict(X_train)
    print("scnn_output calculated (128 dim vectors)")

    pca = PCA(n_components=21)
    pca_output = pca.fit_transform(scnn_output)

    print("pca output", pca_output)

    param_grid = {
        'n_estimators': [50, 100, 200, 1000],
        'max_depth': [None, 10, 20, 30],
    }

    # Create a Random Forest Classifier
    rf = RandomForestClassifier()

    # Create a GridSearchCV object
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', verbose=3)

    # Fit the model with Grid Search
    print("searching...")
    grid_search.fit(pca_output, y_train)

    # Print the best hyperparameters and corresponding score
    print("Best hyperparameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)


def get_best_svm_params():  # temporary function, should implement pipeline later
    scnn = _build_shallow_cnn()
    print("scnn built")

    train_dh = DataHandler(parent_directory="data/maize_split/train")
    X_train, y_train = train_dh.get_images_and_labels()
    y_train = train_dh.get_encoded_labels(y_train)

    scnn_output = scnn.predict(X_train)
    print("scnn_output calculated (128 dim vectors)")

    pca = PCA(n_components=21)
    pca_output = pca.fit_transform(scnn_output)
    scale_gamma = 1 / (np.array(pca_output).var() * 21)
    print("scale gamma", scale_gamma)

    param_grid = {
        'C': [1, 10],
        'gamma': ["scale", "auto", scale_gamma]
    }

    # Create a Support Vector Machine Classifier
    svm = SVC()

    # Create a GridSearchCV object
    grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, scoring='accuracy', verbose=3)

    # Fit the model with Grid Search
    print("searching...")
    grid_search.fit(pca_output, y_train)

    # Print the best hyperparameters and corresponding score
    print("Best hyperparameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)


# function: using a specified (input) full-cnn model, build a sklearn pipeline for scnn-rf
# function: using a specified (input) full-cnn model, build a sklearn pipeline for scnn-svm


if __name__ == "__main__":
    # base_vgg_model = create_full_cnn_model()
    # get_best_rf_params()
    get_best_svm_params()

    # will likely run all three of the build functions (might as well). 3 new models will be added to models directory.
    # TODO: need to retrain the entire vgg16 model, potentially with all layers unfrozen, then try it. Save model.
    # TODO: need to form a pipeline.
