# construct and train the full-size CNN. Then use the resulting model to form the basis for the rf & svm pipelines.
# 3 main functions in total, each saving a model. Pipeline constructors will NOT call the first function.

from keras import Sequential
from keras.applications import vgg16
from keras.layers import Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping

from data.data_prep import DataHandler


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
                        callbacks=es, verbose=2)

    return model
    # use GridSearchCV, possibly in conjunction with EarlyStopping, save best model for use with other funcs


# function: using a specified (input) full-cnn model, build a sklearn pipeline for scnn-rf
# function: using a specified (input) full-cnn model, build a sklearn pipeline for scnn-svm


if __name__ == "__main__":
    pass
    # will likely run all three of the build functions (might as well). 3 new models will be added to models directory.
