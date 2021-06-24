export const IMDBBinaryClassification = `from tensorflow.keras.datasets import imdb
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

# Data preprocessing (expects the outputs: partial_x_train, x_val, x_test, partial_y_train, y_val, y_test)
# Example (courtosey to François Chollet):
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
y_train = np.asarray(train_labels).astype("float32")
y_test = np.asarray(test_labels).astype("float32")
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# Building and compiling the model
callbacks_list = [
    keras.callbacks.EarlyStopping(
        monitor="val_binary_accuracy",
        patience=3,
    ),
    keras.callbacks.ModelCheckpoint(
        filepath="saved_model",
        monitor="val_loss",
        save_best_only=True,
    )
]
model = keras.Sequential([
    keras.layers.Dense(8, activation="relu"), 
    keras.layers.Dense(16, activation="relu"), 
    keras.layers.Dense(8, activation="relu"), 
    keras.layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer=keras.optimizers.RMSprop(0.01),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[keras.metrics.BinaryAccuracy()])

# Training the model
history = model.fit(partial_x_train,
    partial_y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_val, y_val),
    callbacks=callbacks_list)

# Optionally load the model
model = keras.models.load_model("saved_model")

# Evaluate
results = model.evaluate(x_test, y_test)

# Plot the training and validation loss
history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, "bo", label="Training loss")
plt.plot(epochs, val_loss_values, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Plot the training and validation accuracy
plt.clf()
acc = history_dict["binary_accuracy"]
val_acc = history_dict["val_binary_accuracy"]
plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Train on colab: https://research.google.com/colaboratory/
`


export const MNISTCategoricalClassification = `from tensorflow import keras
from tensorflow.keras import layers
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# Data preprocessing (expects the outputs: partial_x_train, x_val, x_test, partial_y_train, y_val, y_test)
# Example (courtesy to François Chollet):
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((60000, 28 * 28))
x_train = x_train.astype("float32") / 255
x_test = x_test.reshape((10000, 28 * 28))
x_test = x_test.astype("float32") / 255
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# Building and compiling the model
callbacks_list = [
    keras.callbacks.EarlyStopping(
        monitor="val_sparse_categorical_accuracy",
        patience=3,
    ),
    keras.callbacks.ModelCheckpoint(
        filepath="saved_model",
        monitor="val_loss",
        save_best_only=True,
    )
]
model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
])
model.compile(optimizer=keras.optimizers.RMSprop(0.01),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()])

# Training the model
history = model.fit(partial_x_train,
    partial_y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_val, y_val),
    callbacks=callbacks_list)

# Optionally load the model
model = keras.models.load_model("saved_model")

# Evaluate
results = model.evaluate(x_test, y_test)

# Plot the training and validation loss
history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, "bo", label="Training loss")
plt.plot(epochs, val_loss_values, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Plot the training and validation accuracy
plt.clf()
acc = history_dict["sparse_categorical_accuracy"]
val_acc = history_dict["val_sparse_categorical_accuracy"]
plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Train on colab: https://research.google.com/colaboratory/
`

export const MNISTCategoricalClassificationWithCNN = `from tensorflow import keras
from tensorflow.keras import layers
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# Data preprocessing (expects the outputs: partial_x_train, x_val, x_test, partial_y_train, y_val, y_test)
# Example (courtesy to François Chollet):
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# change x_train.shape from (60000, 28, 28) to (60000, 28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# Building and compiling the model
callbacks_list = [
    keras.callbacks.EarlyStopping(
        monitor="val_sparse_categorical_accuracy",
        patience=3,
    ),
    keras.callbacks.ModelCheckpoint(
        filepath="saved_model",
        monitor="val_loss",
        save_best_only=True,
    )
]
model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(filters=32, kernel_size=3, activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(filters=64, kernel_size=3, activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(filters=128, kernel_size=3, activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(10, activation="softmax")
])

model.compile(optimizer=keras.optimizers.RMSprop(0.01),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()])

# Training the model
history = model.fit(partial_x_train,
    partial_y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_val, y_val),
    callbacks=callbacks_list)

# Optionally load the model
model = keras.models.load_model("saved_model")

# Evaluate
results = model.evaluate(x_test, y_test)

# Plot the training and validation loss
history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, "bo", label="Training loss")
plt.plot(epochs, val_loss_values, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Plot the training and validation accuracy
plt.clf()
acc = history_dict["sparse_categorical_accuracy"]
val_acc = history_dict["val_sparse_categorical_accuracy"]
plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Train on colab: https://research.google.com/colaboratory/
`

export const BostonHousingRegression = `from tensorflow.keras.datasets import boston_housing
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Data preprocessing (expects the outputs: partial_x_train, x_val, x_test, partial_y_train, y_val, y_test)
# Example (courtesy to François Chollet):
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
mean = x_train.mean(axis=0)
x_train -= mean
std = x_train.std(axis=0)
x_train /= std
x_val = x_train[:50]
partial_x_train = x_train[50:]
y_val = y_train[:50]
partial_y_train = y_train[50:]
x_test -= mean
x_test /= std

# Building and compiling the model
callbacks_list = [
    keras.callbacks.EarlyStopping(
        monitor="val_mean_absolute_error",
        patience=3,
    ),
    keras.callbacks.ModelCheckpoint(
        filepath="saved_model",
        monitor="val_loss",
        save_best_only=True,
    )
]
model = keras.Sequential([
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(1)
    ])

model.compile(optimizer=keras.optimizers.RMSprop(0.01),
    loss=keras.losses.MeanSquaredError(),
    metrics=[keras.metrics.MeanAbsoluteError()])

# Training the model
history = model.fit(partial_x_train,
    partial_y_train,
    epochs=400,
    batch_size=1,
    validation_data=(x_val, y_val),
    callbacks=callbacks_list)

# Optionally retrain the model with all the training data till the alleged epoch that has the least val_mean_absolute_error. It would make more sense if we have a lot of data.
# history = model.fit(x_train,
#     y_train,
#     epochs=6,
#     batch_size=1)

# Optionally load the model
model = keras.models.load_model("saved_model")

# Evaluate and predict
test_mse_score, test_mae_score = model.evaluate(x_test, y_test)
predictions = model.predict(x_test)
print(test_mse_score, test_mae_score, predictions[0])

# Plot the training and validation loss
history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, "bo", label="Training loss")
plt.plot(epochs, val_loss_values, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Plot the training and validation accuracy
plt.clf()
mae = history_dict["mean_absolute_error"]
val_mae = history_dict["val_mean_absolute_error"]
plt.plot(epochs, mae, "bo", label="Training error")
plt.plot(epochs, val_mae, "b", label="Validation error")
plt.title("Training and validation mae")
plt.xlabel("Epochs")
plt.ylabel("Mean Absolute Error")
plt.legend()
plt.show()

# Train on colab: https://research.google.com/colaboratory/
`