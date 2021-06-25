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

export const MNISTCategoricalClassificationWithTransferLearningAndFineTuning = `from tensorflow import keras
from tensorflow.keras import layers
from keras.datasets import mnist
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.transform import resize

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Plot some images
for i in range(9):
  plt.subplot(3, 3, i+1)
  plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))
  plt.axis("off")

# VGG takes (32, 32, 3) as the min height & width. MNIST is only 28 x 28. Hence, we need to resize the images
# We will preserve the range 0-255 here and use VGG 16's preprocessing function to preprocess it later. You 
# can pick one of the three preprocessing methods (preprocess 0, 1 or 2)
# VGG needs 3-channel images as input data
resized_x_train = []
resized_x_test = []
# Preprocess 0
for image in x_train:
  resized_x_train.append(resize(image, (32, 32), anti_aliasing=True, preserve_range=False))
for image in x_test:
  resized_x_test.append(resize(image, (32, 32), anti_aliasing=True, preserve_range=False))
x_train = np.asarray(resized_x_train)
x_test = np.asarray(resized_x_test)
x_train = np.stack((x_train,)*3, axis=-1)
x_test = np.stack((x_test,)*3, axis=-1)
# End preprocess 0

# If you pick preprocess 0, you don't need to rerun the following code snippet
for image in x_train:
  resized_x_train.append(resize(image, (32, 32), anti_aliasing=True, preserve_range=True))
for image in x_test:
  resized_x_test.append(resize(image, (32, 32), anti_aliasing=True, preserve_range=True))
print(np.array(resized_x_train).shape)

# Plot the resized images
for i in range(9):
  plt.subplot(3, 3, i+1)
  plt.imshow(resized_x_train[i], cmap=plt.get_cmap('gray'))
  plt.axis("off")

# Preprocess 1
# VGG needs 3-channel images as input data
x_train = np.asarray(resized_x_train).astype("float32") / 255
x_test = np.asarray(resized_x_test).astype("float32") / 255
print(x_train.shape) # Expects: (60000, 32, 32)
x_train = np.stack((x_train,)*3, axis=-1)
x_test = np.stack((x_test,)*3, axis=-1)
print(x_train.shape) # Expects: (60000, 32, 32, 3)

# Preprocess 2
# VGG needs 3-channel images as input data, as well as its preprocessing function
print(x_train.shape) # Expects: (60000, 28, 28)
x_train = np.array(resized_x_train)
x_test = np.array(resized_x_test)
x_train = np.stack((x_train,)*3, axis=-1)
x_test = np.stack((x_test,)*3, axis=-1)
print(x_train.shape) # Expects: (60000, 32, 32, 3)
print(np.amax(x_train[0])) # Expects: 255.0
for image in x_train:
  image = keras.applications.vgg16.preprocess_input(image)
for image in x_test:
  image = keras.applications.vgg16.preprocess_input(image)
print(np.amax(x_train[0])) # Expects: 149.061

# Train/val split
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

conv_base = keras.applications.vgg16.VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(32, 32, 3))
conv_base.trainable = False
data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.1),
        layers.experimental.preprocessing.RandomZoom(0.2),
    ]
)
inputs = keras.Input(shape=(32, 32, 3))
x = data_augmentation(inputs)
x = conv_base(x)
x = layers.Flatten()(x)
# x = layers.Dense(256)(x)
# x = layers.Dropout(0.5)(x)
x = layers.Dense(256)(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(128)(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(64)(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs, outputs)
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="rmsprop",
              metrics=["accuracy"])

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="transferLearningModel",
        save_best_only=True,
        monitor="val_loss"),
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3,
    ),
]
history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=50,
    batch_size=512,
    validation_data=(x_val, y_val),
    callbacks=callbacks)

model = keras.models.load_model("transferLearningModel")
testLoss, testAcc = model.evaluate(x_test, y_test)
print(f"Test loss: {testLoss:.3f}")
print(f"Test accuracy: {testAcc:.3f}")

accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(accuracy) + 1)
plt.plot(epochs, accuracy, "bo", label="Training accuracy")
plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.show()

# Now to add fine tuning on top of transfer learning we do this:
# 1. Add your custom network on top of an already-trained base network.
# 2. Freeze the base network.
# 3. Train the part you added.
# 4. Unfreeze some layers in the base network. (Note that you should not unfreeze "batch normalization" layers, which is not relevant here since there are no such layers in VGG16. Batch normalization and its impact on fine-tuning is explained in the next chapter.)
# 5. Jointly train both these layers and the part you added.
# Note that the first three parts we have already done in the previous transfer learning section

conv_base.trainable = True
for layer in conv_base.layers[:-4]:
    layer.trainable = False

# Note that even if we recompile it here,  the weigths from before won't get altered
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.RMSprop(learning_rate=1e-5),
              metrics=["accuracy"])
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="transferLearningPlusFineTuningModel",
        save_best_only=True,
        monitor="val_loss"),
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3,
    ),
]

history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=50,
    batch_size=512,
    validation_data=(x_val, y_val),
    callbacks=callbacks)

model = keras.models.load_model("transferLearningPlusFineTuningModel")
testLoss, testAcc = model.evaluate(x_test, y_test)
print(f"Test loss: {testLoss:.3f}")
print(f"Test accuracy: {testAcc:.3f}")

accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(accuracy) + 1)
plt.plot(epochs, accuracy, "bo", label="Training accuracy")
plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.show()

# Interesting points:
# - Because the ImageNET weights were trained on RGB images, if we were to pass the input data as 3-channel greyscale images,
#   the accuracy of the model won't be very high because the weights cannot be properly leveraged
# - For 3-channel greyscale images, it can be better to scale the images manually rather than using the model's preprocesing
#   function like keras.applications.vgg16.preprocess_input(image)
# - The reason why test_accuracy can be higher than train_accuracy is due to regularization (e.g. Dropout).
#   See https://keras.io/getting_started/faq/#why-is-my-training-loss-much-higher-than-my-testing-loss

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