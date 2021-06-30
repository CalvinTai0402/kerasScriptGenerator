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

export const OxfordPetsImageSegmentation = `from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras import layers
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import random
import os

# Download data
!wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
!wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
!tar -xf images.tar.gz
!tar -xf annotations.tar.gz

# Data preprocessing
input_dir = "images/"
target_dir = "annotations/trimaps/"

input_img_paths = sorted(
    [os.path.join(input_dir, fname)
     for fname in os.listdir(input_dir)
     if fname.endswith(".jpg")])
target_paths = sorted(
    [os.path.join(target_dir, fname)
     for fname in os.listdir(target_dir)
     if fname.endswith(".png") and not fname.startswith(".")])

# Plot an image
plt.axis("off")
plt.imshow(load_img(input_img_paths[9]))

# Plot a mask
def display_target(target_array):
    normalized_array = (target_array.astype("uint8") - 1) * 127
    plt.axis("off")
    plt.imshow(normalized_array[:, :, 0])

img = img_to_array(load_img(target_paths[9], color_mode="grayscale", target_size=(200,200)))
display_target(img)

# More data preprocessing
img_size = (200, 200)
num_imgs = len(input_img_paths)

random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_paths)

def path_to_input_image(path):
    return img_to_array(load_img(path, target_size=img_size))

def path_to_target(path):
    img = img_to_array(
        load_img(path, target_size=img_size, color_mode="grayscale"))
    img = img.astype("uint8") - 1
    return img

input_imgs = np.zeros((num_imgs,) + img_size + (3,), dtype="float32")
targets = np.zeros((num_imgs,) + img_size + (1,), dtype="uint8")
for i in range(num_imgs):
    input_imgs[i] = path_to_input_image(input_img_paths[i])
    targets[i] = path_to_target(target_paths[i])

# Train/val/test split
num_test_samples = 100
num_val_samples = 1000 + num_test_samples
train_input_imgs = input_imgs[:-num_val_samples]
train_targets = targets[:-num_val_samples]
val_input_imgs = input_imgs[-num_val_samples:-num_test_samples]
val_targets = targets[-num_val_samples:-num_test_samples]
test_input_imgs = input_imgs[-num_test_samples:]
test_targets = targets[-num_test_samples:]

# Build model
def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))
    x = layers.experimental.preprocessing.Rescaling(1./255)(inputs)

    x = layers.Conv2D(64, 3, strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.Conv2D(128, 3, strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.Conv2D(256, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(256, 3, activation="relu", padding="same")(x)

    x = layers.Conv2DTranspose(256, 3, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(256, 3, activation="relu", padding="same", strides=2)(x)
    x = layers.Conv2DTranspose(128, 3, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(128, 3, activation="relu", padding="same", strides=2)(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", padding="same", strides=2)(x)

    outputs = layers.Conv2D(3, num_classes, activation="softmax", padding="same")(x)

    model = keras.Model(inputs, outputs)
    return model

model = get_model(img_size=img_size, num_classes=3)
model.summary()

# Compile model
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="rmsprop",
              metrics=["accuracy"])
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="ImageSegmentationModel",
        save_best_only=True,
        monitor="val_loss"),
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3,
    ),
]

# Train model
history = model.fit(train_input_imgs, train_targets,
                    epochs=50,
                    callbacks=callbacks,
                    batch_size=64,
                    validation_data=(val_input_imgs, val_targets))

# Plot the training and validation loss/accuracy
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

# Predict on the test set
model = keras.models.load_model("ImageSegmentationModel")
test_image = test_input_imgs[4]
plt.subplot(1, 2, 1)
plt.axis("off")
plt.imshow(array_to_img(test_image))
mask = model.predict(np.expand_dims(test_image, 0))[0]
def display_mask(pred):
    mask = np.argmax(pred, axis=-1)
    mask *= 127
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.imshow(mask)
display_mask(mask)

# Train on colab: https://research.google.com/colaboratory/
`

export const heatMapGeneration = `from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras import layers
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import random
import os

# Download model
model = keras.applications.xception.Xception(weights="imagenet")
model.summary()

# Get the image and preprocess it
img_path = keras.utils.get_file(
    fname="elephant.jpg",
    origin="https://img-datasets.s3.amazonaws.com/elephant.jpg")

def get_img_array(img_path, target_size):
    img = keras.preprocessing.image.load_img(img_path, target_size=target_size)
    array = keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    array = keras.applications.xception.preprocess_input(array)
    return array

img_array = get_img_array(img_path, target_size=(299, 299))

# Take a look at what the model thinks the image is
preds = model.predict(img_array)
print(keras.applications.xception.decode_predictions(preds, top=3)[0])
print(np.argmax(preds[0])) # African_elephant is class 386

# Generate heatmap
last_conv_layer_name = "block14_sepconv2_act"
classifier_layer_names = [
    "avg_pool",
    "predictions",
]
last_conv_layer = model.get_layer(last_conv_layer_name)
last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)
classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
x = classifier_input
for layer_name in classifier_layer_names:
    x = model.get_layer(layer_name)(x)
classifier_model = keras.Model(classifier_input, x)
with tf.GradientTape() as tape:
    last_conv_layer_output = last_conv_layer_model(img_array)
    tape.watch(last_conv_layer_output)
    preds = classifier_model(last_conv_layer_output)
    top_pred_index = tf.argmax(preds[0])
    top_class_channel = preds[:, top_pred_index]

grads = tape.gradient(top_class_channel, last_conv_layer_output)
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
last_conv_layer_output = last_conv_layer_output.numpy()[0]
for i in range(pooled_grads.shape[-1]):
    last_conv_layer_output[:, :, i] *= pooled_grads[i]
heatmap = np.mean(last_conv_layer_output, axis=-1)

# Plot heatmap
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)

# Superimpose heatmap and image
img = keras.preprocessing.image.load_img(img_path)
img = keras.preprocessing.image.img_to_array(img)

heatmap = np.uint8(255 * heatmap)

jet = cm.get_cmap("jet")
jet_colors = jet(np.arange(256))[:, :3]
jet_heatmap = jet_colors[heatmap]

jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

superimposed_img = jet_heatmap * 0.4 + img
superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

save_path = "elephant_cam.jpg"
superimposed_img.save(save_path)

# Train on colab: https://research.google.com/colaboratory/
`

export const activationVisualization = `from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras import layers
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import random
import os

# Download model
model = keras.applications.xception.Xception(weights="imagenet")
model.summary()

# Get the image and preprocess it
img_path = keras.utils.get_file(
    fname="cat.jpg",
    origin="https://img-datasets.s3.amazonaws.com/cat.jpg")

def get_img_array(img_path, target_size):
    img = load_img(
        img_path, target_size=target_size)
    array = img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array

img_tensor = get_img_array(img_path, target_size=(299, 299))

# Plot the image
plt.axis("off")
plt.imshow(img_tensor[0].astype("uint8"))
plt.show()

# Instantiating a model that returns layer activations
layer_outputs = []
layer_names = []
for layer in model.layers:
    if isinstance(layer, (layers.SeparableConv2D, layers.MaxPooling2D)):
        layer_outputs.append(layer.output)
        layer_names.append(layer.name)
activation_model = keras.Model(inputs=model.input, outputs=layer_outputs)

# Compute and plot the activation
activations = activation_model.predict(img_tensor)
first_layer_activation = activations[0]
plt.matshow(first_layer_activation[0, :, :, 5], cmap="viridis")

# Plot all activations. Note that the details of the specific input become
# less and less noticeable as we go deeper into the network, whereas the 
# features that resemble the class (cat) become more prominent.
images_per_row = 16
for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]
    size = layer_activation.shape[1]
    n_cols = n_features // images_per_row
    display_grid = np.zeros(((size + 1) * n_cols - 1,
                             images_per_row * (size + 1) - 1))
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_index = col * images_per_row + row
            channel_image = layer_activation[0, :, :, channel_index].copy()
            if channel_image.sum() != 0:
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype("uint8")
            display_grid[
                col * (size + 1): (col + 1) * size + col,
                row * (size + 1) : (row + 1) * size + row] = channel_image
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.axis("off")
    plt.imshow(display_grid, aspect="auto", cmap="viridis")

# Train on colab: https://research.google.com/colaboratory/
`