export const optimizers = [
    { name: 'keras.optimizers.SGD(0.01)', value: 'keras.optimizers.SGD(0.01)' },
    { name: 'keras.optimizers.RMSprop(0.01)', value: 'keras.optimizers.RMSprop(0.01)' },
    { name: 'keras.optimizers.Adam(0.01)', value: 'keras.optimizers.Adam(0.01)' },
    { name: 'keras.optimizers.Adadelta(0.01)', value: 'keras.optimizers.Adadelta(0.01)' },
    { name: 'keras.optimizers.Adagrad(0.01)', value: 'keras.optimizers.Adagrad(0.01)' },
    { name: 'keras.optimizers.Adamax(0.01)', value: 'keras.optimizers.Adamax(0.01)' },
    { name: 'keras.optimizers.Nadam(0.01)', value: 'keras.optimizers.Nadam(0.01)' },
    { name: 'keras.optimizers.ftrl(0.01)', value: 'keras.optimizers.ftrl(0.01)' }];

export const losses = [
    { name: 'keras.losses.BinaryCrossentropy()', value: 'keras.losses.BinaryCrossentropy()' },
    { name: 'keras.losses.CategoricalCrossentropy()', value: 'keras.losses.CategoricalCrossentropy()' },
    { name: 'keras.losses.SparseCategoricalCrossentropy()', value: 'keras.losses.SparseCategoricalCrossentropy()' },
    { name: 'keras.losses.Poisson()', value: 'keras.losses.Poisson()' },
    { name: 'keras.losses.KLDivergence()', value: 'keras.losses.KLDivergence()' },
    { name: 'keras.losses.MeanSquaredError()', value: 'keras.losses.MeanSquaredError()' },
    { name: 'keras.losses.MeanAbsoluteError()', value: 'keras.losses.MeanAbsoluteError()' },
    { name: 'keras.losses.MeanAbsolutePercentageError()', value: 'keras.losses.MeanAbsolutePercentageError()' },
    { name: 'keras.losses.MeanSquaredLogarithmicError()', value: 'keras.losses.MeanSquaredLogarithmicError()' },
    { name: 'keras.losses.CosineSimilarity()', value: 'keras.losses.CosineSimilarity()' },
    { name: 'keras.losses.Huber()', value: 'keras.losses.Huber()' },
    { name: 'keras.losses.LogCosh()', value: 'keras.losses.LogCosh()' },
    { name: 'keras.losses.Hinge()', value: 'keras.losses.Hinge()' },
    { name: 'keras.losses.SquaredHinge()', value: 'keras.losses.SquaredHinge()' },
    { name: 'keras.losses.CategoricalHinge()', value: 'keras.losses.CategoricalHinge()' }]

export const metrics = [
    { name: 'keras.metrics.Accuracy()', value: 'keras.metrics.Accuracy()' },
    { name: 'keras.metrics.BinaryAccuracy()', value: 'keras.metrics.BinaryAccuracy()' },
    { name: 'keras.metrics.CategoricalAccuracy()', value: 'keras.metrics.CategoricalAccuracy()' },
    { name: 'keras.metrics.TopKCategoricalAccuracy()', value: 'keras.metrics.TopKCategoricalAccuracy()' },
    { name: 'keras.metrics.SparseTopKCategoricalAccuracy()', value: 'keras.metrics.SparseTopKCategoricalAccuracy()' },
    { name: 'keras.metrics.BinaryCrossentropy()', value: 'keras.metrics.BinaryCrossentropy()' },
    { name: 'keras.metrics.CategoricalCrossentropy()', value: 'keras.metrics.CategoricalCrossentropy()' },
    { name: 'keras.metrics.SparseCategoricalCrossentropy()', value: 'keras.metrics.SparseCategoricalCrossentropy()' },
    { name: 'keras.metrics.KLDivergence()', value: 'keras.metrics.KLDivergence()' },
    { name: 'keras.metrics.Poisson()', value: 'keras.metrics.Poisson()' },
    { name: 'keras.metrics.MeanSquaredError()', value: 'keras.metrics.MeanSquaredError()' },
    { name: 'keras.metrics.RootMeanSquaredError()', value: 'keras.metrics.RootMeanSquaredError()' },
    { name: 'keras.metrics.MeanAbsoluteError()', value: 'keras.metrics.MeanAbsoluteError()' },
    { name: 'keras.metrics.MeanAbsolutePercentageError()', value: 'keras.metrics.MeanAbsolutePercentageError()' },
    { name: 'keras.metrics.MeanSquaredLogarithmicError()', value: 'keras.metrics.MeanSquaredLogarithmicError()' },
    { name: 'keras.metrics.CosineSimilarity()', value: 'keras.metrics.CosineSimilarity()' },
    { name: 'keras.metrics.LogCoshError()', value: 'keras.metrics.LogCoshError()' },
    { name: 'keras.metrics.AUC()', value: 'keras.metrics.AUC()' },
    { name: 'keras.metrics.Precision()', value: 'keras.metrics.Precision()' },
    { name: 'keras.metrics.Recall()', value: 'keras.metrics.Recall()' },
    { name: 'keras.metrics.TruePositives()', value: 'keras.metrics.TruePositives()' },
    { name: 'keras.metrics.TrueNegatives()', value: 'keras.metrics.TrueNegatives()' },
    { name: 'keras.metrics.FalsePositives()', value: 'keras.metrics.FalsePositives()' },
    { name: 'keras.metrics.FalseNegatives()', value: 'keras.metrics.FalseNegatives()' },
    { name: 'keras.metrics.PrecisionAtRecall(0.5)', value: 'keras.metrics.PrecisionAtRecall(0.5)' },
    { name: 'keras.metrics.SensitivityAtSpecificity(0.5)', value: 'keras.metrics.SensitivityAtSpecificity(0.5)' },
    { name: 'keras.metrics.SpecificityAtSensitivity(0.5)', value: 'keras.metrics.SpecificityAtSensitivity(0.5)' },
    { name: 'keras.metrics.MeanIoU(num_classes=2)', value: 'keras.metrics.MeanIoU(num_classes=2)' },
    { name: 'keras.metrics.Hinge()', value: 'keras.metrics.Hinge()' },
    { name: 'keras.metrics.SquaredHinge()', value: 'keras.metrics.SquaredHinge()' },
    { name: 'keras.metrics.CategoricalHinge()', value: 'keras.metrics.CategoricalHinge()' }]

export const layers = [
    { name: 'keras.layers.Dense(10, activation="relu")', value: 'keras.layers.Dense(10, activation="relu")' },
    { name: 'keras.layers.Dense(1, activation="sigmoid")', value: 'keras.layers.Dense(1, activation="sigmoid")' },
    { name: 'keras.layers.Dense(10, activation="softmax")', value: 'keras.layers.Dense(10, activation="softmax")' },
    { name: 'keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu")', value: 'keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu")' },
    { name: 'keras.layers.MaxPooling2D(pool_size=2)', value: 'keras.layers.MaxPooling2D(pool_size=2)' },
    { name: 'keras.layers.Flatten()', value: 'keras.layers.Flatten()' },
    { name: 'keras.layers.Input(shape=(28, 28, 1))', value: 'keras.layers.Input(shape=(28, 28, 1))' },
    { name: 'keras.layers.experimental.preprocessing.Rescaling(1./255)', value: 'keras.layers.experimental.preprocessing.Rescaling(1./255)(inputs)' }]

export const examples = [
    { name: 'IMDBBinaryClassification', value: 'IMDBBinaryClassification' },
    { name: 'MNISTCategoricalClassification', value: 'MNISTCategoricalClassification' },
    { name: 'MNISTCategoricalClassificationWithCNN', value: 'MNISTCategoricalClassificationWithCNN' },
    { name: 'BostonHousingRegression', value: 'BostonHousingRegression' }]

export const callbacks = [
    { name: 'keras.callbacks.EarlyStopping(monitor="val_sparse_categorical_accuracy",patience=3)', value: 'keras.callbacks.EarlyStopping(monitor="val_sparse_categorical_accuracy",patience=3)' },
    { name: 'keras.callbacks.ModelCheckpoint(filepath="saved_model",monitor="val_loss",save_best_only=True)', value: 'keras.callbacks.ModelCheckpoint(filepath="saved_model",monitor="val_loss",save_best_only=True)' }]

export let fileContent = `from tensorflow.keras.datasets import imdb
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

# Data preprocessing (expects the outputs: partial_x_train, x_val, x_test, partial_y_train, y_val, y_test)
# Example (courtesy to François Chollet):
# def vectorize_sequences(sequences, dimension=10000):
#     results = np.zeros((len(sequences), dimension))
#     for i, sequence in enumerate(sequences):
#         results[i, sequence] = 1.
#     return results
# (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
# x_train = vectorize_sequences(train_data)
# x_test = vectorize_sequences(test_data)
# y_train = np.asarray(train_labels).astype("float32")
# y_test = np.asarray(test_labels).astype("float32")
# x_val = x_train[:10000]
# partial_x_train = x_train[10000:]
# y_val = y_train[:10000]
# partial_y_train = y_train[10000:]

# Building and compiling the model
callbacks_list = [CALLBACKS]
model = keras.Sequential([
    LAYERS
])
model.compile(optimizer=OPTIMIZERS,
    loss=LOSS,
    metrics=[METRICS])

# Training the model
history = model.fit(partial_x_train,
    partial_y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_val, y_val),
    callbacks=callbacks_list)

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
acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]
plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Train on colab: https://research.google.com/colaboratory/
`


