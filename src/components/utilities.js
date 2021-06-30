export const colabFileUploadDownload = `# Download
from google.colab import files
files.download('example.txt')

# Upload
from google.colab import files
files.upload()
`

export const shuffleData = `from tensorflow.keras.datasets import boston_housing
import numpy as np
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
np.random.seed(1337)
np.random.shuffle(x_train)
np.random.seed(1337)
np.random.shuffle(y_train)
`

export const trainValTestSplit = `# Usually test data already come separated
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]
`

export const standardization = `from tensorflow.keras.datasets import boston_housing
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
mean = x_train.mean(axis=0)
x_train -= mean
std = x_train.std(axis=0)
x_train /= std
x_test -= mean
x_test /= std
# Note that we scale the test data by the mean and std of the TRAIN data
`

export const normalization = `from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.datasets import boston_housing
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
# fit scaler on training data and then transform on training and test data
norm = MinMaxScaler().fit(x_train)
x_train = norm.transform(x_train)
x_test = norm.transform(x_test)`

export const kFoldCrossValidation = `k = 3
num_validation_samples = len(data) // k
np.random.shuffle(data)
validation_scores = []
for fold in range(k):
    validation_data = data[num_validation_samples * fold:
                           num_validation_samples * (fold + 1)]
    training_data = np.concatenate(
        data[:num_validation_samples * fold],
        data[num_validation_samples * (fold + 1):])
    model = get_model()
    model.fit(training_data, ...)
    validation_score = model.evaluate(validation_data, ...)
    validation_scores.append(validation_score)
validation_score = np.average(validation_scores)
model = get_model()
model.fit(data, ...)
test_score = model.evaluate(test_data, ...)
`

export const iterativeKFoldCrossValidation = `# The idea is to repeatedly perform k-fold cross validation
# and before each iteration, shuffle the data. Something like:
validation_scores = []
for i in range(3):
    k = 3
    num_validation_samples = len(data) // k
    np.random.shuffle(data)
    for fold in range(k):
        validation_data = data[num_validation_samples * fold:
                            num_validation_samples * (fold + 1)]
        training_data = np.concatenate(
            data[:num_validation_samples * fold],
            data[num_validation_samples * (fold + 1):])
        model = get_model()
        model.fit(training_data, ...)
        validation_score = model.evaluate(validation_data, ...)
        validation_scores.append(validation_score)
validation_score = np.average(validation_scores)
model = get_model()
model.fit(data, ...)
test_score = model.evaluate(test_data, ...)
`