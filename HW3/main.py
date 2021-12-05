import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LEARNING_RATE = 0.1
CLASS_SIZE = 10
FEATURES_SIZE = 100
training_matrix = pd.read_csv('training.csv', header=None, skiprows=1)
testing_matrix = pd.read_csv('testing.csv', header=None, skiprows=1)

weight_matrix = np.random.uniform(low=-0.01, high=0.01,
                                  size=(FEATURES_SIZE + 1, CLASS_SIZE))  # Random matrix for weights
weight_matrix_best = np.zeros((FEATURES_SIZE + 1, CLASS_SIZE))  # Best possible values for matrix for weights


def softmax(x):
    value = np.exp(x - np.max(x))
    return value / value.sum()


def encode(y_value):
    return np.transpose(np.eye(10)[y_value])


def accuracy_confusion_matrix(x):
    correct = 0
    total = 0
    for i in range(len(x)):
        for j in range(len(x[i])):
            total += x[i][j]
            if i == j:
                correct += x[i][j]
    return correct / total


def forward(slp_object):
    for dataset_row_count in range(slp_object.number_of_rows):  # might need to make slp_object SLP_TRAIN
        row = slp_object.x.iloc[dataset_row_count, :]
        row = np.append(row, 1)
        row = np.transpose(row)

        o = np.zeros(CLASS_SIZE)

        for i in range(CLASS_SIZE):
            o[i] = np.dot(weight_matrix[:, i], row)

        slp_object.y_matrix[dataset_row_count] = softmax(o)


class SLP(object):
    def __init__(self, matrix):
        self.labels = matrix.iloc[:, 0]  # Gets only the first column (Just class name)
        self.x = matrix.iloc[:, 1:] / 255  # Gets all rows and columns except the first one
        self.number_of_rows = matrix.shape[0]
        self.y_matrix = np.zeros((self.number_of_rows, CLASS_SIZE))


def train(SLP_Train, SLP_Test):
    global weight_matrix_best
    accuracy_best = 0
    best_label_train = []
    best_label_test = []
    confusion_matrix_train = np.zeros((CLASS_SIZE, CLASS_SIZE))  # Confusion matrix for training set with zeroes
    confusion_matrix_test = np.zeros((CLASS_SIZE, CLASS_SIZE))  # Confusion matrix for testing set with zeroes

    for epoch in range(50):

        temp_weight = np.zeros(weight_matrix.shape)  # Copy w_matrix to the temp matrix

        for dataset_row_count in range(SLP_Train.number_of_rows):
            row = SLP_Train.x.iloc[dataset_row_count, :]
            row = np.append(row, 1)
            row = np.transpose(row)

            o = np.zeros(CLASS_SIZE)

            for i in range(CLASS_SIZE):
                o[i] = np.dot(weight_matrix[:, i], row)

            SLP_Train.y_matrix[dataset_row_count] = softmax(o)

            encode_matrix = encode(SLP_Train.labels.iloc[dataset_row_count])

            for i in range(CLASS_SIZE):
                for j in range(FEATURES_SIZE + 1):
                    temp_weight[j, i] += (encode_matrix[i] - SLP_Train.y_matrix[dataset_row_count][i]) * row[j]

            for i in range(CLASS_SIZE):
                for j in range(FEATURES_SIZE + 1):
                    weight_matrix[j, i] += temp_weight[j, i] * LEARNING_RATE

        forward(SLP_Train)
        forward(SLP_Test)

        label_train = []
        label_test = []

        correct_train = 0
        correct_test = 0
        for dataset_row_count in range(SLP_Train.number_of_rows):
            if SLP_Train.labels.iloc[dataset_row_count] == np.argmax(SLP_Train.y_matrix[dataset_row_count]):
                correct_train += 1
            if SLP_Test.labels.iloc[dataset_row_count] == np.argmax(SLP_Test.y_matrix[dataset_row_count]):
                correct_test += 1

            label_train.append(np.argmax(SLP_Train.y_matrix[dataset_row_count]))
            label_test.append(np.argmax(SLP_Test.y_matrix[dataset_row_count]))

        accuracy_training = correct_train / SLP_Train.number_of_rows
        accuracy_test = correct_test / SLP_Test.number_of_rows

        print("Epoch: ", epoch + 1, "Training Accuracy: ", accuracy_training, "Testing Accuracy: ", accuracy_test)

        if accuracy_test > accuracy_best:
            accuracy_best = accuracy_test
            weight_matrix_best = weight_matrix
            best_label_train = label_train
            best_label_test = label_test

    print('weight matrix best : ', weight_matrix_best.shape)

    for i in range(SLP_Train.number_of_rows):
        confusion_matrix_train[best_label_train[i], SLP_Train.labels[i]] += 1

    for i in range(SLP_Test.number_of_rows):
        confusion_matrix_test[best_label_test[i], SLP_Test.labels[i]] += 1

    print('Confusion train : ', confusion_matrix_train)
    print('Confusion test : ', confusion_matrix_test)


def plot():
    fig, axes = plt.subplots(4, 3)
    fig.set_figheight(8)
    fig.set_figwidth(8)
    fig.suptitle('Mean Images')
    for label in range(12):
        if label < 10:
            axes[label // 3][label % 3].imshow(weight_matrix_best[:-1, :][:, label].reshape(10, 10), )
        axes[label // 3][label % 3].axis('off')
    plt.show()


if __name__ == '__main__':
    SLP_Train = SLP(training_matrix)
    SLP_Test = SLP(testing_matrix)

    train(SLP_Train, SLP_Test)
    plot()
