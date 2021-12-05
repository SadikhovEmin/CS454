import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt


training_matrix = pd.read_csv('training.csv', header=None, skiprows=1)
testing_matrix = pd.read_csv('testing.csv', header=None, skiprows=1)
LEARNING_RATE = 0.1
CLASS_SIZE = 10
FEATURES_SIZE = 100


def accuracy_confusion_matrix(x):
    correct = 0
    total = 0
    for i in range(len(x)):
        for j in range(len(x[i])):
            total += x[i][j]
            if i == j:
                correct += x[i][j]
    return correct / total


def encode(y):
    return np.transpose(np.eye(10)[y])


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class MLP(object):
    def __init__(self, matrix):
        self.labels = matrix.iloc[:, 0]  # Gets only the first column (Just class name)
        self.x = matrix.iloc[:, 1:] / 255  # Gets all rows and columns except the first one
        self.number_of_rows = matrix.shape[0]
        self.y_matrix = np.zeros((self.number_of_rows, CLASS_SIZE))


def forward(mlp_object, hidden_layer, weight_matrix, field_matrix):
    for instance in range(mlp_object.number_of_rows):
        row = mlp_object.x.iloc[instance, :]
        row = np.append(1, row)
        row = np.transpose(row)

        logit = np.zeros(hidden_layer)
        for h in range(hidden_layer):
            w_h_T = np.transpose(weight_matrix[:, h])
            logit[h] = np.dot(w_h_T, row)

        logit = sigmoid(logit)
        logit = np.append(1, logit)
        logit = np.transpose(logit)

        activation = np.zeros(CLASS_SIZE)
        for i in range(CLASS_SIZE):
            activation[i] = np.dot(np.transpose(field_matrix[:, i]), logit)

        mlp_object.y_matrix[instance] = softmax(activation)


def train(MLP_Train, MLP_Test):
    for H in [5, 10, 25, 50, 75]:
        accuracy_best = 0
        best_label_train = []
        best_label_test = []

        training_accuracies = []
        test_accuracies = []

        field_matrix = np.random.uniform(low=-0.01, high=0.01, size=(H + 1, CLASS_SIZE))
        weight_matrix = np.random.uniform(low=-0.01, high=0.01, size=(FEATURES_SIZE + 1, H))
        for epochs in range(50):
            random_list = list(range(MLP_Train.number_of_rows))
            random.shuffle(random_list)

            for element in random_list:
                row = MLP_Train.x.iloc[element, :]
                row = np.append(1, row)
                row = np.transpose(row)

                logit = np.zeros(H)

                for h in range(H):
                    w_h_T = np.transpose(weight_matrix[:, h])
                    logit[h] = np.dot(w_h_T, row)

                logit = sigmoid(logit)

                logit = np.append(1, logit)
                logit = np.transpose(logit)

                activation = np.zeros(CLASS_SIZE)
                for i in range(CLASS_SIZE):
                    activation[i] = np.dot(np.transpose(field_matrix[:, i]), logit)

                MLP_Train.y_matrix[element] = softmax(activation)

                change_field_matrix = np.zeros(field_matrix.shape)
                change_weight_matrix = np.zeros(weight_matrix.shape)

                r_t = encode(MLP_Train.labels.iloc[element])

                for i in range(CLASS_SIZE):
                    change_field_matrix[:, i] = LEARNING_RATE * (r_t[i] - MLP_Train.y_matrix[element][i]) * logit

                for h in range(H):
                    total = 0.0
                    for i in range(CLASS_SIZE):
                        total += (r_t[i] - MLP_Train.y_matrix[element][i]) * field_matrix[h, i]
                    change_weight_matrix[:, h] = LEARNING_RATE * total * logit[h + 1] * (1 - logit[h + 1]) * row

                for i in range(CLASS_SIZE):
                    field_matrix[:, i] += change_field_matrix[:, i]

                for h in range(H):
                    weight_matrix[:, h] += change_weight_matrix[:, h]

            forward(MLP_Train, H, weight_matrix, field_matrix)
            forward(MLP_Test, H, weight_matrix, field_matrix)

            label_train = []
            label_test = []

            correct_train = 0
            correct_test = 0
            for t in range(MLP_Train.number_of_rows):
                if MLP_Train.labels.iloc[t] == np.argmax(MLP_Train.y_matrix[t]):
                    correct_train += 1
                if MLP_Test.labels.iloc[t] == np.argmax(MLP_Test.y_matrix[t]):
                    correct_test += 1

                label_train.append(np.argmax(MLP_Train.y_matrix[t]))
                label_test.append(np.argmax(MLP_Test.y_matrix[t]))

            accuracy_training = correct_train / MLP_Train.number_of_rows
            accuracy_test = correct_test / MLP_Test.number_of_rows

            training_accuracies.append(accuracy_training)
            test_accuracies.append(accuracy_test)

            print("H: ", H, " Epoch: ", epochs + 1, "Training Accuracy: ", accuracy_training, "Testing Accuracy: ",
                  accuracy_test)

            if accuracy_test > accuracy_best:
                accuracy_best = accuracy_test
                best_label_train = label_train
                best_label_test = label_test

        confusion_matrix_train = np.zeros((CLASS_SIZE, CLASS_SIZE))
        confusion_matrix_test = np.zeros((CLASS_SIZE, CLASS_SIZE))

        for i in range(MLP_Train.number_of_rows):
            confusion_matrix_train[best_label_train[i], MLP_Train.labels[i]] += 1

        for i in range(MLP_Test.number_of_rows):
            confusion_matrix_test[best_label_test[i], MLP_Test.labels[i]] += 1

        print('H = ', H)
        print('Confusion train : ', confusion_matrix_train)
        print('Confusion test : ', confusion_matrix_test)

        plt.title("H = " + str(H))
        plt.plot(training_accuracies, label='Training Accuracy')
        plt.plot(test_accuracies, label='Testing Accuracy')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend(loc='lower right')
        plt.show()


if __name__ == '__main__':
    MLP_Train = MLP(training_matrix)
    MLP_Test = MLP(testing_matrix)

    train(MLP_Train, MLP_Test)
