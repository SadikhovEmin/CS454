from csv import reader
import math
import matplotlib.pyplot as plt
from math import log


def mean(data, class_index):
    values = 0
    count = 0
    for row in data:
        if row[1] == class_index:
            values += int(row[0])
            count += 1

    return values / count


def square(x):
    return float(x) * float(x)


''' Need to change this implementation'''


def myPow(x, n):
    p = 1
    if n < 0:
        x = 1 / x
        n = abs(n)

    # Exponentiation by Squaring

    while n:
        if n % 2:
            p *= x
        x *= x
        n //= 2
    return p


def sqrt(x):
    # Base cases
    if (x == 0 or x == 1):
        return x

    # Starting from 1, try all numbers until
    # i*i is greater than or equal to x.
    i = 1.0
    result = 1.0
    while result <= x:
        i += 1
        result = i * i

    return i


def std(data, class_index):
    average = mean(data, class_index)
    result = 0.0  # The result
    count = 0.0  # Count number of rows for particular class index
    for row in data:
        if row[1] == class_index:
            result += square(float(row[0]) - average)
            count += 1.0

    # return sqrt(result / count)
    return math.sqrt(result / count)


def priors(data):
    class1, class2, class3 = 0, 0, 0
    count = 0

    for i in data:
        if i[1] == "1":
            class1 += 1
        elif i[1] == "2":
            class2 += 1
        elif i[1] == "3":
            class3 += 1
        count += 1
    return (class1 / count), (class2 / count), (class3 / count)


def likelihoods(data):
    list_of_likelihoods = []
    class1_mean, class2_mean, class3_mean = mean(data, "1"), mean(data, "2"), mean(data, "3")
    class1_std, class2_std, class3_std = std(data, "1"), std(data, "2"), std(data, "3")

    for i in data:
        # Here I need to get initial value and calculate likelihood for all of 3 classes
        # likelihoods_class_1, likelihoods_class_2, likelihoods_class_3 = 0, 0, 0
        likelihoods_class_1 = 1.0 / (class1_std * math.sqrt(2.0 * math.pi)) * math.exp(
            -1.0 * ((square(float(i[0]) - class1_mean)) / (2.0 * square(class1_std))))
        likelihoods_class_2 = 1.0 / (class2_std * math.sqrt(2.0 * math.pi)) * math.exp(
            -1.0 * ((square(float(i[0]) - class2_mean)) / (2.0 * square(class2_std))))
        likelihoods_class_3 = 1.0 / (class3_std * math.sqrt(2.0 * math.pi)) * math.exp(
            -1.0 * ((square(float(i[0]) - class3_mean)) / (2.0 * square(class3_std))))
        list_of_likelihoods.append((likelihoods_class_1, likelihoods_class_2, likelihoods_class_3))  # Appending tuple

    return list_of_likelihoods


def posteriors(data):
    list_of_posteriors = []
    class1_prior, class2_prior, class3_prior = priors(data)
    list_of_likelihoods = likelihoods(data)
    count = 0

    for i in data:
        posterior_class_1 = (list_of_likelihoods[count][0] * class1_prior) / (
                (list_of_likelihoods[count][0] * class1_prior) + (list_of_likelihoods[count][1] * class2_prior) + (
                list_of_likelihoods[count][2] * class3_prior))
        posterior_class_2 = (list_of_likelihoods[count][1] * class2_prior) / (
                (list_of_likelihoods[count][0] * class1_prior) + (list_of_likelihoods[count][1] * class2_prior) + (
                list_of_likelihoods[count][2] * class3_prior))
        posterior_class_3 = (list_of_likelihoods[count][2] * class3_prior) / (
                (list_of_likelihoods[count][0] * class1_prior) + (list_of_likelihoods[count][1] * class2_prior) + (
                list_of_likelihoods[count][2] * class3_prior))
        list_of_posteriors.append((posterior_class_1, posterior_class_2, posterior_class_3))

        count += 1

    return list_of_posteriors


with open('training.csv') as training_file:
    csv_reader = reader(training_file)
    list_of_rows = list(csv_reader)
    print(list_of_rows)
    list_of_rows.pop(0)
    print(list_of_rows)  # This is the data without including the 0 row (age and class)

    ''' Seperating Ages '''
    # ages_class_1, ages_class_2, ages_class_3 = [], [], []
    #
    # for i in list_of_rows:
    #     if i[1] == "1":
    #         ages_class_1.append(int(i[0]))
    #     elif i[1] == "2":
    #         ages_class_2.append(int(i[0]))
    #     elif i[1] == "3":
    #         ages_class_3.append(int(i[0]))
    # print('ages ', ages_class_2)
    age = set()
    for i in list_of_rows:
        age.add(int(i[0]))

    print(mean(list_of_rows, "3"))
    print('standart deviation ', std(list_of_rows, "1"), std(list_of_rows, "2"), std(list_of_rows, "3"))

    print('-------------------------')
    print(priors(list_of_rows))
    x, y, z = priors(list_of_rows)
    print(x, y, z)
    print('-------------------------')
    print('likelihoods ', likelihoods(list_of_rows))
    print('posteriors ', posteriors(list_of_rows))


    ''' PLOTTING '''
    print('-------------------------')
    list_of_likelihoods = likelihoods(list_of_rows)
    list_of_posteriors = posteriors(list_of_rows)

    likelihood_class_1 = []
    likelihood_class_2 = []
    likelihood_class_3 = []

    posteriors_class_1 = []
    posteriors_class_2 = []
    posteriors_class_3 = []

    for i in list_of_likelihoods:
        # print('inside loop ' , i)
        likelihood_class_1.append(float(i[0]))
        likelihood_class_2.append(float(i[1]))
        likelihood_class_3.append(float(i[2]))

    for i in list_of_posteriors:
        posteriors_class_1.append(float(i[0]))
        posteriors_class_2.append(float(i[1]))
        posteriors_class_3.append(float(i[2]))

    print(len(likelihood_class_1))

    print('class 1 likelihood ', likelihood_class_1)
    # print(len(ages_class_1))
    plt.xlabel('Age')
    plt.ylim(0, 1)
    plt.scatter(list(age), likelihood_class_1)
    plt.scatter(list(age), likelihood_class_2)
    plt.scatter(list(age), likelihood_class_3)

    plt.scatter(list(age), posteriors_class_1, marker='x')
    plt.scatter(list(age), posteriors_class_2, marker='x')
    plt.scatter(list(age), posteriors_class_3, marker='x')

    plt.show()
