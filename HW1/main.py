from csv import reader
import math
import matplotlib.pyplot as plt
from math import log


def mean(age):
    values = 0
    count = 0
    for row in age:
        values += int(row)
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


def std(age):
    average = mean(age)
    result = 0.0  # The result
    count = 0.0  # Count number of rows for particular class index
    for row in age:
        result += square(float(row) - average)
        count += 1.0

    # return sqrt(result / count)
    return math.sqrt(result / count)


def priors(class_age, total_count):
    return class_age / total_count


def likelihoods(age, class_mean, class_std):
    list_of_likelihoods = []

    for i in age:
        likelihoods_class_1 = 1.0 / (class_std * math.sqrt(2.0 * math.pi)) * math.exp(
            -1.0 * ((square(float(i) - class_mean)) / (2.0 * square(class_std))))
        list_of_likelihoods.append(likelihoods_class_1)  # Appending tuple

    return list_of_likelihoods


def posteriors(age, class_1_likelihood, class_2_likelihood, class_3_likelihood, class1_prior, class2_prior,
               class3_prior):
    list_of_posteriors = []
    count = 0

    for i in age:
        posterior_class_1 = (class_1_likelihood[count] * class1_prior) / (
                (class_1_likelihood[count] * class1_prior) + (class_2_likelihood[count] * class2_prior) + (
                class_3_likelihood[count] * class3_prior))
        list_of_posteriors.append(posterior_class_1)
        count += 1

    return list_of_posteriors


with open('training.csv') as training_file:
    csv_reader = reader(training_file)
    list_of_rows = list(csv_reader)
    print(list_of_rows)
    list_of_rows.pop(0)
    print(list_of_rows)  # This is the data without including the 0 row (age and class)

    age_class_1 = []
    age_class_2 = []
    age_class_3 = []

    age = []

    # Stores the number of ages in class1 2 and 3 and total
    class1_count_age, class2_count_age, class3_count_age, total_count_age = 0, 0, 0, 0

    for i in list_of_rows:
        if i[1] == "1":
            age_class_1.append(int(i[0]))
            age.append(int(i[0]))
            class1_count_age += 1
            total_count_age += 1
        elif i[1] == "2":
            age_class_2.append(int(i[0]))
            age.append(int(i[0]))
            class2_count_age += 1
            total_count_age += 1
        elif i[1] == "3":
            age_class_3.append(int(i[0]))
            age.append(int(i[0]))
            class3_count_age += 1
            total_count_age += 1

    # Means
    mean_class_1, mean_class_2, mean_class_3 = mean(age_class_1), mean(age_class_2), mean(age_class_3)

    # STD
    std_class_1, std_class_2, std_class_3 = std(age_class_1), std(age_class_2), std(age_class_3)

    # Priors
    class_1_priors, class_2_priors, class_3_priors = priors(class1_count_age, total_count_age), priors(class2_count_age,
                                                                                                       total_count_age), priors(
        class3_count_age, total_count_age)

    # Likelihoods
    class_1_likelihood, class_2_likelihood, class_3_likelihood = likelihoods(age, mean_class_1,
                                                                             std_class_1), likelihoods(age,
                                                                                                       mean_class_2,
                                                                                                       std_class_2), likelihoods(
        age, mean_class_3, std_class_3)

    # Posteriors
    class_1_posteriors, class_2_posteriors, class_3_posteriors = posteriors(age, class_1_likelihood, class_2_likelihood,
                                                                            class_3_likelihood, class_1_priors,
                                                                            class_2_priors,
                                                                            class_3_priors), posteriors(age,
                                                                                                        class_2_likelihood,
                                                                                                        class_1_likelihood,
                                                                                                        class_3_likelihood,
                                                                                                        class_1_priors,
                                                                                                        class_2_priors,
                                                                                                        class_3_priors), posteriors(
        age, class_3_likelihood, class_2_likelihood, class_1_likelihood, class_1_priors, class_2_priors,
        class_3_priors)

    print('mean ', mean_class_1, mean_class_2, mean_class_3)
    print('std ', std(age_class_1), std(age_class_2), std(age_class_3))
    print('priors ', class_1_priors, class_2_priors, class_3_priors)
    print('age_class_1 ', age_class_1)
    print('age_class_1 len ', len(age_class_1))
    print('age_class_2 ', age_class_2)
    print('age_class_2 len ', len(age_class_2))
    print('age_class_3 ', age_class_3)
    print('age_class_3 len ', len(age_class_3))
    print('likelihoods class 1', class_1_likelihood)
    print('likelihoods class 2', class_2_likelihood)
    print('likelihoods class 3', class_3_likelihood)
    print('posteriors class 1', class_1_posteriors)
    print('posteriors class 2', class_2_posteriors)
    print('posteriors class 3', class_3_posteriors)

    ''' PLOTTING '''
    age_set = set()
    for i in list_of_rows:
        age_set.add(int(i[0]))

    # LIKELIHOODS
    plot_likelihood_class_1 = likelihoods(list(age_set), mean(age_class_1), std(age_class_1))
    plot_likelihood_class_2 = likelihoods(list(age_set), mean(age_class_2), std(age_class_2))
    plot_likelihood_class_3 = likelihoods(list(age_set), mean(age_class_3), std(age_class_3))

    # POSTERIORS
    plot_posterior_class_1 = posteriors(list(age_set), plot_likelihood_class_1, plot_likelihood_class_2,
                                        plot_likelihood_class_3, class_1_priors, class_2_priors, class_3_priors)
    plot_posterior_class_2 = posteriors(list(age_set), plot_likelihood_class_2, plot_likelihood_class_1,
                                        plot_likelihood_class_3, class_1_priors, class_2_priors, class_3_priors)
    plot_posterior_class_3 = posteriors(list(age_set), plot_likelihood_class_3, plot_likelihood_class_2,
                                        plot_likelihood_class_1, class_1_priors, class_2_priors, class_3_priors)

    plt.plot(list(age_set), plot_posterior_class_1, color='red', linestyle='dashed')
    plt.plot(list(age_set), plot_posterior_class_2, color='green', linestyle='dashed')
    plt.plot(list(age_set), plot_posterior_class_3, color='blue', linestyle='dashed')

    plt.plot(list(age_set), plot_likelihood_class_1, color='red')
    plt.plot(list(age_set), plot_likelihood_class_2, color='green')
    plt.plot(list(age_set), plot_likelihood_class_3, color='purple')
    #

    plt.legend(["P(C=1|X)", "P(C=2|X)", "P(C=3|X)", "P(X|C=1)", "P(X|C=2)", "P(X|C=3)"])

    plt.scatter(age_class_1, [-0.1] * class1_count_age, color='r', marker="x")
    plt.scatter(age_class_2, [-0.2] * class2_count_age, color='g', marker="o")
    plt.scatter(age_class_3, [-0.3] * class3_count_age, color='b', marker="+")

    plt.show()
