from csv import reader


def mean(data):
    values = 0
    count = 0
    for row in data:
        values += float(row)
        count += 1

    return values / count


with open('training.csv') as training_file:
    csv_reader = reader(training_file)
    training_csv = list(csv_reader)
    # print(training_csv)
    training_csv.pop(0)
    # print(training_csv)

    petal_length_setosa, petal_width_setosa = [], []
    petal_length_versicolor, petal_width_versicolor = [], []
    petal_length_virginica, petal_width_virginica = [], []

    for i in training_csv:
        # print(i[2])

        if i[2] == "Iris-setosa":
            petal_length_setosa.append(float(i[0]))
            petal_width_setosa.append(float(i[1]))
        elif i[2] == "Iris-versicolor":
            petal_length_versicolor.append(float(i[0]))
            petal_width_versicolor.append(float(i[1]))
        elif i[2] == "Iris-virginica":
            petal_length_virginica.append(float(i[0]))
            petal_width_virginica.append(float(i[1]))

    ''' SETOSA '''
    petal_length_setosa_mean = mean(petal_length_setosa)
    petal_width_setosa_mean = mean(petal_width_setosa)

    print('Length : ', petal_length_setosa_mean, 'Width : ', petal_width_setosa_mean)

    ''' VERSICOLOR '''
    petal_length_versicolor_mean = mean(petal_length_versicolor)
    petal_width_versicolor_mean = mean(petal_width_versicolor)

    print('Length : ', petal_length_versicolor_mean, 'Width : ', petal_width_versicolor_mean)

    ''' VIRGINICA '''
    petal_length_virginica_mean = mean(petal_length_virginica)
    petal_width_virginica_mean = mean(petal_width_virginica)

    print('Length : ', petal_length_virginica_mean, 'Width : ', petal_width_virginica_mean)

    ''' Confusion matrix '''

#
# if __name__ == '__main__':
#     pass
