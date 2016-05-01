import operator

import numpy
import matplotlib
# I don't want to waste my time to setting matplotlib up.
matplotlib.use('TkAgg')
import matplotlib.pyplot


def read_testset(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [x for x in (x.strip() for x in lines) if x]

    count = len(lines)
    matrix = numpy.zeros((count, 3))
    class_labels = []

    for row, line in enumerate(lines):
        components = line.split('\t')
        matrix[row, :] = components[0:3]
        class_labels.append(components[-1])
    return matrix, class_labels


def create_dataset():
    group = numpy.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(in_vector, dataset, labels, k):
    size = dataset.shape[0]
    diff = numpy.tile(in_vector, (size, 1)) - dataset
    square_diff = diff ** 2
    square_distances = square_diff.sum(axis=1)
    distances = square_distances ** 0.5
    sorted_distance_indices = distances.argsort()

    class_counts = {}
    for index in sorted_distance_indices[:k]:
        label = labels[index]
        class_counts[label] = class_counts.get(label, 0) + 1
    sorted_class_counts = sorted(class_counts.items(),
                                 key=operator.itemgetter(1),
                                 reverse=True)
    return sorted_class_counts[0][0]


def label_to_color(label):
    return {
        'didntLike': [1.0, 0.0, 0.0],
        'smallDoses': [1.0, 1.0, 0.0],
        'largeDoses': [0.0, 1.0, 0.0],
    }[label]


def visualize(column1, column2):
    data, labels = read_testset('datingTestSet.txt')
    figure = matplotlib.pyplot.figure()
    axis = figure.add_subplot(111)
    colors = [label_to_color(x) for x in labels]
    axis.scatter(data[:, column1], data[:, column2], c=colors)
    matplotlib.pyplot.show()


def main():
    visualize(0, 1)


if __name__ == '__main__':
    main()
