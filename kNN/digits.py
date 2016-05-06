import glob
import operator
import os
import shutil
import zipfile

import numpy


SIZE = 32


def read_image(filename):
    vector = numpy.zeros((1, SIZE * SIZE))

    with open(filename) as f:
        for i in range(SIZE):
            line = f.readline()
            for j in range(SIZE):
                value = int(line[j])
                vector[0, SIZE * i + j] = value
    return vector


def get_class_label(filename):
    name, ext = os.path.splitext(os.path.basename(filename))
    return int(name.split('_', 1)[0])


def classify(in_vector, dataset, labels, k):
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


def unzip_data():
    if os.path.isdir('trainingDigits') and os.path.isdir('testDigits'):
        return

    if os.path.isdir('testDigits'):
        shutil.rmtree('testDigits')

    if os.path.isdir('trainingDigits'):
        shutil.rmtree('trainingDigits')

    with zipfile.ZipFile('digits.zip') as z:
        z.extractall()


def test_handwrite_class():
    unzip_data()
    labels = []
    # Collect training data
    traing_filenames = glob.glob(os.path.join('trainingDigits', '*.txt'))
    count = len(traing_filenames)
    training_matrix = numpy.zeros((count, SIZE * SIZE))
    for i, filename in enumerate(traing_filenames):
        class_label = get_class_label(filename)
        labels.append(class_label)
        training_matrix[i, :] = read_image(filename)

    # Test
    test_filenames = glob.glob(os.path.join('testDigits', '*.txt'))
    errors = 0
    count = len(test_filenames)
    for i, filename in enumerate(test_filenames):
        class_label = get_class_label(filename)
        test_vector = read_image(filename)
        result = classify(test_vector, training_matrix,
                          labels, 3)
        actual = result
        expected = class_label
        print(filename)
        print("The classifier came back with: {actual}, "
              "the real answer is: {expected}".format(
                  expected=expected,
                  actual=actual,
              ))
        if actual != expected:
            errors += 1

    print()
    print("The total number of errors: {}".format(errors))
    print()
    print("The total error rate: {}".format(errors / count))


def main():
    test_handwrite_class()


if __name__ == '__main__':
    main()
