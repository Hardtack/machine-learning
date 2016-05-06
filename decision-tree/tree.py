import math
import operator


def shannon_entropy(dataset):
    count = len(dataset)
    label_counts = {}

    # Collect labels
    for feature in dataset:
        label = feature[-1]
        label_counts[label] = label_counts.get(label, 0) + 1

    # Calculate entropy
    entropy = 0.0
    for key, value in label_counts.items():
        probability = value / count
        entropy -= probability * math.log(probability, 2)
    return entropy


def split_dataset(dataset, axis, value):
    return_dataset = []
    for feature in dataset:
        if feature[axis] == value:
            reduced = feature[:axis]
            reduced.extend(feature[axis + 1:])
            return_dataset.append(reduced)
    return return_dataset


def choose_best_feature(dataset):
    feature_count = len(dataset[0]) - 1
    base_entropy = shannon_entropy(dataset)
    best_info_gain = 0.0
    best_feature = -1

    for i in range(feature_count):
        values = set(row[i] for row in dataset)
        entropy = 0.0

        for value in values:
            sub_dataset = split_dataset(dataset, i, value)
            probability = len(sub_dataset) / float(len(dataset))
            entropy += probability * shannon_entropy(sub_dataset)

        info_gain = base_entropy - entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def majority(classes):
    counts = {}
    for vote in classes:
        counts[vote] = counts.get(vote, 0) + 1
    sorted_items = sorted(counts.items(),
                          key=operator.itemgetter(1),
                          reverse=True)
    return sorted_items[0][0]


def make_tree(dataset, labels):
    classes = [row[-1] for row in dataset]
    if len(set(classes)) == 1:
        return classes[0]
    if len(dataset[0]) == 1:
        return majority(classes)
    best_feature = choose_best_feature(dataset)
    best_label = labels[best_feature]

    tree = {
        best_label: {}
    }
    labels = labels.copy()
    labels.pop(best_feature)

    values = set(row[best_feature] for row in dataset)
    for value in values:
        tree[best_label][value] = make_tree(
            split_dataset(dataset, best_feature, value),
            labels
        )
    return tree


def create_dataset():
    dataset = [[1, 1, 'maybe'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataset, labels


def main():
    dataset, labels = create_dataset()
    print(make_tree(dataset, labels))


if __name__ == '__main__':
    main()
