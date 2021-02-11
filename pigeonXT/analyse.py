def get_number_of_annotations_per_label(annotations, labels):
    """
    Gets the number of annotations per label
    :param annotations: a list of labelled (annotated) examples
    :param labels: a list of labels used when annotated
    :return: annotations_count: a dictionary of counts per label
    """
    annotations_count = {}
    chosen_labels_list = [annotated_labels for annotated_labels in annotations.values()]

    for label in labels:
        annotations_count[label] = sum(label in chosen_labels for chosen_labels in chosen_labels_list)

    return annotations_count
