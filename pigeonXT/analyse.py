"""Helper functions to analyse annotations."""


def get_number_of_annotations_per_label(
    annotations: dict,
    labels: list,
) -> dict[str, int]:
    """
    Get the number of annotations per label.

    # older function, only works on dicts

    Parameters
    ----------
    annotations : Union[dict, DataFrame]
        list of labelled (annotated) examples
    labels : list
        list of labels used when annotated

    Returns
    -------
    :return: annotations_count: a dictionary of counts per label
    """
    annotations_count = {}
    chosen_labels_list = [annotated_labels for annotated_labels in annotations.values()]

    for label in labels:
        annotations_count[label] = sum(
            label in chosen_labels for chosen_labels in chosen_labels_list
        )

    return annotations_count
