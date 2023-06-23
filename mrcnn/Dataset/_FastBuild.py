import os
from ._CustomDataset import CustomDataset


def load_images_dataset(
        dataset_path: str,
        dataset_type: str,
        split_train_test: float = 0.
):
    dataset_path = os.path.join(dataset_path, dataset_type)
    annotation_path = search_annotations_file(dataset_path)

    dataset_train = CustomDataset()
    dataset_train.load_custom(
        annotation_path, dataset_path,
        split_train_test=split_train_test
    )
    dataset_train.prepare()
    return dataset_train


def search_annotations_file(dataset_path: str) -> str:
    for file in os.listdir(dataset_path):
        if file.endswith('.json'):
            return os.path.join(dataset_path, file)

    raise Exception('No annotations found in dataset {}'.format(dataset_path))
