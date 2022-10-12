__author__ = "Bo Pang"
__copyright__ = "Copyright 2022, IRP Project"
__credits__ = ["Bo Pang"]
__license__ = "Apache 2.0"
__version__ = "1.0"
__email__ = "bo.pang21@imperial.ac.uk"

"""
This file stores functions related to the processing of data sets
"""

import numpy as np


def normalize(data, scale_range=None):
    """
    Min Max scale for a batch of data

    Args:
        data: The data to be processed
        scale_range: Scaling range

    Returns:
        The data normalised to scale_range
    """
    if scale_range is None:
        scale_range = [0, 1]
    data_max, data_min = np.max(data), np.min(data)
    if np.isclose(data_max, data_min):
        return data
    data_std = (data - data_min) / (data_max - data_min)
    return data_std * (max(scale_range) - min(scale_range)) + min(scale_range)


def create_shifted_frames(data):
    """
    Helper function for splitting a single piece
    of data into source and target

    Args:
        data: Individual data to be split

    Returns:
        Source data x, Target data y

    """
    x = data[..., :-1]
    y = data[..., -1:]
    return x, y


def get_big_fire(dataset):
    """
    Filter the dataset for fires with a variation of >1000
    between day 2 and final

    Args:
        dataset: Data set to be filtered

    Returns:
        Filtered data set
    """
    """
    Find big fire
    """
    init_fire = dataset[..., 2]
    final_fire = dataset[..., -1]
    sum = [np.sum(final_fire[i] - init_fire[i])
           for i in range(init_fire.shape[0])]
    sum = np.array(sum)
    indice = np.where((sum) > 1000)
    return dataset[indice]


def load_dataset(fpath):
    """
    Reads data sets from files and normalises geographical
    and meteorological information.

    Args:
        fpath: Path of the data set in the file system

    Returns:
        Entire data set
    """
    dataset = np.load(fpath)
    # Normalization
    for j in range(dataset.shape[-1]):
        if 2 < j < 14:
            dataset[:, :, :, j] = normalize(dataset[:, :, :, j])
    return dataset


def split_dataset(dataset):
    """
    Split the dataset into training, validation
    and test sets in a ratio of 8:1:1

    Args:
        dataset: Original full dataset

    Returns:
        Training set, validation set, test set
    """
    # Split into train and validation sets using
    # indexing to optimize memory.
    indexes = np.arange(dataset.shape[0])
    # np.random.seed(seed=4) # setseed
    # np.random.seed(seed=5)
    # np.random.shuffle(indexes)
    train_index = indexes[: int(0.8 * dataset.shape[0])]
    val_index = indexes[
                int(0.8 * dataset.shape[0]):int(0.9 * dataset.shape[0])]
    test_index = indexes[int(0.9 * dataset.shape[0]):]

    train_dataset = dataset[train_index]
    val_dataset = dataset[val_index]
    test_dataset = dataset[test_index]
    return train_dataset, val_dataset, test_dataset


def setup_dataset(data_path):
    """
    Read the dataset, normalise it. After dividing the dataset,
    data augmentation is performed on the training set, and
    finally the dataset is split into Source and Target.

    Args:
        data_path: Path of the data set in the file system

    Returns:
        full dataset, train_dataset, val_dataset, test_dataset,
        x_train, y_train, x_val, y_val, x_test, y_test

    """
    dataset = load_dataset(data_path)
    train_dataset, val_dataset, test_dataset = split_dataset(dataset)

    train_dataset = get_big_fire(train_dataset)
    # Data Augmentation
    train_dataset_roted = np.rot90(train_dataset, k=1, axes=(1, 2))
    train_dataset = np.concatenate(
        (train_dataset, train_dataset_roted), axis=0)
    train_dataset_roted = np.rot90(train_dataset_roted, k=1, axes=(1, 2))
    train_dataset = np.concatenate(
        (train_dataset, train_dataset_roted), axis=0)
    train_dataset_roted = np.rot90(train_dataset_roted, k=1, axes=(1, 2))
    train_dataset = np.concatenate(
        (train_dataset, train_dataset_roted), axis=0)

    x_train, y_train = create_shifted_frames(train_dataset)
    x_val, y_val = create_shifted_frames(val_dataset)
    x_test, y_test = create_shifted_frames(test_dataset)

    return dataset, train_dataset, val_dataset, test_dataset, x_train, y_train, x_val, y_val, x_test, y_test
