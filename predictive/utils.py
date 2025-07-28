import pathlib
from typing import NamedTuple

from pandas import DataFrame


class SplitDataset(NamedTuple):
    train: DataFrame
    valid: DataFrame


def split_train_valid(
    features: DataFrame,
    target: DataFrame,
) -> tuple[SplitDataset, SplitDataset]:
    """Split a dataframe given the pre-defined train and valid splits."""
    return (
        _split_dataset(features),
        _split_dataset(target),
    )

def _split_dataset(dataset: DataFrame, valid_cutoff_year: int = 2017) -> SplitDataset:
    valid = dataset[dataset['Order Date'].dt.year >= valid_cutoff_year]
    train = dataset[dataset['Order Date'].dt.year < valid_cutoff_year]

    return SplitDataset(train=train, valid=valid)


