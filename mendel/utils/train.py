import typer
import pydantic
from torch.utils.data import DataLoader
import torch.nn as nn
from rich import print

class DataLoaders(pydantic.BaseModel):
    """Class for keeping track of our data."""
    # The data to train on. Depending on the desired model, this can be the full
    # data, or a dataset with only a single class removed (sans-x)
    training: DataLoader

    # The first dataset we evaluate on. This should have the 
    # same classes as the `training data`
    in_distribution_eval: DataLoader

    # The second dataset we evaluate on. If `training` is missing a class,
    # then this is the dataset with all teh classes and vice versa
    out_of_distribution_eval: DataLoader

    # This is the final dataset we evaluate on. This dataset comprises
    # of only a single class -- that which differentiates `in_distribution_eval`
    # form `out_of_distribution_eval`
    in_out_delta_eval: DataLoader


class TrainingMetrics(pydantic.BaseModel):
    pass


class Trainer:
    def __init__(self,
        model: nn.Module,
        loss_fn = nn.CrossEntropyLoss,
    ) -> None:
        pass

    def train(self) -> TrainingMetrics:
        pass

