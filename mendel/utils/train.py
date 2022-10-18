import pydantic
import rich
import torch
import torch.nn as nn
import torch.optim as optim
from rich.progress import track
from torch.utils.data import DataLoader


class Metrics(pydantic.BaseModel):
    training_loss: pydantic.conlist(item_type=float)
    training_accuracy: pydantic.conlist(item_type=float)
    testing_loss: pydantic.conlist(item_type=float)
    testing_accuracy: pydantic.conlist(item_type=float)


class Trainer(pydantic.BaseModel):
    model: nn.Module
    model_name: pydantic.constr(min_length=5)
    data_loader: DataLoader
    manual_seed: pydantic.conint(min=13)
    loss_function = nn.CrossEntropyLoss
    learning_rate: float = 1e-3
    epochs: int = 15


def run_train_loop():
    pass


def run_evaluation_loop():
    pass


def train(trainer: Trainer) -> Metrics:
    generator: torch.Generator = torch.Generator()
    generator.manual_seed(trainer.manual_seed)

    for epoch_id in track(range(trainer.epochs), description="Training..."):
        rich.print(f"Epoch {epoch_id+1}")
        run_train_loop()
        run_evaluation_loop()
