import enum
import typing

import pydantic
import rich
import torch
import torch.nn as nn
import torch.optim as optim
import typer
from rich.progress import track
from torch.utils.data import DataLoader


class AvailableDataset(str, enum.Enum):
    Mnist = "MNIST"
    CifarTen = "CIFAR10"


class Metrics(pydantic.BaseModel):
    training_loss: pydantic.conlist(item_type=float)
    training_accuracy: pydantic.conlist(item_type=float)
    testing_loss: pydantic.conlist(item_type=float)
    testing_accuracy: pydantic.conlist(item_type=float)


class Trainer(pydantic.BaseModel):
    model: nn.Module
    model_name: pydantic.constr(min_length=5)
    data_loader: DataLoader
    manual_seed: pydantic.conint(gt=0)
    loss_function = nn.CrossEntropyLoss
    learning_rate: float = 1e-3
    epochs: int = 15

    class Config:
        arbitrary_types_allowed = True


def run_train_loop(
    data_loader: DataLoader,
    model: nn.Module,
    loss_function: nn.CrossEntropyLoss,
    optimizer: optim.Adam,
) -> pydantic.conlist(item_type=float):
    """
    Train the model on the entire training dataset for a single epoch and report
    back the losses.
    """
    size, losses = len(data_loader.dataset), []
    for batch, (X, y) in enumerate(data_loader):
        pred = model(X)
        loss = loss_function(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            print(f"loss: {loss.item():>7f}  [{batch * len(X):>5d}/{size:>5d}]")
        losses.append(loss.item())
    return losses


def run_evaluation_loop(
    data_loader: DataLoader,
    model: nn.Module,
    loss_function: nn.CrossEntropyLoss,
) -> pydantic.conlist(item_type=float, max_items=2, min_items=2):
    """
    Evaluate the model on the evaluation data, provided by the data-loader, and
    return the `[accuracy, evaluation-loss]`
    """
    size, num_batches = len(data_loader.dataset), len(data_loader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in data_loader:
            pred = model(X)
            test_loss += loss_function(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size

    rich.print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )
    return [100 * correct, test_loss]


def save_model(
    model: nn.Module,
    save_dir: pydantic.DirectoryPath,
    model_name: pydantic.constr(strip_whitespace=True, min_length=5),
) -> None:
    """
    Save the trained model to disk
    """
    torch.save(model.state_dict(), f"{save_dir}{model_name}.pth")
    torch.print(f"Finished training the {model_name} Model!")


def save_metrics(
    metrics: Metrics,
    save_dir: pydantic.DirectoryPath,
    model_name: pydantic.constr(strip_whitespace=True, min_length=5),
) -> None:
    """
    Save the losses and accuracies to disk
    """
    pass


def train(trainer: Trainer) -> Metrics:
    generator: torch.Generator = torch.Generator()
    generator.manual_seed(trainer.manual_seed)

    metrics: typing.List[Metrics] = []
    for epoch_id in track(range(trainer.epochs), description="Training..."):
        rich.print(f"Epoch {epoch_id+1}")
        optimizer = optim.Adam(trainer.model.parameters(), lr=trainer.learning_rate)
        _training_metrics = run_train_loop(
            data_loader=trainer.data_loader,
            model=trainer.model,
            loss_function=trainer.loss_function,
            optimizer=optimizer,
        )
        _evaluation_metrics = run_evaluation_loop(
            data_loader=trainer.data_loader,
            model=trainer.model,
            loss_function=trainer.loss_function,
        )

    save_metrics(metrics=metrics)
    save_model()


def mendel(model_name: str, data_dir: str, masked_class: str):
    rich.print("Mendel: A new Perspective on Neural Network Interpretability")


if __name__ == "__main__":
    typer.run(mendel)
