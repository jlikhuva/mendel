import pandas as pd
import pydantic
import torch
from torch.utils.data import DataLoader, Dataset


class MnistDataset(Dataset):
    def __init__(
        self, data_file_location: pydantic.constr(strip_whitespace=True, min_length=1)
    ) -> None:
        dataframe: pd.DataFrame = pd.read_csv(data_file_location, header=None)
        Y, X = dataframe.loc[:, 0], dataframe.loc[:, 1:]
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(list(self.X.loc[idx, :]), dtype=torch.float32)
        y = torch.tensor(self.Y[idx])
        return x, y


def load_data(filename: str, root_dir: str) -> DataLoader:
    if not filename:
        return None
    data_file_location = f"{root_dir}{filename}"
    loader = DataLoader(
        MNISTDataset(data_file_location=data_file_location), batch_size=64
    )
    return loader
