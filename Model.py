import torch
import torch.nn as nn
from typing import Callable, Optional
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.decomposition import PCA
from autoencoder import autoencoderCNN
from train import BaseTrainer
from torch.utils.data import Dataset
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


class Model:
    """
    Generate the model of type Representation Learning
    """

    # ===================================================================
    def __init__(self) -> None:
        self.model = None

    # ==================================================================
    def score(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("infer needs to be defined")

    # ===================================================================
    def train(self, x: np.ndarray) -> None:
        raise NotImplementedError("train needs to be defined")

    # ===================================================================
    def compress(self, z: torch.tensor) -> float:

        x = z.cpu().detach().numpy()

        if self.model is None:

            self.train(x)

        return self.score(x)


class ModelIsoForest(Model):
    # ===================================================================
    def __init__(
        self, n_estimators: Optional[int] = 100, random_state: int = 0
    ) -> None:
        self.model = None
        self.random_state = random_state

    # ===================================================================
    def score(self, x: np.ndarray) -> np.ndarray:
        assert self.model is not None, "The ISoForest model should not be None"
        x = x.reshape(len(x), -1)
        return self.model.score_samples(x)

    # ===================================================================
    def train(self, x: np.ndarray) -> None:
        self.model = IsolationForest(n_estimators=100, random_state=self.random_state)
        x = x.reshape(len(x), -1)
        self.model.fit(x)


class ModelOCSVM(Model):
    """
    Better to be applied on a reduced-dimensional samples as its computation
    time is very very long
    """

    # ===================================================================
    def __init__(
        self,
        kernel: str = "rbf",
        nu: int = 0.5,
        gamma: str = "scale",
        number_component: Optional[int] = 1000,
    ) -> None:
        self.model = None
        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma
        # take number_component samples into training
        self.number_component = number_component

    # ===================================================================
    def score(self, x: np.ndarray) -> np.ndarray:
        assert self.model is not None, "The ocsvm model should not be None"
        x = x.reshape(len(x), -1)
        return self.model.score_samples(x)

    # ===================================================================
    def train(self, x: np.ndarray) -> None:
        self.model = OneClassSVM(self.kernel, self.nu, self.gamma)

        x = x.reshape(len(x), -1)
        x = x[0 : self.number_component + 1, :]
        self.model.fit(x)


class ModelRecon(Model):
    """
    Generate the model based on the reconsturction error (measured by L2 norm)
    """

    # ===================================================================
    def __init__(self) -> None:
        super().__init__()

    # ===================================================================
    def infer(self, x: np.ndarray) -> None:
        raise NotImplementedError("train needs to be defined")

    # ===================================================================
    def score(self, x: np.ndarray) -> np.ndarray:
        x = x.reshape(len(x), -1)
        x_res = x - self.infer(x)
        return -np.power(x_res, 2).sum(axis=1)


class ModelAutoencoder(ModelRecon):
    # ===================================================================
    def __init__(self, loss: Callable, num_epochs: int) -> None:
        super().__init__()
        self.loss = loss
        self.num_epochs = num_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===================================================================
    def train(self, x: np.ndarray) -> None:
        self.model = autoencoderCNN()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.trainer = BaseTrainer(self.model, optimizer, self.loss, self.num_epochs)
        dataloader = self.build_dataloader(x)
        for epoch in range(self.num_epochs):
            self.trainer.train(epoch, dataloader)

    # ===================================================================
    @staticmethod
    def build_dataloader(x: np.ndarray) -> DataLoader:
        class myDataset(Dataset):
            def __init__(
                self, data: torch.tensor, transform: Optional[Callable] = None
            ) -> None:

                self.data = data

            def __len__(self) -> int:

                return len(self.data)

            def __getitem__(self, k: int) -> np.ndarray:

                return self.data[k]

        return DataLoader(
            myDataset(torch.from_numpy(x), transform=ModelAutoencoder.transform()),
            batch_size=128,
        )

    # ===================================================================
    @staticmethod
    def transform_compose() -> Callable:
        return transforms.Compose(
            [
                transforms.RandomAffine(
                    degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)
                ),
            ]
        )

    # ===================================================================
    @staticmethod
    def transform() -> Callable:
        return lambda x: x

    # ===================================================================
    def infer(self, z: np.ndarray) -> np.ndarray:
        assert self.model is not None, "Autoencoder should not be none"
        self.model.eval()

        x = torch.from_numpy(z).reshape((-1, 1, 28, 28))

        x = self.transform()(x.to(self.device))

        res_ten = self.model(x)
        res_np = res_ten.cpu().detach().numpy()

        res_np = res_np * 0.5 + 0.5
        return res_np.reshape(len(x), -1)


class ModelPCA(ModelRecon):
    # ===================================================================
    def __init__(self, num_component: int) -> None:
        super().__init__()
        self.num_component = num_component

    # ===================================================================
    def train(self, x: np.ndarray) -> None:
        x = x.reshape(len(x), -1)
        self.model = PCA(self.num_component)
        self.model.fit(x)

    # ===================================================================
    def infer(self, x: np.ndarray) -> np.ndarray:
        assert self.model is not None, "PCA should not be none"
        x = x.reshape(len(x), -1)

        return self.model.inverse_transform(self.model.transform(x))
