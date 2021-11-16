import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import torch.nn as nn
from dataset import MnistAnomaly
from Model import ModelPCA
from Model import ModelIsoForest
from Model import ModelOCSVM
from Model import ModelAutoencoder
import sys
from typing import Optional
from pathlib import Path
import pandas as pd


def eval_algo(algo: str, digit: Optional[int]) -> None:

    if digit is None:
        digits = list(range(10))
    else:
        digits = [digit]

    aucs = []
    names = []

    for i in digits:
        print(f" this is the {i}th anormal category")
        abnormal_digit = [i]
        train_set = MnistAnomaly(
            root=".",
            train=True,
            transform=transforms.ToTensor(),
            anomaly_categories=abnormal_digit,
        )

        # train model
        train_loader = DataLoader(train_set, batch_size=len(train_set))
        x, _ = next(iter(train_loader))

        if algo.lower() == "pca":

            model = ModelPCA(num_component=18)

        elif algo.lower() == "autoencodercnn":

            model = ModelAutoencoder(nn.MSELoss(), 20)
        elif algo.lower() == "ocsvm":

            model = ModelOCSVM()
        else:
            model = ModelIsoForest()

        print(f"run {algo.upper()}")

        model.compress(x)

        # test model
        test_set = MnistAnomaly(
            root=".",
            train=False,
            transform=transforms.ToTensor(),
            anomaly_categories=abnormal_digit,
        )
        test_loader = DataLoader(test_set, batch_size=len(test_set))
        x_test, y_test = next(iter(test_loader))

        # compute score
        with torch.no_grad():
            score_test = model.compress(x_test)

        # compute rocauc
        roc_auc = roc_auc_score(y_test, score_test)
        aucs.append(roc_auc)
        names.append(f"AUC_{i}")

        print(f"auc={roc_auc:.03f}")

    print("roc_auc per digit:")
    print(["{:0.3f} ".format(auc) for auc in aucs])
    avg_auc = sum(aucs) / len(aucs)
    print(f"average roc_auc: {avg_auc:.03f}")

    dict_results = dict(zip(names, aucs))
    dict_results["algo"] = algo
    dict_results["avg"] = avg_auc

    if digit is None:

        res = "result.csv"

        rows = []

        if Path(res).is_file():

            frame = pd.read_csv(res)

            if not frame.query(f"algo == '{algo}'").empty:
                answer = input(f"overwrite results from {algo}: y / n ")
                if answer.lower().startswith("n"):
                    return

            rows = frame.query(f"algo != '{algo}'").to_dict("records")

        pd.DataFrame(rows + [dict_results]).to_csv(res, index=False)


def main():

    if len(sys.argv) == 3:
        algo = sys.argv[1]
        digit = int(sys.argv[2])
    elif len(sys.argv) == 2:
        algo = sys.argv[1]
        digit = None
    else:
        print(f"{sys.argv[0]} algo [digit]")
        exit(1)

    assert algo.lower() in ["pca", "autoencodercnn", "isoforest", "ocsvm"]

    if digit is not None:
        assert -1 < digit < 10

    eval_algo(algo, digit)


if __name__ == "__main__":

    main()
