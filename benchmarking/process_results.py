from pathlib import Path
from pprint import pprint
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding
from torchvision.transforms import Compose, Lambda, Resize, ToTensor
import torch
import torch.nn as nn
from torchvision.models.squeezenet import SqueezeNet
from sklearn.metrics import roc_curve, auc, RocCurveDisplay, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from evaluate import load as load_metric

"""Fil for å behandle resultater."""

# df = pd.read_csv("./benchmarking_results/imdb-orgfedavg/nc3/run1/cache.csv",
#                  sep=",")


def combine_containers(df, num_clients):
    """Combine results from containers for supernodes/links and client/serverapps
    e.g., supernode-0 and clientapp-0 are combined into client-0."""
    
    combined_containers_df = pd.DataFrame(dict())

    # Handle server.
    combined_containers_df['server'] = df['serverapp'] + df['superlink']
    # Handle clients.
    if "laptop-clientapp-0" in df.keys():
        combined_containers_df['laptop-client-1'] = (df['laptop-clientapp-0'] 
                                                     + df['laptop-supernode-0'])
    if "laptop-clientapp-1" in df.keys():
        combined_containers_df['laptop-client-2'] = (df['laptop-clientapp-1'] 
                                                     + df['laptop-supernode-1'])
    if "laptop-clientapp-2" in df.keys():
        combined_containers_df['laptop-client-3'] = (df['laptop-clientapp-2'] 
                                                     + df['laptop-supernode-2'])
    if "laptop-clientapp-3" in df.keys():
        combined_containers_df['laptop-client-4'] = (df['laptop-clientapp-3'] 
                                                     + df['laptop-supernode-3'])
    if "desktop-clientapp-0" in df.keys():
        combined_containers_df['desktop-client-1'] = (df['desktop-clientapp-0'] 
                                                     + df['desktop-supernode-0'])
    if "desktop-clientapp-1" in df.keys():
        combined_containers_df['desktop-client-2'] = (df['desktop-clientapp-1'] 
                                                     + df['desktop-supernode-1'])
    if "desktop-clientapp-2" in df.keys():
        combined_containers_df['desktop-client-3'] = (df['desktop-clientapp-2'] 
                                                     + df['desktop-supernode-2'])
    if "desktop-clientapp-3" in df.keys():
        combined_containers_df['desktop-client-4'] = (df['desktop-clientapp-3'] 
                                                     + df['desktop-supernode-3'])
    
    return combined_containers_df

def load_imdb(partition_id: int, num_partitions: int, model_name: str):
    """Load IMDB data (training and eval)"""
    partitioner = IidPartitioner(num_partitions=num_partitions)
    fds = FederatedDataset(
        dataset="stanfordnlp/imdb",
        partitioners={"test": partitioner},
    )
    partition = fds.load_partition(partition_id, "test")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], truncation=True, add_special_tokens=True, max_length=512
        )

    partition = partition.map(tokenize_function, batched=True)
    partition = partition.remove_columns("text")
    partition = partition.rename_column("label", "labels")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    testloader = DataLoader(
        partition, batch_size=32, collate_fn=data_collator
    )

    return testloader

def load_mnist_squeezenet(
    partition_id,
    num_partitions,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    partitioner = IidPartitioner(num_partitions=num_partitions)
    fds = FederatedDataset(
        dataset="ylecun/mnist",
        partitioners={"test": partitioner},
        trust_remote_code=True,
    )
    partition = fds.load_partition(partition_id, "test")

    # Resize and repeat channels to use MNIST, which have grayscale images,
    # with squeezenet, which expects 3 channels.
    # Ref: https://discuss.pytorch.org/t/fine-tuning-squeezenet-for-mnist-dataset/31221/2
    pytorch_transforms = Compose(
        [Resize(224), ToTensor(), Lambda(lambda x: x.expand(3, -1, -1))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
        return batch

    def collate_fn(batch):
        """Change the dictionary to tuple to keep the exact dataloader behavior."""
        images = [item["image"] for item in batch]
        labels = [item["label"] for item in batch]

        images_tensor = torch.stack(images)
        labels_tensor = torch.tensor(labels)

        return images_tensor, labels_tensor

    partition = partition.with_transform(apply_transforms)
    # Use whole dataset.
    testloader = DataLoader(
        partition, batch_size=32, collate_fn=collate_fn, num_workers=1
    )
    return testloader

def load_mnist_logreg(
    partition_id,
    num_partitions,
    scale="classic"
) -> tuple[DataLoader, DataLoader]:
    # Only initialize `FederatedDataset` once
    partitioner = IidPartitioner(num_partitions=num_partitions)
    fds = FederatedDataset(
        dataset="ylecun/mnist",
        partitioners={"test": partitioner},
        trust_remote_code=True,
    )
    partition = fds.load_partition(partition_id, "test")

    # Resize and repeat channels to use MNIST, which have grayscale images,
    # with squeezenet, which expects 3 channels.
    # Ref: https://discuss.pytorch.org/t/fine-tuning-squeezenet-for-mnist-dataset/31221/2
    # For all types, convert from 3d-structure to 1d-structure.
    if scale == "classic":
        # Only convert to tensor, keeping form 1*28*28.
        pytorch_transforms = Compose(
            [ToTensor(), Lambda(lambda x: x.ravel())]
        )
    elif scale == "larger":
        # Enlargen the images and convert to tensor, getting 1*224*244.
        pytorch_transforms = Compose(
            [Resize(224), ToTensor(), Lambda(lambda x: x.ravel())]
        )
    elif scale == "rgb":
        # Enlargen the images, convert to RBG (3 channels) and convert to tensor,
        # getting 3*224*244.
        pytorch_transforms = Compose(
            [Resize(224), ToTensor(), Lambda(lambda x: x.expand(3, -1, -1)), Lambda(lambda x: x.ravel())]
        )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
        return batch

    def collate_fn(batch):
        """Change the dictionary to tuple to keep the exact dataloader behavior."""
        images = [item["image"] for item in batch]
        labels = [item["label"] for item in batch]

        images_tensor = torch.stack(images)
        labels_tensor = torch.tensor(labels)

        return images_tensor, labels_tensor

    partition = partition.with_transform(apply_transforms)
    # Use whole dataset.
    testloader = DataLoader(
        partition, batch_size=32, collate_fn=collate_fn, num_workers=1
    )
    return testloader

class LogisticRegression(nn.Module):
    """Take n_datapoints x n_features matrix as input.
    Multiply with n_features x n_classes weights matrix to get
    n_datapoints x n_classes. Apply softmax to this result to get
    predictions."""
    def __init__(self, n_features, n_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_features, n_classes)

    def forward(self, x):
        # Tror dim = -1 er riktig for å ta av hver rad.
        y_predicted = torch.softmax(self.linear(x), -1)
        return y_predicted

def plot_ROC_AUC(path):
    """Må endre paths her. Paths må være til benchmarking_results/..."""
    if 'bert' in path:
        total_path = path+f'/run{1}/bert-tmp-r3/'
        print(type(total_path))
        net = AutoModelForSequenceClassification.from_pretrained(
                total_path, num_labels=2, local_files_only=True
            )
        # trainloader, testloader = load_imdb(0, 1, total_path) # Bruker hele datasettet som utgangspunkt.
        testloader = load_imdb(0, 1, "prajjwal1/bert-tiny") # Bruker hele datasettet som utgangspunkt.

        # Predict hele testsettet med modellen.
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.eval()
        store = []
        for batch in testloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = net(**batch)
            logits = outputs.logits
            # predictions = torch.argmax(logits, dim=-1)
            probs = torch.sigmoid(logits)
            store += probs.tolist()
        predict_probas = np.asarray(store)
        true_labels = np.asarray([testloader.dataset[i]['labels'] for i in range(len(testloader.dataset))])

        # Compute values for curve.
        fpr, tpr, thresholds = roc_curve(true_labels, predict_probas[:, 1])
        roc_auc = auc(fpr, tpr)
        # Create subplots.
        fig, ax = plt.subplots(figsize=(6, 6))
        # Plot curves
        ax.plot(fpr, tpr, lw=2, label=f'ROC trained model (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='gray', lw=2, label="Chance level (AUC = 0.5)", linestyle='--')  # random guess line
        # Misc things.
        ax.set(xlabel="False Positive Rate",
            ylabel="True Positive Rate",)
        ax.legend(loc="lower right")
        plt.show()

    elif 'squeezenet' in path:
        model = SqueezeNet("1_1", 10)
        total_path = path+f'/run{1}/squeezenet-tmp-r3'
        storage = os.path.abspath(total_path)
        model.load_state_dict(torch.load(storage, weights_only=True))
        model.eval()

        testloader = load_mnist_squeezenet(0, 1)

        # Predict hele testsettet med modellen.
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        store = []
        for batch in testloader:
            images = batch[0].to(device)
            labels = batch[1].to(device)
            with torch.no_grad():
                outputs = model(images)
            # predictions = torch.argmax(logits, dim=-1)
            probs = torch.softmax(outputs, dim=1)
            store += probs.tolist()
        predict_probas = np.asarray(store)
        true_labels = np.asarray([testloader.dataset[i]['label'] for i in range(len(testloader.dataset))])
        # print(predict_probas.shape)

        train_labels = np.asarray([testloader.dataset[i]['label'] for i in range(len(testloader.dataset))])
        labelbinarizer = LabelBinarizer().fit(train_labels)
        true_labels = labelbinarizer.transform(true_labels)
        # print(predict_probas.shape)
        # print(true_labels_onehot.shape)

        # Create subplots.
        fig, ax = plt.subplots(figsize=(6, 6))
        fpr, tpr, roc_auc = dict(), dict(), dict()
        # Create subplots.
        ax.plot([0, 1], [0, 1], color='gray', lw=2, label="Chance level (AUC = 0.5)", linestyle='--')  # random guess line
        for i in range(predict_probas.shape[1]):
            # Compute values for curve.
            fpr[f"class {i+1}"], tpr[f"class {i+1}"], thresholds = roc_curve(true_labels[:, i], predict_probas[:, i])
            roc_auc[f"class {i+1}"] = auc(fpr[f"class {i+1}"], tpr[f"class {i+1}"])
            # Plot curves
            ax.plot(fpr[f"class {i+1}"], tpr[f"class {i+1}"], lw=2, label=f'Class {i+1} (AUC = {roc_auc[f"class {i+1}"]:.4f})')

        # micro average.
        fpr["micro average"], tpr["micro average"], threshold = roc_curve(true_labels.ravel(), predict_probas.ravel())
        roc_auc["micro average"] = auc(fpr["micro average"], tpr["micro average"])
        ax.plot(fpr["micro average"], tpr["micro average"], lw=2, label=f'micro average (AUC = {roc_auc["micro average"]:.4f})')

        # macro average.
        fpr_grid = np.linspace(0.0, 1.0, 1000)
        # Interpolate all ROC curves at these points
        mean_tpr = np.zeros_like(fpr_grid)
        for i in range(10):
            mean_tpr += np.interp(fpr_grid, fpr[f"class {i+1}"], tpr[f"class {i+1}"])  # linear interpolation
        # Average it and compute AUC
        mean_tpr /= 10

        fpr["macro average"] = fpr_grid
        tpr["macro average"] = mean_tpr
        roc_auc["macro average"] = auc(fpr["macro average"], tpr["macro average"])
        ax.plot(fpr["macro average"], tpr["macro average"], lw=2, label=f'macro average (AUC = {roc_auc["macro average"]:.4f})')

        
        # Misc things.
        ax.set(xlabel="False Positive Rate",
            ylabel="True Positive Rate",)
        ax.legend(loc="lower right")
        plt.show()

    elif 'logreg' in path:
        scale = "classic"
        if scale == "classic":
            model = LogisticRegression(28*28, 10)
        elif scale == "larger":
            model = LogisticRegression(224*224, 10)
        elif scale == "rgb":
            model = LogisticRegression(3*224*224, 10)
        total_path = path+f'/run{1}/logreg-tmp-r3'
        storage = os.path.abspath(total_path)
        model.load_state_dict(torch.load(storage, weights_only=True))
        model.eval()

        testloader = load_mnist_logreg(0, 1)

        # Predict hele testsettet med modellen.
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        store = []
        for batch in testloader:
            images = batch[0].to(device)
            labels = batch[1].to(device)
            with torch.no_grad():
                outputs = model(images)
            # predictions = torch.argmax(logits, dim=-1)
            probs = torch.softmax(outputs, dim=1)
            store += probs.tolist()
        predict_probas = np.asarray(store)
        true_labels = np.asarray([testloader.dataset[i]['label'] for i in range(len(testloader.dataset))])
        # print(predict_probas.shape)

        train_labels = np.asarray([testloader.dataset[i]['label'] for i in range(len(testloader.dataset))])
        labelbinarizer = LabelBinarizer().fit(train_labels)
        true_labels = labelbinarizer.transform(true_labels)
        # print(predict_probas.shape)
        # print(true_labels_onehot.shape)

        # Create subplots.
        fig, ax = plt.subplots(figsize=(6, 6))
        fpr, tpr, roc_auc = dict(), dict(), dict()
        # Create subplots.
        ax.plot([0, 1], [0, 1], color='gray', lw=2, label="Chance level (AUC = 0.5)", linestyle='--')  # random guess line
        for i in range(predict_probas.shape[1]):
            # Compute values for curve.
            fpr[f"class {i+1}"], tpr[f"class {i+1}"], thresholds = roc_curve(true_labels[:, i], predict_probas[:, i])
            roc_auc[f"class {i+1}"] = auc(fpr[f"class {i+1}"], tpr[f"class {i+1}"])
            # Plot curves
            ax.plot(fpr[f"class {i+1}"], tpr[f"class {i+1}"], lw=2, label=f'Class {i+1} (AUC = {roc_auc[f"class {i+1}"]:.4f})')

        # micro average.
        fpr["micro average"], tpr["micro average"], threshold = roc_curve(true_labels.ravel(), predict_probas.ravel())
        roc_auc["micro average"] = auc(fpr["micro average"], tpr["micro average"])
        ax.plot(fpr["micro average"], tpr["micro average"], lw=2, label=f'micro average (AUC = {roc_auc["micro average"]:.4f})')

        # macro average.
        fpr_grid = np.linspace(0.0, 1.0, 1000)
        # Interpolate all ROC curves at these points
        mean_tpr = np.zeros_like(fpr_grid)
        for i in range(10):
            mean_tpr += np.interp(fpr_grid, fpr[f"class {i+1}"], tpr[f"class {i+1}"])  # linear interpolation
        # Average it and compute AUC
        mean_tpr /= 10

        fpr["macro average"] = fpr_grid
        tpr["macro average"] = mean_tpr
        roc_auc["macro average"] = auc(fpr["macro average"], tpr["macro average"])
        ax.plot(fpr["macro average"], tpr["macro average"], lw=2, label=f'macro average (AUC = {roc_auc["macro average"]:.4f})')

        
        # Misc things.
        ax.set(xlabel="False Positive Rate",
            ylabel="True Positive Rate",)
        ax.legend(loc="lower right")
        plt.show()

    print("Loaded model and dataset!")

def plot_ROC_AUC_multiple(paths, names, dataset_name, nrows, ncols):
    """Plotter ROC plots med AUC score for modellen som ligger
    på hver path i paths."""
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(30,15))
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['xtick.labelsize'] = 30
    plt.rcParams['ytick.labelsize'] = 30
    plt.rcParams['legend.fontsize'] = 20

    """Anta at paths og names er nested list med [nrows][ncols] indekser."""
    for i in range(len(paths)):
        for j in range(len(paths[i])):
            path = paths[i][j]
            name = names[i][j]
            if 'bert' in path:
                total_path = path+f'/run{1}/bert-tmp-r3/'
                print(type(total_path))
                net = AutoModelForSequenceClassification.from_pretrained(
                        total_path, num_labels=2, local_files_only=True
                    )
                # trainloader, testloader = load_imdb(0, 1, total_path) # Bruker hele datasettet som utgangspunkt.
                testloader = load_imdb(0, 1, "prajjwal1/bert-tiny") # Bruker hele datasettet som utgangspunkt.

                # Predict hele testsettet med modellen.
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                net.eval()
                store = []
                for batch in testloader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    with torch.no_grad():
                        outputs = net(**batch)
                    logits = outputs.logits
                    # predictions = torch.argmax(logits, dim=-1)
                    probs = torch.sigmoid(logits)
                    store += probs.tolist()
                predict_probas = np.asarray(store)
                true_labels = np.asarray([testloader.dataset[k]['labels'] for k in range(len(testloader.dataset))])

                # Compute values for curve.
                fpr, tpr, thresholds = roc_curve(true_labels, predict_probas[:, 1])
                roc_auc = auc(fpr, tpr)
                # Plot curves
                ax[i,j].plot(fpr, tpr, lw=3, label=f'ROC trained model (AUC = {roc_auc:.2f})')
                ax[i,j].plot([0, 1], [0, 1], color='gray', lw=3, label="Chance level", linestyle='--')  # random guess line
                # Misc things.
                ax[i,j].set_xlabel("False Positive Rate", fontsize=20)
                ax[i,j].set_ylabel("True Positive Rate", fontsize=20)
                ax[i,j].set_title(f"{name}", fontsize=22)
                # ax[i,j].legend(loc="lower right")
                ax[i,j].tick_params(axis='both', which='major', labelsize=22)

            elif 'squeezenet' in path:
                model = SqueezeNet("1_1", 10)
                total_path = path+f'/run{1}/squeezenet-tmp-r3'
                print(f"plotting for {total_path}")
                storage = os.path.abspath(total_path)
                model.load_state_dict(torch.load(storage, weights_only=True))
                model.eval()

                testloader = load_mnist_squeezenet(0, 1)
                
                # Predict hele testsettet med modellen.
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                model.eval()
                store = []
                true_labels = []
                for batch in testloader:
                    images = batch[0].to(device)
                    labels = batch[1].to(device)
                    with torch.no_grad():
                        outputs = model(images)
                    # predictions = torch.argmax(logits, dim=-1)
                    probs = torch.softmax(outputs, dim=1)
                    store += probs.tolist()
                    true_labels += labels.tolist()
                predict_probas = np.asarray(store)
                true_labels = np.asarray(true_labels)
                print(predict_probas.shape)
                # true_labels = np.asarray([valloader.dataset[i]['label'] for i in range(len(valloader.dataset))])
                print(true_labels.shape)
                # print(predict_probas.shape)

                train_labels = np.asarray([testloader.dataset[k]['label'] for k in range(len(testloader.dataset))])
                labelbinarizer = LabelBinarizer().fit(train_labels)
                true_labels = labelbinarizer.transform(true_labels) # Onehot encoded.
                print(true_labels.shape)
                # print(predict_probas.shape)
                # print(true_labels_onehot.shape)

                fpr, tpr, roc_auc = dict(), dict(), dict()
                # Create subplots.
                ax[i,j].plot([0, 1], [0, 1], color='gray', lw=3, label="Chance level", linestyle='--')  # random guess line
                for k in range(predict_probas.shape[1]):
                    # Compute values for curve.
                    fpr[f"class {k+1}"], tpr[f"class {k+1}"], thresholds = roc_curve(true_labels[:, k], predict_probas[:, k])
                    roc_auc[f"class {k+1}"] = auc(fpr[f"class {k+1}"], tpr[f"class {k+1}"])
                    # Plot curves
                    ax[i,j].plot(fpr[f"class {k+1}"], tpr[f"class {k+1}"], lw=3, label=f'Class {k+1}')

                # micro average.
                fpr["micro average"], tpr["micro average"], threshold = roc_curve(true_labels.ravel(), predict_probas.ravel())
                roc_auc["micro average"] = auc(fpr["micro average"], tpr["micro average"])
                ax[i,j].plot(fpr["micro average"], tpr["micro average"], lw=3, label=f'micro average')

                # macro average.
                fpr_grid = np.linspace(0.0, 1.0, 1000)
                # Interpolate all ROC curves at these points
                mean_tpr = np.zeros_like(fpr_grid)
                for k in range(10):
                    mean_tpr += np.interp(fpr_grid, fpr[f"class {k+1}"], tpr[f"class {k+1}"])  # linear interpolation
                # Average it and compute AUC
                mean_tpr /= 10

                fpr["macro average"] = fpr_grid
                tpr["macro average"] = mean_tpr
                roc_auc["macro average"] = auc(fpr["macro average"], tpr["macro average"])
                ax[i,j].plot(fpr["macro average"], tpr["macro average"], lw=3, label=f'macro average')

                
                # Misc things.
                ax[i,j].set_xlabel("False Positive Rate", fontsize=20)
                ax[i,j].set_ylabel("True Positive Rate", fontsize=20)
                ax[i,j].set_title(f"{name}", fontsize=22)
                # ax[i,j].legend(loc="lower right")
                ax[i,j].tick_params(axis='both', which='major', labelsize=22)

            elif 'logreg' in path:
                scale = "classic"
                if scale == "classic":
                    model = LogisticRegression(28*28, 10)
                elif scale == "larger":
                    model = LogisticRegression(224*224, 10)
                elif scale == "rgb":
                    model = LogisticRegression(3*224*224, 10)
                total_path = path+f'/run{1}/logreg-tmp-r3'
                storage = os.path.abspath(total_path)
                print(f"plotting for {total_path}")
                storage = os.path.abspath(total_path)
                model.load_state_dict(torch.load(storage, weights_only=True))
                model.eval()

                testloader = load_mnist_logreg(0, 1)
                
                # Predict hele testsettet med modellen.
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                model.eval()
                store = []
                true_labels = []
                for batch in testloader:
                    images = batch[0].to(device)
                    labels = batch[1].to(device)
                    with torch.no_grad():
                        outputs = model(images)
                    # predictions = torch.argmax(logits, dim=-1)
                    probs = torch.softmax(outputs, dim=1)
                    store += probs.tolist()
                    true_labels += labels.tolist()
                predict_probas = np.asarray(store)
                true_labels = np.asarray(true_labels)
                print(predict_probas.shape)
                # true_labels = np.asarray([valloader.dataset[i]['label'] for i in range(len(valloader.dataset))])
                print(true_labels.shape)
                # print(predict_probas.shape)

                train_labels = np.asarray([testloader.dataset[k]['label'] for k in range(len(testloader.dataset))])
                labelbinarizer = LabelBinarizer().fit(train_labels)
                true_labels = labelbinarizer.transform(true_labels) # Onehot encoded.
                print(true_labels.shape)
                # print(predict_probas.shape)
                # print(true_labels_onehot.shape)

                fpr, tpr, roc_auc = dict(), dict(), dict()
                # Create subplots.
                ax[i,j].plot([0, 1], [0, 1], color='gray', lw=3, label="Chance level", linestyle='--')  # random guess line
                for k in range(predict_probas.shape[1]):
                    # Compute values for curve.
                    fpr[f"class {k+1}"], tpr[f"class {k+1}"], thresholds = roc_curve(true_labels[:, k], predict_probas[:, k])
                    roc_auc[f"class {k+1}"] = auc(fpr[f"class {k+1}"], tpr[f"class {k+1}"])
                    # Plot curves
                    ax[i,j].plot(fpr[f"class {k+1}"], tpr[f"class {k+1}"], lw=3, label=f'Class {k+1}')

                # micro average.
                fpr["micro average"], tpr["micro average"], threshold = roc_curve(true_labels.ravel(), predict_probas.ravel())
                roc_auc["micro average"] = auc(fpr["micro average"], tpr["micro average"])
                ax[i,j].plot(fpr["micro average"], tpr["micro average"], lw=3, label=f'micro average')

                # macro average.
                fpr_grid = np.linspace(0.0, 1.0, 1000)
                # Interpolate all ROC curves at these points
                mean_tpr = np.zeros_like(fpr_grid)
                for k in range(10):
                    mean_tpr += np.interp(fpr_grid, fpr[f"class {k+1}"], tpr[f"class {k+1}"])  # linear interpolation
                # Average it and compute AUC
                mean_tpr /= 10

                fpr["macro average"] = fpr_grid
                tpr["macro average"] = mean_tpr
                roc_auc["macro average"] = auc(fpr["macro average"], tpr["macro average"])
                ax[i,j].plot(fpr["macro average"], tpr["macro average"], lw=3, label=f'macro average')

                
                # Misc things.
                ax[i,j].set_xlabel("False Positive Rate", fontsize=20)
                ax[i,j].set_ylabel("True Positive Rate", fontsize=20)
                ax[i,j].set_title(f"{name}", fontsize=22)
                # ax[i,j].legend(loc="lower right")
                ax[i,j].tick_params(axis='both', which='major', labelsize=22)
    
    # Fix legend.
    handles, labels = ax[0,0].get_legend_handles_labels()
    print(handles)
    print(labels)
    # fig.legend(handles, labels, loc='upper center')
    # Place legend outside the plot (right side)
    ax[0,1].legend(
        handles=handles,
        labels=labels,
        # ncol=7, # Spread out over columns 
        loc='center right',           # Anchor point of legend
        bbox_to_anchor=(1.3, 0.45),  # Position relative to axes
        borderaxespad=0.0,           # Padding between axes and legend
        frameon=True                 # Draw a box around legend
    )
    # ax[0,1].legend(handles, labels, loc='upper center')

    print("Done plotting.")
    # Misc things.
    fig.suptitle(f"ROC plots for various experiments on {dataset_name}", fontsize=30)
    fig.tight_layout()
    fig.savefig(f"./figures/{dataset_name}/roc_auc.pdf")
    # plt.show()

def plot_metric_multiple(metric, dfs, dataset_name, nrows, ncols):
    """Plotter metric for dfs med names og dataset_name."""
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, figsize=(30,15))
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['xtick.labelsize'] = 30
    plt.rcParams['ytick.labelsize'] = 30
    plt.rcParams['legend.fontsize'] = 20
    # plt.rcParams['font.size'] = 30

    if metric == "CPU":
        unit = "cores"
    elif metric == "Memory":
        unit = "MB"
    elif metric == "Network":
        unit = "MB"

    bokstaver = {(0, 0): "a", (0, 1): "b", (0, 2): "c", (1,0): "d", (1,1): "e", (1,2): "f"}
    from matplotlib.ticker import MaxNLocator

    i = 0
    j = 0
    for key in list(dfs.keys()):
        df = dfs[key]
        x = np.arange(df[list(df.keys())[0]].shape[0])
        for subkey in list(df.keys()):
            label = subkey
            if subkey == "laptop average":
                label = "computer 2 average"
                color = "tab:red"
            elif subkey == "server":
                color = "tab:green"
            elif subkey == "desktop average":
                label = "computer 1 average"
                color = "tab:blue"
            elif subkey == "client with server":
                color = "tab:brown"
            y = df[subkey]
            ax[i,j].plot(x, y, label=label, linewidth=3, color=color)
        if "SecAgg" in str(key):
            key = "SecAgg: C3"
        # ax[i,j].set(xlabel="Time (seconds)",
        #         ylabel=f"Usage ({unit})", # Må sette inn
        #         title=f"{bokstaver[(i,j)]}): {metric} usage in {unit} for {key}")
        ax[i,j].set_xlabel("Time (seconds)", fontsize=20)
        ax[i,j].set_ylabel(f"Usage ({unit})", fontsize=20)
        ax[i,j].set_title(f"{bokstaver[(i,j)]}) {key}", fontsize=22)
        ax[i,j].xaxis.set_major_locator(MaxNLocator(integer=True))
        ax[i,j].yaxis.set_major_locator(MaxNLocator(integer=True))
        # ax[i,j].legend()
        ax[i,j].tick_params(axis='both', which='major', labelsize=22)
        # Increment j.
        j += 1
        # If whole row is done, go to next row.
        if j == ncols:
            j = 0
            i += 1
    
    # Lag felles legend for alle subplots.
    # Hent ut handles og labels for secagg (0,0) og for en av fhe (-1,-1).
    # Finn indeks for client with server verdier siden det er den eneste som secagg ikke har.
    # Kombiner til felles legend.
    handles_fhe, labels_fhe = ax[-1,-1].get_legend_handles_labels()
    index = labels_fhe.index("client with server")
    handles, labels = ax[0,0].get_legend_handles_labels()
    handles.append(handles_fhe[index])
    labels.append(labels_fhe[index])
    print(handles)
    print(labels)
    # fig.legend(handles, labels, loc='upper center')
    ax[0,1].legend(handles, labels, loc='upper center')
            
    print("Done plotting.")
    # Misc things.
    fig.suptitle(f"{metric} usage in {unit} on dataset {dataset_name}", fontsize=30)
    fig.tight_layout()
    fig.savefig(f"./figures/{dataset_name}/{metric}.pdf")
    # plt.show()

def plot_df(df):
    fig, ax = plt.subplots(figsize=(10, 10))
    x = np.arange(df[list(df.keys())[0]].shape[0])
    for key in list(df.keys()):
        y = df[key]
        ax.plot(x, y, label=f"{key}")
    plt.legend()
    plt.show()

def save_plot_df(df, name):
    fig, ax = plt.subplots(figsize=(10, 10))
    x = np.arange(df[list(df.keys())[0]].shape[0])
    for key in list(df.keys()):
        y = df[key]
        ax.plot(x, y, label=f"{key}")
    plt.title(name)
    plt.legend()
    fig.savefig(f"./figures/{name}.png", format="png")

def save_2_plot_vertical(df1, df2, name):
    fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 10))

    x1 = np.arange(df1[list(df1.keys())[0]].shape[0])
    for key in list(df1.keys()):
        y1 = df1[key]
        ax1.plot(x1, y1, label=f"{key}")

    x2 = np.arange(df2[list(df2.keys())[0]].shape[0])
    for key in list(df2.keys()):
        y2 = df2[key]
        ax2.plot(x2, y2, label=f"{key}")

    plt.suptitle(name)
    ax1.legend()
    ax2.legend()
    fig.savefig(f"./figures/{name}.png", format="png")

def save_2_plot_horizontal(df1, df2, name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))

    x1 = np.arange(df1[list(df1.keys())[0]].shape[0])
    for key in list(df1.keys()):
        y1 = df1[key]
        ax1.plot(x1, y1, label=f"{key}")

    x2 = np.arange(df2[list(df2.keys())[0]].shape[0])
    for key in list(df2.keys()):
        y2 = df2[key]
        ax2.plot(x2, y2, label=f"{key}")

    plt.suptitle(name)
    ax1.legend()
    ax2.legend()
    fig.savefig(f"./figures/{name}.png", format="png")

def save_2_plot_combined(df1, df2, name):
    fig, ax = plt.subplots(figsize=(10, 10))

    x1 = np.arange(df1[list(df1.keys())[0]].shape[0])
    for key in list(df1.keys()):
        y1 = df1[key]
        ax.plot(x1, y1, label=f"{key}-org")

    x2 = np.arange(df2[list(df2.keys())[0]].shape[0])
    for key in list(df2.keys()):
        y2 = df2[key]
        ax.plot(x2, y2, label=f"{key}-ckks")

    plt.suptitle(name)
    ax.legend()
    fig.savefig(f"./figures/{name}.png", format="png")

def compute_mean_std_runtime(path):
    """Compute mean and std for runtime over all the runs in path.
    Merge runtimes for different rounds for ml-desktop, ml-laptop, noise,
    keygen, dec, block-no-dec, block and take max of ml-desktop and ml-laptop."""
    # Create filepaths.
    path = Path(path)
    runs = list(path.iterdir())
    file_paths = [run/"runtime.csv" for run in runs]
    # Create pandas dataframes for each path.
    dfs = []
    for filepath in file_paths:
        df = pd.read_csv(filepath, sep=",")
        if "CKKS-NF" in str(path):
            # Har glemt å trekke fra noise estimation fra other category.
            df['other'] -= (df['noise-r1'] + df['noise-r2'] + df['noise-r3'])
        dfs.append(df)

    dfs2 = []
    for df in dfs:
        # Sum times for each round, e.g., ml-desktop-r1 + r2 + r3.
        new_df = pd.DataFrame(dict())
        new_df['runtime'] = df['runtime']
        # Beregn maks av tiden på stasjonær og bærbar.
        new_df['ml'] = max(sum([df[f'ml-desktop-r{i}'].item() for i in range(1,4)]), sum([df[f'ml-laptop-r{i}'].item() for i in range(1,4)]))
        if "CKKS-NF" in str(path):
            new_df['noise'] = sum([df[f'noise-r{i}'] for i in range(1,4)])
        if "fhefedavg" in str(path):
            new_df['keygen'] = sum([df[f'keygen-r{i}'] for i in range(1,4)])
            new_df['dec'] = sum([df[f'dec-r{i}'] for i in range(1,4)])
            new_df['block-no-dec'] = sum([df[f'block-no-dec-r{i}'] for i in range(1,4)])
            new_df['block'] = sum([df[f'block-r{i}'] for i in range(1,4)])
        new_df['other'] = df['other']
        dfs2.append(new_df)

    if len(runs) == 3:
        # Compute mean and std over the 3 runs.
        mean_df = pd.DataFrame(dict())
        std_df = pd.DataFrame(dict())
        for key in list(dfs2[0].keys()):
            mean_df[key] = [np.mean([df[key].to_numpy()[0] for df in dfs2])]
            std_df[key] = [np.std([df[key].to_numpy()[0] for df in dfs2])]

        return mean_df, std_df
    
    else:
        # Kun et run, så std blir bare null.
        # Mean er også meningsløst å regne ut, men jeg vet ikke om resultatet formateres noe,
        # så gjør det for sikkerhets skyld.
        mean_df = pd.DataFrame(dict())
        for key in list(dfs2[0].keys()):
            mean_df[key] = [np.mean([df[key].to_numpy()[0] for df in dfs2])]
        return mean_df

def compute_mean_std_model_metrics(path, computer="windows"):
    """Compute mean and std of the recorded metrics for each round
    and for metrics from evaluating the final model on the testing set."""

    """Loss."""
    # Create filepaths.
    path = Path(path)
    runs = list(path.iterdir())
    file_paths = [run/"loss.csv" for run in runs]
    # Create pandas dataframes for each path.
    dfs = []
    for filepath in file_paths:
        df = pd.read_csv(filepath, sep=",")
        dfs.append(df)
    # Compute mean and std over the 3 runs.
    loss_mean_df = pd.DataFrame(dict())
    loss_std_df = pd.DataFrame(dict())
    for key in list(dfs[0].keys()):
        loss_mean_df[key] = [np.mean([df[key].to_numpy()[0] for df in dfs])]
        loss_std_df[key] = [np.std([df[key].to_numpy()[0] for df in dfs])]

    """Accuracy."""
    # Create filepaths.
    path = Path(path)
    runs = list(path.iterdir())
    file_paths = [run/"accuracy.csv" for run in runs]
    # Create pandas dataframes for each path.
    dfs = []
    for filepath in file_paths:
        df = pd.read_csv(filepath, sep=",")
        dfs.append(df)
    # Compute mean and std over the 3 runs.
    accuracy_mean_df = pd.DataFrame(dict())
    accuracy_std_df = pd.DataFrame(dict())
    for key in list(dfs[0].keys()):
        accuracy_mean_df[key] = [np.mean([df[key].to_numpy()[0] for df in dfs])]
        accuracy_std_df[key] = [np.std([df[key].to_numpy()[0] for df in dfs])]
    
    if "bert" in str(path):
        """F1 score."""
        # Create filepaths.
        path = Path(path)
        runs = list(path.iterdir())
        file_paths = [run/"f1.csv" for run in runs]
        # Create pandas dataframes for each path.
        dfs = []
        for filepath in file_paths:
            df = pd.read_csv(filepath, sep=",")
            dfs.append(df)
        # Compute mean and std over the 3 runs.
        f1_mean_df = pd.DataFrame(dict())
        f1_std_df = pd.DataFrame(dict())
        for key in list(dfs[0].keys()):
            f1_mean_df[key] = [np.mean([df[key].to_numpy()[0] for df in dfs])]
            f1_std_df[key] = [np.std([df[key].to_numpy()[0] for df in dfs])]

    if computer == "linux":
        """Compute max, mean and std from evaluating final model on test set."""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        loss_list = []
        accuracy_list = []
        f1_list = []
        if "logreg" in str(path):
            for run in range(1,len(runs)+1):
                # Load final model and testing set.
                model = LogisticRegression(28*28, 10)
                total_path = Path(str(path)+f'/run{run}/logreg-tmp-r3')
                print(total_path)
                storage = os.path.abspath(total_path)
                print(storage)
                model.load_state_dict(torch.load(storage, weights_only=True))
                print("Har loada model.")
                model.eval()
                testloader = load_mnist_logreg(0, 1)
                print("Har loada testing set.")

                """Validate the model on the test set."""
                model.to(device)
                criterion = torch.nn.CrossEntropyLoss().to(device)
                correct, loss = 0, 0.0
                with torch.no_grad():
                    for batch in testloader:
                        images = batch[0].to(device)
                        labels = batch[1].to(device)
                        outputs = model(images)
                        loss += criterion(outputs, labels).item()
                        correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
                        print("Inne i for loop for evaluation")
                print("Ferdig med for loop")
                accuracy = correct / len(testloader.dataset)
                loss = loss / len(testloader)
                loss_list.append(loss)
                accuracy_list.append(accuracy)

        elif "squeezenet" in str(path):
            for run in range(1,len(runs)+1):
                # Load final model and testing set.
                model = SqueezeNet("1_1", 10)
                total_path = Path(str(path)+f'/run{run}/squeezenet-tmp-r3')
                storage = os.path.abspath(total_path)
                model.load_state_dict(torch.load(storage, weights_only=True))
                model.eval()
                testloader = load_mnist_squeezenet(0, 1)

                """Validate the model on the test set."""
                model.to(device)
                criterion = torch.nn.CrossEntropyLoss()
                correct, loss = 0, 0.0
                with torch.no_grad():
                    for batch in testloader:
                        images = batch[0].to(device)
                        labels = batch[1].to(device)
                        outputs = model(images)
                        loss += criterion(outputs, labels).item()
                        correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
                accuracy = correct / len(testloader.dataset)
                loss = loss / len(testloader)
                loss_list.append(loss)
                accuracy_list.append(accuracy)

        elif "bert" in str(path):
            for run in range(1,len(runs)+1):
                # Load final model and testing set.
                total_path = Path(str(path)+f'/run{1}/bert-tmp-r3/')
                print(type(total_path))
                model = AutoModelForSequenceClassification.from_pretrained(
                        total_path, num_labels=2, local_files_only=True
                    )
                testloader = load_imdb(0, 1, "prajjwal1/bert-tiny") # Bruker hele datasettet som utgangspunkt.

                metric = load_metric("accuracy")
                metric_f1 = load_metric("f1")
                loss = 0
                model.eval()
                for batch in testloader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    with torch.no_grad():
                        outputs = model(**batch)
                    logits = outputs.logits
                    loss += outputs.loss.item()
                    predictions = torch.argmax(logits, dim=-1)
                    metric.add_batch(predictions=predictions, references=batch["labels"])
                    metric_f1.add_batch(predictions=predictions, references=batch["labels"])
                loss /= len(testloader.dataset)
                accuracy = metric.compute()["accuracy"]
                f1 = metric_f1.compute()["f1"]
                loss_list.append(loss)
                accuracy_list.append(accuracy)
                f1_list.append(f1)
                """Jeg så en tilfeldig kommentar på flower forumet som sa at F1-score
                ikke kan averages, at det blir feil. Det gjaldt også i forhold til 
                å average f1-score fra hver client. Må finne ut av om det stemmer.
                Istedenfor måtte man beregne TP, TN, FP og FN og bruke summene av de for å beregne f1 på nytt."""
        
        # Compute mean and std for evaluation on testing set.
        loss_mean_df['test'] = np.mean(loss_list)
        loss_std_df['test'] = np.std(loss_list)
        accuracy_mean_df['test'] = np.mean(accuracy_list)
        accuracy_std_df['test'] = np.std(accuracy_list)
        if "bert" in str(path):
            f1_mean_df['test'] = np.mean(f1_list)
            f1_std_df['test']= np.std(f1_list)

    if len(runs) == 3:
        if "bert" in str(path):
            return loss_mean_df, loss_std_df, accuracy_mean_df, accuracy_std_df, f1_mean_df, f1_std_df
        else:
            return loss_mean_df, loss_std_df, accuracy_mean_df, accuracy_std_df
    
    else:
        # Kun et run.
        if "bert" in str(path):
            return loss_mean_df, accuracy_mean_df, f1_mean_df
        else:
            return loss_mean_df, accuracy_mean_df

def compute_max_mean_std_container_metrics(path, type):
    """Compute max, mean and std of chosen container metric indicated by type."""
    # Create filepaths.
    path = Path(path)
    runs = list(path.iterdir())

    if type == "network":
        # For network, kombiner sent og recv for hvert måletidspunkt.
        file_paths = [run/f"{type}" for run in runs]

        # Create dataframes for each path and remove rows with at least one '-'.
        dfs = []
        dfs_sent = []
        dfs_recv = []
        lenghts = []
        for filepath in file_paths:
            df_sent = pd.read_csv(str(filepath)+"_sent.csv", sep=",")
            # Remove rows which contain at least one '-'.
            num_bindestreker = (max([len(df_sent[df_sent[key] == "-"]) for key in list(df_sent.keys())]))
            if num_bindestreker > 0:
                df_sent = df_sent.iloc[0:-num_bindestreker]
            df_sent = df_sent.astype(np.float64)
            lenghts.append(df_sent.shape[0])
            dfs_sent.append(df_sent)

            df_recv = pd.read_csv(str(filepath)+"_recv.csv", sep=",")
            # Remove rows which contain at least one '-'.
            num_bindestreker = (max([len(df_recv[df_recv[key] == "-"]) for key in list(df_recv.keys())]))
            if num_bindestreker > 0:
                df_recv = df_recv.iloc[0:-num_bindestreker]
            df_recv = df_recv.astype(np.float64)
            lenghts.append(df_recv.shape[0])
            dfs_recv.append(df_recv)

        # Find shortest df and resize all of them to the same shape.
        shortest = np.min(lenghts)
        for i in range(len(dfs_sent)):
            dfs_sent[i] = dfs_sent[i].iloc[0:shortest]
            dfs_recv[i] = dfs_recv[i].iloc[0:shortest]
        
        # Merge dfs_sent and dfs_recv
        for i in range(len(dfs_sent)):
            df = dfs_sent[i] + dfs_recv[i]
            dfs.append(df)

    else:
        file_paths = [run/f"{type}.csv" for run in runs]

        # Create dataframes for each path and remove rows with at least one '-'.
        dfs = []
        lenghts = []
        for filepath in file_paths:
            df = pd.read_csv(filepath, sep=",")
            # Remove rows which contain at least one '-'.
            num_bindestreker = (max([len(df[df[key] == "-"]) for key in list(df.keys())]))
            if num_bindestreker > 0:
                df = df.iloc[0:-num_bindestreker]
            df = df.astype(np.float64)
            lenghts.append(df.shape[0])
            dfs.append(df)

        # Find shortest df and resize all of them to the same shape.
        shortest = np.min(lenghts)
        for i in range(len(dfs)):
            dfs[i] = dfs[i].iloc[0:shortest]

    # Reformat container names.
    for i in range(len(dfs)):
        clipped_names_df = pd.DataFrame(dict())
        # Remove 'dockerfiles-' start of each name.
        # Remove trailing '-1' from each name.
        for key in list(dfs[i].keys()):
            new_key_list = key.split("-")
            new_key_list.pop(0)
            new_key_list.pop(-1)
            new_key = "-".join(new_key_list)

            clipped_names_df[new_key] = dfs[i][key]
        dfs[i] = clipped_names_df
    
    # Merge apps and supers.
    for i in range(len(dfs)):
        if "3" in str(path):
            dfs[i] = combine_containers(dfs[i], 3)
        elif "5" in str(path):
            dfs[i] = combine_containers(dfs[i], 5)
        elif "7" in str(path):
            dfs[i] = combine_containers(dfs[i], 7)

    # Ønsker å beregne gjennomsnitt av clients for hver PC.
    # For kryptering, la laptop-client-1 være utenfor snitt siden den launcher subprocess server
    # og har kanskje annen ressursbruk.
    average_dfs = [pd.DataFrame(dict()) for _ in range(len(dfs))]
    desktop_keys = [key for key in list(dfs[0].keys()) if "desktop" in key]
    laptop_keys = [key for key in list(dfs[0].keys()) if "laptop" in key]
    for i in range(len(dfs)):
        # Beregn average over nøklene for hvert run.
        # E.g., hvis det er to clients på desktop så summer verdiene ved hvert sekund og del på 2.
        average_dfs[i]['desktop average'] = np.mean([dfs[i][key] for key in desktop_keys], 0)
        if "fhefedavg" in str(path):
            # laptop-client-1 launcher server subprocess og har kanskje annen ressursbruk, så utelat den fra snitt.
            if "nc3" in str(path):
                # For nc3 kjører bare client with server på computer 2. laptop average skal dermed ikke beregnes.
                average_dfs[i]['laptop average'] = np.zeros(shortest)
            else:
                average_dfs[i]['laptop average'] = np.mean([dfs[i][key] for key in laptop_keys if "1" not in key], 0) 
            average_dfs[i]['client with server'] = dfs[i]['laptop-client-1']
        else:
            average_dfs[i]['laptop average'] = np.mean([dfs[i][key] for key in laptop_keys], 0)
        average_dfs[i]['server'] = dfs[i]['server']
    
    # Compute mean and std over the 3 runs.
    max_df = pd.DataFrame(dict())
    mean_df = pd.DataFrame(dict())
    std_df = pd.DataFrame(dict())
    for key in list(average_dfs[0].keys()):
        max_df[key] = [np.max([df[key].to_numpy() for df in average_dfs])]
        mean_df[key] = [np.mean([df[key].to_numpy() for df in average_dfs])]
        std_df[key] = [np.std([df[key].to_numpy() for df in average_dfs])]
    """Må endre dette slik at output matcher input til table."""

    # if type == "memory" or type == "cache":
    # elif type == "network_recv" or type == "network_sent":
    # if type in ['cache', 'network_recv', 'network_sent'] or (type == "memory" and "logreg" in str(path)):
    # For network in and out convert from bytes to megabytes.
    if type != "cpu":
        def b_to_mb(x):
            # MB = 10^6 B.
            return x / 1e6
        max_df = max_df.apply(b_to_mb, raw=True)
        mean_df = mean_df.apply(b_to_mb, raw=True)
        std_df = std_df.apply(b_to_mb, raw=True)
    # elif type == "memory":
    #     # For memory and cache convert from bytes to gigabytes.
    #     def b_to_gb(x):
    #         # GB = 10^9 B.
    #         return x / 1e9
    #     max_df = max_df.apply(b_to_gb, raw=True)
    #     mean_df = mean_df.apply(b_to_gb, raw=True)
    #     std_df = std_df.apply(b_to_gb, raw=True)
    
    return max_df, mean_df, std_df

def format_table_runtime(dict, type="org"):
    """Format tables for runtime.
    dict has the algorithm names as keys, e.g., FedAvg and SecAgg, and 
    their dfs as values. type should be specified as "org", "fhefedav" or "ckks-nf".
    
    General table format for FedAvg and SecAgg is
     & Runtime & Clientside ML & Other \\
    \\multirow{3}{4em}{algorithm 1 name} & ... & ... & ... \\
    \\multirow{3}{4em}{algorithm 2 name} & ... & ... & ... \\
    ...

    and for encryption schemes it is
     & Runtime & Clientside ML & KeyGen & Dec & Block no dec & Block total & Other \\
    \\multirow{3}{4em}{algorithm 1 name} & ... & ... & ... & ... & ... & ... & ... \\
    \\multirow{3}{4em}{algorithm 2 name} & ... & ... & ... & ... & ... & ... & ... \\
    ...

    and for ckks-nf it is 
     & Runtime & Clientside ML & Noise & KeyGen & Dec & Block no dec & Block total & Other \\
    \\multirow{3}{4em}{algorithm 1 name} & ... & ... & ... & ... & ... & ... & ... & ... \\
    \\multirow{3}{4em}{algorithm 2 name} & ... & ... & ... & ... & ... & ... & ... & ... \\
    ...
    """
    print("\n\n\n")

    if type == "org":
        print("\\hline")
        print("Parameters & Runtime & Clientside ML & Other \\\\")
        print("\\hline")
        for name in list(dict.keys()):
            timem = [dict[name]["time mean"][key].item() for key in list(dict[name]["time mean"].keys())]
            times = [dict[name]["time std"][key].item() for key in list(dict[name]["time std"].keys())]
            print(f"{name}" 
                  f" & {timem[0]:.2f} ({times[0]:.2f}) & {timem[1]:.2f} ({times[1]:.2f}) & {timem[2]:.2f} ({times[2]:.2f}) \\\\")
            print("\\hline")
    elif type == "fhefedavg":
        print("\\hline")
        print("Parameters & Runtime & Clientside ML & KeyGen & Dec & Block no dec & Other \\\\")
        print("\\hline")
        for name in list(dict.keys()):
            timem = [dict[name]["time mean"][key].item() for key in list(dict[name]["time mean"].keys())]
            try:
                times = [dict[name]["time std"][key].item() for key in list(dict[name]["time std"].keys())]
                print(f"{name}" 
                    f" & {timem[0]:.2f} ({times[0]:.2f}) & {timem[1]:.2f} ({times[1]:.2f}) & {timem[2]:.2f} ({times[2]:.2f})"
                    f" & {timem[3]:.2f} ({times[3]:.2f}) & {timem[4]:.2f} ({times[4]:.2f}) & {timem[5]:.2f} ({times[5]:.2f})"
                    f" \\\\")
            except KeyError:
                print(f"{name}" 
                    f" & {timem[0]:.2f} & {timem[1]:.2f} & {timem[2]:.2f}"
                    f" & {timem[3]:.2f} & {timem[4]:.2f} & {timem[5]:.2f}"
                    f" \\\\")
            print("\\hline")
    elif type == "ckks-nf":
        print("\\hline")
        print("Parameters & Runtime & Clientside ML & Noise & KeyGen & Dec & Block no dec & Other \\\\")
        print("\\hline")
        for name in list(dict.keys()):
            timem = [dict[name]["time mean"][key].item() for key in list(dict[name]["time mean"].keys())]
            try:
                times = [dict[name]["time std"][key].item() for key in list(dict[name]["time std"].keys())]
                print(f"{name}" 
                    f" & {timem[0]:.2f} ({times[0]:.2f}) & {timem[1]:.2f} ({times[1]:.2f}) & {timem[2]:.2f} ({times[2]:.2f})"
                    f" & {timem[3]:.2f} ({times[3]:.2f}) & {timem[4]:.2f} ({times[4]:.2f}) & {timem[5]:.2f} ({times[5]:.2f})"
                    f" & {timem[6]:.2f} ({times[6]:.2f}) \\\\")
            except KeyError:
                print(f"{name}" 
                    f" & {timem[0]:.2f} & {timem[1]:.2f} & {timem[2]:.2f}"
                    f" & {timem[3]:.2f} & {timem[4]:.2f} & {timem[5]:.2f}"
                    f" & {timem[6]:.2f} \\\\")
            print("\\hline")
    
    print("\n\n\n")

def format_table_loss_acc(dict, model="logreg"):
    """Format tables for loss and accuracy.
    dict has the algorithm names as keys, e.g., FedAvg and SecAgg, and 
    their dfs as values.
    
    General table format is
     & \\multicolumn{3}{c}{Loss} & \\multicolumn{3}{c}{Accuracy} \\
     & Round 1 & Round 2 & Round 3 & Round 1 & Round 2 & Round 3 \\
    \\multirow{3}{4em}{algorithm 1 name} & ... & ... & ... & ... & ... & ... \\
    \\multirow{3}{4em}{algorithm 2 name} & ... & ... & ... & ... & ... & ... \\
    ..."""
    # Need double the amount of \ to escape two of them.
    print("\n\n\n")

    
    if model == "bert":
        print("\\hline")
        names = list(dict.keys())
        names_print = ""
        for name in names:
            names_print += name + " & "
        names_print = names_print.rstrip(" & ")
        print(f" & Parameters & " + names_print + " \\\\" + "\n\\hline")
        loss_line_r1 = "\\multirow{3}{8em}{Loss} & Round 1"
        loss_line_r2 = " & Round 2"
        loss_line_r3 = " & Round 3"
        for name in names:
            try:
                loss_line_r1 += f" & {dict[name]['loss mean']['round-1'].item():.4f} ({dict[name]['loss std']['round-1'].item():.4f})"
            except KeyError:
                loss_line_r1 += f" & {dict[name]['loss mean']['round-1'].item():.4f}"
            try:
                loss_line_r2 += f" & {dict[name]['loss mean']['round-2'].item():.4f} ({dict[name]['loss std']['round-2'].item():.4f})"
            except KeyError:
                loss_line_r2 += f" & {dict[name]['loss mean']['round-2'].item():.4f}"
            try:
                loss_line_r3 += f" & {dict[name]['loss mean']['round-3'].item():.4f} ({dict[name]['loss std']['round-3'].item():.4f})"
            except KeyError:
                loss_line_r3 += f" & {dict[name]['loss mean']['round-3'].item():.4f}"
        loss_line_r1 += " \\\\"
        loss_line_r2 += " \\\\"
        loss_line_r3 += " \\\\"
        print(loss_line_r1  + "\n\\cline{2-5}")
        print(loss_line_r2  + "\n\\cline{2-5}")
        print(loss_line_r3  + "\n\\hline")
        acc_line_r1 = "\\multirow{3}{8em}{Accuracy} & Round 1"
        acc_line_r2 = " & Round 2"
        acc_line_r3 = " & Round 3"
        for name in names:
            try:
                acc_line_r1 += f" & {dict[name]['accuracy mean']['round-1'].item():.4f} ({dict[name]['accuracy std']['round-1'].item():.4f})"
            except KeyError:
                acc_line_r1 += f" & {dict[name]['accuracy mean']['round-1'].item():.4f}"
            try:
                acc_line_r2 += f" & {dict[name]['accuracy mean']['round-2'].item():.4f} ({dict[name]['accuracy std']['round-2'].item():.4f})"
            except KeyError:
                acc_line_r2 += f" & {dict[name]['accuracy mean']['round-2'].item():.4f}"
            try:
                acc_line_r3 += f" & {dict[name]['accuracy mean']['round-3'].item():.4f} ({dict[name]['accuracy std']['round-3'].item():.4f})"
            except KeyError:
                acc_line_r3 += f" & {dict[name]['accuracy mean']['round-3'].item():.4f}"
        acc_line_r1 += " \\\\"
        acc_line_r2 += " \\\\"
        acc_line_r3 += " \\\\"
        print(acc_line_r1  + "\n\\cline{2-5}")
        print(acc_line_r2  + "\n\\cline{2-5}")
        print(acc_line_r3  + "\n\\hline")
        f1_line_r1 = "\\multirow{3}{8em}{$F_1$ score} & Round 1"
        f1_line_r2 = " & Round 2"
        f1_line_r3 = " & Round 3"
        for name in names:
            try:
                f1_line_r1 += f" & {dict[name]['f1 mean']['round-1'].item():.4f} ({dict[name]['f1 std']['round-1'].item():.4f})"
            except KeyError:
                f1_line_r1 += f" & {dict[name]['f1 mean']['round-1'].item():.4f}"
            try:
                f1_line_r2 += f" & {dict[name]['f1 mean']['round-2'].item():.4f} ({dict[name]['f1 std']['round-2'].item():.4f})"
            except KeyError:
                f1_line_r2 += f" & {dict[name]['f1 mean']['round-2'].item():.4f}"
            try:
                f1_line_r3 += f" & {dict[name]['f1 mean']['round-3'].item():.4f} ({dict[name]['f1 std']['round-3'].item():.4f})"
            except KeyError:
                f1_line_r3 += f" & {dict[name]['f1 mean']['round-3'].item():.4f}"
        f1_line_r1 += " \\\\"
        f1_line_r2 += " \\\\"
        f1_line_r3 += " \\\\"
        print(f1_line_r1  + "\n\\cline{2-5}")
        print(f1_line_r2  + "\n\\cline{2-5}")
        print(f1_line_r3  + "\n\\hline")
        """Må flippe table slik at loss, accuracy og f1 vises nedover.
        Må printe orgfedavg og secagg i samme tabell siden jeg ikke bare kan skjøte på."""
        # print("\\multirow{3}{8em}{Parameters} & " + f"{dict[names[0]["loss mean"]]}")


        # print(f"{name}"
        #     f" & {lossm[0]:.4f} ({losss[0]:.4f}) & {lossm[1]:.4f} ({losss[1]:.4f}) & {lossm[2]:.4f} ({losss[2]:.4f})"
        #     f" & {accm[0]:.4f} ({accs[0]:.4f}) & {accm[1]:.4f} ({accs[1]:.4f}) & {accm[2]:.4f} ({accs[2]:.4f})"
        #     f" & {f1m[0]:.4f} ({f1s[0]:.4f}) & {f1m[1]:.4f} ({f1s[1]:.4f}) & {f1m[2]:.4f} ({f1s[2]:.4f}) \\\\")
        # print("\\hline")
    else:
        print("\\hline")
        print("\\multirow{2}{8em}{Parameters} & \\multicolumn{3}{c|}{Loss} & \\multicolumn{3}{c|}{Accuracy} \\\\")
        print("\\cline{2-7}")
        print(" & Round 1 & Round 2 & Round 3 & Round 1 & Round 2 & Round 3 \\\\")
        print("\\hline")
        for name in list(dict.keys()):
            lossm = [dict[name]["loss mean"][key].item() for key in list(dict[name]["loss mean"].keys())]
            accm = [dict[name]["accuracy mean"][key].item() for key in list(dict[name]["accuracy mean"].keys())]
            try:
                losss = [dict[name]["loss std"][key].item() for key in list(dict[name]["loss std"].keys())]
                accs = [dict[name]["accuracy std"][key].item() for key in list(dict[name]["accuracy std"].keys())]

                print(f"{name}"
                    f" & {lossm[0]:.4f} ({losss[0]:.4f}) & {lossm[1]:.4f} ({losss[1]:.4f}) & {lossm[2]:.4f} ({losss[2]:.4f})"
                    # f" & {lossm[0]:.1E} ({losss[0]:.1E}) & {lossm[1]:.1E} ({losss[1]:.1E}) & {lossm[2]:.1E} ({losss[2]:.1E})"
                    f" & {accm[0]:.4f} ({accs[0]:.4f}) & {accm[1]:.4f} ({accs[1]:.4f}) & {accm[2]:.4f} ({accs[2]:.4f}) \\\\")
            except KeyError:
                print(f"{name}"
                    f" & {lossm[0]:.4f} & {lossm[1]:.4f} & {lossm[2]:.4f}"
                    # f" & {lossm[0]:.1E} & {lossm[1]:.1E} & {lossm[2]:.1E}"
                    f" & {accm[0]:.4f} & {accm[1]:.4f} & {accm[2]:.4f} \\\\")
            print("\\hline")
    
    print("\n\n\n")

def format_table_container_metrics(dict, type="org", memory = False):
    """Format table for container metrics.
    dict has the algorithm names as keys, e.g., FedAvg and SecAgg, and 
    their dfs as values. type should be specified as "org" or "fhefedavg".
    
     & Computer 1 & \\multicolumn{2}{c}{Computer 2} &  \\
     & Client mean & client w server & client mean & server \\
    name 1 & ... & ... & ... & ... \\
    name 2 & ... & ... & ... & ... \\"""
    print("\n\n\n")

    if type == "org":
        print("\\hline")
        print("\\multirow{2}{8em}{Parameters} & Computer 1 & \\multicolumn{2}{c|}{Computer 2}  \\\\")
        print("\\cline{2-5}")
        print(" & Client mean & Client mean & Server \\\\")
        print("\\hline")
        for name in list(dict.keys()):
            max = [dict[name]["max"][key].item() for key in list(dict[name]["max"].keys())]
            mean = [dict[name]["mean"][key].item() for key in list(dict[name]["mean"].keys())]
            std = [dict[name]["std"][key].item() for key in list(dict[name]["std"].keys())]
            if memory == True:
                print(f"{name}" 
                    f" & {max[0]:.0f}, {mean[0]:.0f}, {std[0]:.0f}"
                    f" & {max[1]:.0f}, {mean[1]:.0f}, {std[1]:.0f}"
                    f" & {max[2]:.0f}, {mean[2]:.0f}, {std[2]:.0f} \\\\")
            else:
                print(f"{name}" 
                    f" & {max[0]:.3f}, {mean[0]:.3f}, {std[0]:.3f}"
                    f" & {max[1]:.3f}, {mean[1]:.3f}, {std[1]:.3f}"
                    f" & {max[2]:.3f}, {mean[2]:.3f}, {std[2]:.3f} \\\\")
            print("\\hline")
    else:
        print("\\hline")
        print("\\multirow{2}{8em}{Parameters} & Computer 1 & \\multicolumn{3}{c|}{Computer 2}  \\\\")
        print("\\cline{2-5}")
        print(" & Client mean & Client mean & Client with server & Server \\\\")
        print("\\hline")
        for name in list(dict.keys()):
            max = [dict[name]["max"][key].item() for key in list(dict[name]["max"].keys())]
            mean = [dict[name]["mean"][key].item() for key in list(dict[name]["mean"].keys())]
            std = [dict[name]["std"][key].item() for key in list(dict[name]["std"].keys())]
            # Bytt ut felter med 0 med en bindestrek for å gjøre det mer tydelig
            # at feltet ikke gjelder.
            if abs(max[1]) < 1e-10:
                max[1] = '-'
                mean[1] = '-'
                std[1] = '-'
                if memory == True:
                    print(f"{name}" 
                        f" & {max[0]:.0f}, {mean[0]:.0f}, {std[0]:.0f}"
                        f" & {max[1]}, {mean[1]}, {std[1]}"
                        f" & {max[2]:.0f}, {mean[2]:.0f}, {std[2]:.0f}"
                        f" & {max[3]:.0f}, {mean[3]:.0f}, {std[3]:.0f} \\\\")
                else:
                    print(f"{name}" 
                        f" & {max[0]:.3f}, {mean[0]:.3f}, {std[0]:.3f}"
                        f" & {max[1]}, {mean[1]}, {std[1]}"
                        f" & {max[2]:.3f}, {mean[2]:.3f}, {std[2]:.3f}"
                        f" & {max[3]:.3f}, {mean[3]:.3f}, {std[3]:.3f} \\\\")
            else:
                if memory == True:
                    print(f"{name}" 
                    f" & {max[0]:.0f}, {mean[0]:.0f}, {std[0]:.0f}"
                    f" & {max[1]:.0f}, {mean[1]:.0f}, {std[1]:.0f}"
                    f" & {max[2]:.0f}, {mean[2]:.0f}, {std[2]:.0f}"
                    f" & {max[3]:.0f}, {mean[3]:.0f}, {std[3]:.0f} \\\\")
                else:
                    print(f"{name}" 
                        f" & {max[0]:.3f}, {mean[0]:.3f}, {std[0]:.3f}"
                        f" & {max[1]:.3f}, {mean[1]:.3f}, {std[1]:.3f}"
                        f" & {max[2]:.3f}, {mean[2]:.3f}, {std[2]:.3f}"
                        f" & {max[3]:.3f}, {mean[3]:.3f}, {std[3]:.3f} \\\\")
            print("\\hline")
    
    print("\n\n\n")

def create_dict_for_whole_dir(path, table, metric_type, sec_type):
    """Create a dict to be used in format_table functions for the whole directory in path.
    path should only point to the directory for a whole algorithm, e.g.,
    'benchmarking_results/logreg-orgfedavg/' or 'benchmarking_results/logreg-fhefedavg/BGV/'
    table should be 'runtime', 'model' or 'container' to show what type of table is to be produced.
    If table is 'container', type indicates the metric, e.g., 'cpu' or 'memory'."""

    path = Path(path)
    algorithms = list(path.iterdir())
    # file_paths = [run/f"{type}.csv" for run in runs]

    return_dict = {}
    if table == "runtime":
        for alg in algorithms:
            name = alg.name
            try:
                mean_df, std_df = compute_mean_std_runtime(alg)
                if 'fhefedavg' in str(path):
                    # Do not need total block time.
                    mean_df.pop("block")
                    std_df.pop("block")
                if 'orgfedavg' in str(path):
                    return_dict[f'FedAvg {name}'] = {"time mean": mean_df, "time std": std_df}
                elif 'secagg' in str(path):
                    return_dict[f'SecAgg {name}'] = {"time mean": mean_df, "time std": std_df}
                else:
                    # For FHE there will only be one scheme per table.
                    # Must escape the underscores so that the name is valid in overleaf.
                    # Escape underscores gir for lange navn. Prøve med komma og mellomrom.
                    name_parts = name.split('_')
                    name_parts = relabel_names(name_parts, sec_type)
                    if "logreg" in str(path):
                        # Don't need to denote chunk size for logreg.
                        for i in range(len(name_parts)):
                            if "nb" in name_parts[i]:
                                name_parts.pop(i)
                                break
                    name = ", ".join(name_parts)
                    name = name.rstrip(", ")
                    return_dict[name] = {"time mean": mean_df, "time std": std_df}
            except ValueError:
                mean_df = compute_mean_std_runtime(alg)
                if 'fhefedavg' in str(path):
                    # Do not need total block time.
                    mean_df.pop("block")
                if 'orgfedavg' in str(path):
                    return_dict[f'FedAvg {name}'] = {"time mean": mean_df}
                elif 'secagg' in str(path):
                    return_dict[f'SecAgg {name}'] = {"time mean": mean_df}
                else:
                    # For FHE there will only be one scheme per table.
                    # Must escape the underscores so that the name is valid in overleaf.
                    # Escape underscores gir for lange navn. Prøve med komma og mellomrom.
                    name_parts = name.split('_')
                    name_parts = relabel_names(name_parts, sec_type)
                    if "logreg" in str(path):
                        # Don't need to denote chunk size for logreg.
                        for i in range(len(name_parts)):
                            if "nb" in name_parts[i]:
                                name_parts.pop(i)
                                break
                    name = ", ".join(name_parts)
                    name = name.rstrip(", ")
                    return_dict[name] = {"time mean": mean_df}
    elif table == "model":
        for alg in algorithms:
            name = alg.name
            if 'bert' in str(path):
                try:
                    loss_mean_df, loss_std_df, accuracy_mean_df, accuracy_std_df, f1_mean_df, f1_std_df = compute_mean_std_model_metrics(alg)
                    if 'orgfedavg' in str(path):
                        return_dict[f'FedAvg {name}'] = {"loss mean": loss_mean_df,
                                                        "loss std": loss_std_df,
                                                        "accuracy mean": accuracy_mean_df,
                                                        "accuracy std": accuracy_std_df,
                                                        "f1 mean": f1_mean_df,
                                                        "f1 std": f1_std_df}
                    elif 'secagg' in str(path):
                        return_dict[f'SecAgg {name}'] = {"loss mean": loss_mean_df,
                                                        "loss std": loss_std_df,
                                                        "accuracy mean": accuracy_mean_df,
                                                        "accuracy std": accuracy_std_df,
                                                        "f1 mean": f1_mean_df,
                                                        "f1 std": f1_std_df}
                    else:
                        # For FHE there will only be one scheme per table.
                        # Must escape the underscores so that the name is valid in overleaf.
                        # Escape underscores gir for lange navn. Prøve med komma og mellomrom.
                        name_parts = name.split('_')
                        name_parts = relabel_names(name_parts, sec_type)
                        if "logreg" in str(path):
                            # Don't need to denote chunk size for logreg.
                            for i in range(len(name_parts)):
                                if "nb" in name_parts[i]:
                                    name_parts.pop(i)
                                    break
                        name = ", ".join(name_parts)
                        name = name.rstrip(", ")
                        return_dict[name] = {"loss mean": loss_mean_df,
                                            "loss std": loss_std_df,
                                            "accuracy mean": accuracy_mean_df,
                                            "accuracy std": accuracy_std_df,
                                            "f1 mean": f1_mean_df,
                                            "f1 std": f1_std_df}
                except ValueError:
                    loss_mean_df, accuracy_mean_df, f1_mean_df = compute_mean_std_model_metrics(alg)
                    if 'orgfedavg' in str(path):
                        return_dict[f'FedAvg {name}'] = {"loss mean": loss_mean_df,
                                                        "accuracy mean": accuracy_mean_df,
                                                        "f1 mean": f1_mean_df}
                    elif 'secagg' in str(path):
                        return_dict[f'SecAgg {name}'] = {"loss mean": loss_mean_df,
                                                        "accuracy mean": accuracy_mean_df,
                                                        "f1 mean": f1_mean_df}
                    else:
                        # For FHE there will only be one scheme per table.
                        # Must escape the underscores so that the name is valid in overleaf.
                        # Escape underscores gir for lange navn. Prøve med komma og mellomrom.
                        name_parts = name.split('_')
                        name_parts = relabel_names(name_parts, sec_type)
                        if "logreg" in str(path):
                            # Don't need to denote chunk size for logreg.
                            for i in range(len(name_parts)):
                                if "nb" in name_parts[i]:
                                    name_parts.pop(i)
                                    break
                        name = ", ".join(name_parts)
                        name = name.rstrip(", ")
                        return_dict[name] = {"loss mean": loss_mean_df,
                                            "accuracy mean": accuracy_mean_df,
                                            "f1 mean": f1_mean_df}
            else:
                try:
                    loss_mean_df, loss_std_df, accuracy_mean_df, accuracy_std_df = compute_mean_std_model_metrics(alg)
                    if 'orgfedavg' in str(path):
                        return_dict[f'FedAvg {name}'] = {"loss mean": loss_mean_df,
                                                        "loss std": loss_std_df,
                                                        "accuracy mean": accuracy_mean_df,
                                                        "accuracy std": accuracy_std_df}
                    elif 'secagg' in str(path):
                        return_dict[f'SecAgg {name}'] = {"loss mean": loss_mean_df,
                                                        "loss std": loss_std_df,
                                                        "accuracy mean": accuracy_mean_df,
                                                        "accuracy std": accuracy_std_df}
                    else:
                        # For FHE there will only be one scheme per table.
                        # Must escape the underscores so that the name is valid in overleaf.
                        # Escape underscores gir for lange navn. Prøve med komma og mellomrom.
                        name_parts = name.split('_')
                        name_parts = relabel_names(name_parts, sec_type)
                        if "logreg" in str(path):
                            # Don't need to denote chunk size for logreg.
                            for i in range(len(name_parts)):
                                if "nb" in name_parts[i]:
                                    name_parts.pop(i)
                                    break
                        name = ", ".join(name_parts)
                        name = name.rstrip(", ")
                        return_dict[name] = {"loss mean": loss_mean_df,
                                            "loss std": loss_std_df,
                                            "accuracy mean": accuracy_mean_df,
                                            "accuracy std": accuracy_std_df}
                except ValueError:
                    loss_mean_df, accuracy_mean_df = compute_mean_std_model_metrics(alg)
                    if 'orgfedavg' in str(path):
                        return_dict[f'FedAvg {name}'] = {"loss mean": loss_mean_df,
                                                        "accuracy mean": accuracy_mean_df}
                    elif 'secagg' in str(path):
                        return_dict[f'SecAgg {name}'] = {"loss mean": loss_mean_df,
                                                        "accuracy mean": accuracy_mean_df}
                    else:
                        # For FHE there will only be one scheme per table.
                        # Must escape the underscores so that the name is valid in overleaf.
                        # Escape underscores gir for lange navn. Prøve med komma og mellomrom.
                        name_parts = name.split('_')
                        name_parts = relabel_names(name_parts, sec_type)
                        if "logreg" in str(path):
                            # Don't need to denote chunk size for logreg.
                            for i in range(len(name_parts)):
                                if "nb" in name_parts[i]:
                                    name_parts.pop(i)
                                    break
                        name = ", ".join(name_parts)
                        name = name.rstrip(", ")
                        return_dict[name] = {"loss mean": loss_mean_df,
                                            "accuracy mean": accuracy_mean_df}
    elif table == "container":
        for alg in algorithms:
            name = alg.name
            max_df, mean_df, std_df = compute_max_mean_std_container_metrics(alg, metric_type)
            if 'orgfedavg' in str(path):
                return_dict[f'FedAvg {name}'] = {"max": max_df, "mean": mean_df, "std": std_df}
            elif 'secagg' in str(path):
                return_dict[f'SecAgg {name}'] = {"max": max_df, "mean": mean_df, "std": std_df}
            else:
                # For FHE there will only be one scheme per table.
                # Must escape the underscores so that the name is valid in overleaf.
                # Escape underscores gir for lange navn. Prøve med komma og mellomrom.
                name_parts = name.split('_')
                name_parts = relabel_names(name_parts, sec_type)
                if "logreg" in str(path):
                    # Don't need to denote chunk size for logreg.
                    for i in range(len(name_parts)):
                        if "nb" in name_parts[i]:
                            name_parts.pop(i)
                            break
                name = ", ".join(name_parts)
                name = name.rstrip(", ")
                return_dict[name] = {"max": max_df, "mean": mean_df, "std": std_df}

    return return_dict

def relabel_names(name_parts, type_name="fhefedavg"):
    """Change the parts in name_parts to correspond to the correct terms in the thesis.
    nc -> C,
    sl -> $\\lambda$,
    rd -> N,
    pe -> tc,
    cs -> nb
    
    Also reformat powers of two into 2^n, e.g., 131072 -> $2^{17}$.
    Reformat chunk size X into number of blocks Y."""
    powers_of_two = {"8192": "$2^{13}$",
                     "16384": "$2^{14}$",
                     "131072": "$2^{17}$"}
    
    num_blocks = {
                  f"{int(np.ceil(727_627))}": "1",
                  f"{int(np.ceil(727_627/2))}": "2",
                  f"{int(np.ceil(727_627/3))}": "3",
                  f"{int(np.ceil(727_627/10))}": "10",
                  f"{int(np.ceil(727_627/21))}": "21",
                  f"{int(np.ceil(4_386_179/3))}": "3",
                  f"{int(np.ceil(4_386_179/8))}": "8",
                  f"{int(np.ceil(4_386_179/16))}": "16",
                  f"{int(np.ceil(4_386_179/18))}": "18",
                  f"{int(np.ceil(4_386_179/60))}": "60",
                  f"{int(np.ceil(4_386_179/126))}": "126",
                  }

    # Change nc to C.
    part_0 = name_parts[0]
    part_0 = part_0.lstrip("nc")
    name_parts[0] = "C" + part_0
    # Change sl to \lambda and remove trailing c.
    part_1 = name_parts[1]
    part_1 = part_1.lstrip("sl")
    part_1 = part_1.rstrip("c")
    name_parts[1] = "$\\lambda$" + part_1
    # Change rd to N and convert to 2^i notation.
    part_2 = name_parts[2]
    part_2 = part_2.lstrip("rd")
    name_parts[2] = "N" + powers_of_two[part_2]
    if type_name == "fhefedavg":
        # Change pe to tc.
        part_3 = name_parts[3]
        part_3 = part_3.lstrip("pe")
        name_parts[3] = "tc" + part_3

        if np.any(["cs" in name for name in name_parts]):
            # Change cs to nb.
            part_4 = name_parts[4]
            part_4 = part_4.lstrip("cs")
            # Change from chunk size to num blocks value.
            if part_4 in list(num_blocks.keys()):
                part_4 = num_blocks[part_4]
            else:
                part_4 = "XXX"
            name_parts[4] = "nb" + part_4
    elif type_name == "ckks-nf":
        if np.any(["cs" in name for name in name_parts]):
            # Change cs to nb.
            part_4 = name_parts[4]
            part_4 = part_4.lstrip("cs")
            # Change from chunk size to num blocks value.
            if part_4 in list(num_blocks.keys()):
                part_4 = num_blocks[part_4]
            else:
                part_4 = "XXX"
            name_parts[4] = "nb" + part_4

    # Keep mt as it is, if it is present.

    return name_parts
    

def create_container_metric_pds(path, type, allowed_paths):
    results = {}
    # Create filepaths.
    path = Path(path)
    algorithms = list(path.iterdir())
    # print(algorithms)
    # print(allowed_paths)
    # Filtrer etter de som er i allowed_paths. Plotte alle blir for mye.
    algorithms = [alg for alg in algorithms if alg.name in allowed_paths]

    # Go through each experiment.
    for alg in algorithms:
        path = Path(alg)
        name = alg.name
        runs = list(path.iterdir())
        if type == "network":
            # For network, kombiner sent og recv for hvert måletidspunkt.
            file_paths = [run/f"{type}" for run in runs]

            # Create dataframes for each path and remove rows with at least one '-'.
            dfs = []
            dfs_sent = []
            dfs_recv = []
            lenghts = []
            for filepath in file_paths:
                df_sent = pd.read_csv(str(filepath)+"_sent.csv", sep=",")
                # Remove rows which contain at least one '-'.
                num_bindestreker = (max([len(df_sent[df_sent[key] == "-"]) for key in list(df_sent.keys())]))
                if num_bindestreker > 0:
                    df_sent = df_sent.iloc[0:-num_bindestreker]
                df_sent = df_sent.astype(np.float64)
                lenghts.append(df_sent.shape[0])
                dfs_sent.append(df_sent)

                df_recv = pd.read_csv(str(filepath)+"_recv.csv", sep=",")
                # Remove rows which contain at least one '-'.
                num_bindestreker = (max([len(df_recv[df_recv[key] == "-"]) for key in list(df_recv.keys())]))
                if num_bindestreker > 0:
                    df_recv = df_recv.iloc[0:-num_bindestreker]
                df_recv = df_recv.astype(np.float64)
                lenghts.append(df_recv.shape[0])
                dfs_recv.append(df_recv)

            # Find shortest df and resize all of them to the same shape.
            shortest = np.min(lenghts)
            for i in range(len(dfs_sent)):
                dfs_sent[i] = dfs_sent[i].iloc[0:shortest]
                dfs_recv[i] = dfs_recv[i].iloc[0:shortest]
            
            # Merge dfs_sent and dfs_recv
            for i in range(len(dfs_sent)):
                df = dfs_sent[i] + dfs_recv[i]
                dfs.append(df)

        else:
            file_paths = [run/f"{type}.csv" for run in runs]

            # Create dataframes for each path and remove rows with at least one '-'.
            dfs = []
            lenghts = []
            for filepath in file_paths:
                df = pd.read_csv(filepath, sep=",")
                # Remove rows which contain at least one '-'.
                num_bindestreker = (max([len(df[df[key] == "-"]) for key in list(df.keys())]))
                if num_bindestreker > 0:
                    df = df.iloc[0:-num_bindestreker]
                df = df.astype(np.float64)
                lenghts.append(df.shape[0])
                dfs.append(df)

            # Find shortest df and resize all of them to the same shape.
            shortest = np.min(lenghts)
            for i in range(len(dfs)):
                dfs[i] = dfs[i].iloc[0:shortest]

        # Reformat container names.
        for i in range(len(dfs)):
            clipped_names_df = pd.DataFrame(dict())
            # Remove 'dockerfiles-' start of each name.
            # Remove trailing '-1' from each name.
            for key in list(dfs[i].keys()):
                new_key_list = key.split("-")
                new_key_list.pop(0)
                new_key_list.pop(-1)
                new_key = "-".join(new_key_list)

                clipped_names_df[new_key] = dfs[i][key]
            dfs[i] = clipped_names_df
        
        # Merge apps and supers.
        for i in range(len(dfs)):
            if "3" in str(path):
                dfs[i] = combine_containers(dfs[i], 3)
            elif "5" in str(path):
                dfs[i] = combine_containers(dfs[i], 5)
            elif "7" in str(path):
                dfs[i] = combine_containers(dfs[i], 7)

        # Ønsker å beregne gjennomsnitt av clients for hver PC.
        # For kryptering, la laptop-client-1 være utenfor snitt siden den launcher subprocess server
        # og har kanskje annen ressursbruk.
        average_dfs = [pd.DataFrame(dict()) for _ in range(len(dfs))]
        desktop_keys = [key for key in list(dfs[0].keys()) if "desktop" in key]
        laptop_keys = [key for key in list(dfs[0].keys()) if "laptop" in key]
        for i in range(len(dfs)):
            # Beregn average over nøklene for hvert run.
            # E.g., hvis det er to clients på desktop så summer verdiene ved hvert sekund og del på 2.
            average_dfs[i]['desktop average'] = np.mean([dfs[i][key] for key in desktop_keys], 0)
            if "fhefedavg" in str(path):
                # laptop-client-1 launcher server subprocess og har kanskje annen ressursbruk, så utelat den fra snitt.
                if "nc3" in str(path):
                    # For 3 clients har laptop bare en client,
                    # som betyr at laptop average blir tomt når man eksluderer laptop-client-1.
                    # Fyll den dermed 0.
                    average_dfs[i]['laptop average'] = np.zeros(shortest)
                else:
                    average_dfs[i]['laptop average'] = np.mean([dfs[i][key] for key in laptop_keys if "1" not in key], 0) 
                average_dfs[i]['client with server'] = dfs[i]['laptop-client-1']
            else:
                average_dfs[i]['laptop average'] = np.mean([dfs[i][key] for key in laptop_keys], 0)
            average_dfs[i]['server'] = dfs[i]['server']
        
        # Ta gjennomsnitt over tre runs.
        average_df = pd.DataFrame(dict())
        for key in list(average_dfs[0].keys()):
            average_df[key] = sum([average_dfs[i][key] for i in range(len(average_dfs))])/len(average_dfs)

        # Om tre clients, dropp client average på bærbar.
        if "nc3" in str(path) and "fhefedavg" in str(path):
            average_df.pop("laptop average")

        # Pynt på navn til å være samme format som tabeller.
        name_parts = name.split('_')
        if "secagg" in str(path) or "orgfedavg" in str(path):
            # Secagg skal bare få mellomrom, ingen annen formatering.
            name = ", ".join(name_parts)
        else:
            if "BGV" in str(path) or "BFV" in str(path):
                name_parts = relabel_names(name_parts, "fhefedavg")
                name = ", ".join(name_parts)
            else:
                # CKKS.
                name_parts = relabel_names(name_parts, "ckks-nf")
                name = ", ".join(name_parts)

        if type != "cpu":
            def b_to_mb(x):
                # MB = 10^6 B.
                return x / 1e6
            # Convert from bytes to MB.
            average_df = average_df.apply(b_to_mb, raw=True)

        results[name] = average_df

    return results


if __name__ == "__main__":
    # mean_runs_no_time("./benchmarking_results/mnist-orgfedavg/nc3", "accuracy.csv", False)
    # mean_time = mean_runs_no_time("./benchmarking_results/imdb-orgfedavg/nc3", "runtime.csv", True)
    # mean_time = mean_time['runtime'][0]

    # org = combine_containers(mean_runs_time("./benchmarking_results/logreg-fhefedavg/BGV/nc5_sl128c_rd131072_pe32_cs1310720", "memory.csv", 5),5)
    # plot_df(org)
    # ckks = mean_runs_time("./benchmarking_results/mnist-fhefedavg/CKKS/nc3", "cpu.csv", 3)

    # plot_ROC_AUC('./benchmarking_results/logreg-orgfedavg/nc3')
    # plot_ROC_AUC_multiple(['./benchmarking_results/mnist-orgfedavg/nc3',
    #                        './benchmarking_results/mnist-fhefedavg/CKKS/nc3',
    #                        './benchmarking_results/mnist-secagg/nc3'],
    #                        ["orgfedavg", "secagg", "fhefedavg"],
    #                        "mnist")
    # plot_ROC_AUC_multiple(['./benchmarking_results/imdb-orgfedavg/nc3',
    #                        './benchmarking_results/imdb-secagg/nc3',
    #                        './benchmarking_results/imdb-fhefedavg/CKKS/nc3'],
    #                        ["orgfedavg", "secagg", "fhefedavg"],
    #                        "imdb")

    # for metric in ["cpu", "memory", "cache", "network_recv", "network_sent"]:
    #     for dataset_name in ["mnist", "imdb"]:
    #         plot_metric_multiple(metric, 
    #                             [combine_containers(mean_runs_time(f"./benchmarking_results/{dataset_name}-orgfedavg/nc3", f"{metric}.csv", 3),3),
    #                             combine_containers(mean_runs_time(f"./benchmarking_results/{dataset_name}-secagg/nc3", f"{metric}.csv", 3),3),
    #                             combine_containers(mean_runs_time(f"./benchmarking_results/{dataset_name}-fhefedavg/CKKS/nc3", f"{metric}.csv", 3),3)],
    #                             ["orgfedavg", "secagg", "fhefedavg"],
    #                             dataset_name)
    # plotnine_plot_df(org)

    """Burde også kanskje plotte accuracy, loss og f1-score i tillegg?
    Per round eller noe?"""

    """Må fortsatt average over clients på samme maskin.
    Plote kurve for en client og for average per maskin.
    Client som hoster subprocess server kan plottes alene uansett,
    for å se om den bruker ekstra ressurser eller noe."""

    # plot_ROC_AUC_multiple(['./benchmarking_results/mnist-orgfedavg/nc3'],
    #                       ["orgfedavg","",""],
    #                       "mnist")

    # plot_ROC_AUC("./benchmarking_results/mnist-orgfedavg/nc3")

    # format_table_runtime(create_dict_for_whole_dir("./benchmarking_results/bert-fhefedavg/CKKS-NF",
    #                                                          "runtime",
    #                                                          "",
    #                                                          "fhefedavg"),
    #                                "ckks-nf")

    # orgfedavg = create_dict_for_whole_dir("./benchmarking_results/bert-orgfedavg",
    #                                                          "model",
    #                                                          "",
    #                                                          "fhefedavg")
    # secagg = create_dict_for_whole_dir("./benchmarking_results/bert-secagg",
    #                                                          "model",
    #                                                          "",
    #                                                          "fhefedavg")
    # for key in list(secagg.keys()):
    #     orgfedavg[key] = secagg[key]
    # format_table_loss_acc(orgfedavg,
    #                       "bert")

    # format_table_loss_acc(create_dict_for_whole_dir("./benchmarking_results/squeezenet-fhefedavg/CKKS-NF",
    #                                                          "model",
    #                                                          "",
    #                                                          "fhefedavg"),
    #                       "squeezenet")

    # format_table_container_metrics(create_dict_for_whole_dir("./benchmarking_results/logreg-fhefedavg/CKKS-NF",
    #                                                          "container",
    #                                                          "network",
    #                                                          "fhefedavg"),
    #                                "fhefedavg", False)

    """Plott av Squeezenet kjøringer. for minnebruk som kan brukes i diskusjon."""
    # Plot SecAgg for sammenligning.
    secagg_allowed_paths = ['nc3']
    # Plott BGV med og uten polynomial encoding.
    bgv_allowed_paths = ['nc3_sl256c_rd131072_pe16_cs727627', 'nc3_sl256c_rd131072_pe16_cs242543']
    # Plott CKKS med flere og færre blokker.
    ckks_allowed_paths = ['nc3_sl256c_rd131072_dbhigh_cs727627', 'nc3_sl256c_rd131072_dbhigh_cs242543']
    # PLott BFV.
    bfv_allowed_paths = ['nc3_sl256c_rd131072_pe16_cs242543_mtHPS']
    # Lag pds.
    secagg_pds = create_container_metric_pds("./benchmarking_results/squeezenet-secagg", "memory", secagg_allowed_paths)
    bgv_pds = create_container_metric_pds("./benchmarking_results/squeezenet-fhefedavg/BGV", "memory", bgv_allowed_paths)
    ckks_pds = create_container_metric_pds("./benchmarking_results/squeezenet-fhefedavg/CKKS-NF", "memory", ckks_allowed_paths)
    bfv_pds = create_container_metric_pds("./benchmarking_results/squeezenet-fhefedavg/BFV", "memory", bfv_allowed_paths)
    total_pds = {"SecAgg: C3": secagg_pds['nc3']}
    for key in bgv_allowed_paths:
        key = ", ".join(relabel_names(key.split("_"), "fhefedavg"))
        total_pds["BGV: "+key] = bgv_pds[key]
    for key in bfv_allowed_paths:
        key = ", ".join(relabel_names(key.split("_"), "fhefedavg"))
        total_pds["BFV: "+key] = bfv_pds[key]
    for key in ckks_allowed_paths:
        key = ", ".join(relabel_names(key.split("_"), "ckks-nf"))
        total_pds["CKKS: "+key] = ckks_pds[key]
    pprint(list(total_pds.keys()))

    plot_metric_multiple("Memory", 
                         total_pds, 
                         "Squeezenet", 
                         2, 
                         3)
    
    """Plott av Squeezenet kjøringer. for minnebruk som kan brukes i diskusjon."""
    # Plot SecAgg for sammenligning.
    secagg_allowed_paths = ['nc3']
    # Plott BGV med og uten polynomial encoding.
    bgv_allowed_paths = ['nc3_sl256c_rd131072_pe16_cs727627', 'nc3_sl256c_rd131072_pe16_cs242543']
    # Plott CKKS med flere og færre blokker.
    ckks_allowed_paths = ['nc3_sl256c_rd131072_dbhigh_cs727627', 'nc3_sl256c_rd131072_dbhigh_cs242543']
    # PLott BFV.
    bfv_allowed_paths = ['nc3_sl256c_rd131072_pe16_cs242543_mtHPS']
    # Lag pds.
    secagg_pds = create_container_metric_pds("./benchmarking_results/squeezenet-secagg", "network", secagg_allowed_paths)
    bgv_pds = create_container_metric_pds("./benchmarking_results/squeezenet-fhefedavg/BGV", "network", bgv_allowed_paths)
    ckks_pds = create_container_metric_pds("./benchmarking_results/squeezenet-fhefedavg/CKKS-NF", "network", ckks_allowed_paths)
    bfv_pds = create_container_metric_pds("./benchmarking_results/squeezenet-fhefedavg/BFV", "network", bfv_allowed_paths)
    total_pds = {"SecAgg: nc3": secagg_pds['nc3']}
    for key in bgv_allowed_paths:
        key = ", ".join(relabel_names(key.split("_"), "fhefedavg"))
        total_pds["BGV: "+key] = bgv_pds[key]
    for key in bfv_allowed_paths:
        key = ", ".join(relabel_names(key.split("_"), "fhefedavg"))
        total_pds["BFV: "+key] = bfv_pds[key]
    for key in ckks_allowed_paths:
        key = ", ".join(relabel_names(key.split("_"), "ckks-nf"))
        total_pds["CKKS: "+key] = ckks_pds[key]

    plot_metric_multiple("Network", 
                         total_pds, 
                         "Squeezenet", 
                         2, 
                         3)

    plot_ROC_AUC_multiple([['./benchmarking_results/squeezenet-secagg/nc3',
                            './benchmarking_results/squeezenet-fhefedavg/BGV/nc3_sl256c_rd131072_pe16_cs727627',
                            './benchmarking_results/squeezenet-fhefedavg/BGV/nc3_sl256c_rd131072_pe16_cs242543'],
                           ['./benchmarking_results/squeezenet-fhefedavg/BFV/nc3_sl256c_rd131072_pe16_cs242543_mtHPS',
                            './benchmarking_results/squeezenet-fhefedavg/CKKS-NF/nc3_sl256c_rd131072_dbhigh_cs727627',
                            './benchmarking_results/squeezenet-fhefedavg/CKKS-NF/nc3_sl256c_rd131072_dbhigh_cs242543']],
                          [['a) SecAgg: C3',
                            'b) BGV: C3, $\\lambda$256, N$2^{17}$, tc16, nb1',
                            'c) BGV: C3, $\\lambda$256, N$2^{17}$, tc16, nb3'],
                           ['d) BFV: C3, $\\lambda$256, N$2^{17}$, tc16, nb3, mtHPS',
                            'e) CKKS: C3, $\\lambda$256, N$2^{17}$, dbhigh, nb1',
                            'f) CKKS: C3, $\\lambda$256, N$2^{17}$, dbhigh, nb3']],
                            "Squeezenet",
                            2,
                            3)