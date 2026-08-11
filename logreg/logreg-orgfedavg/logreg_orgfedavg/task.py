from collections import OrderedDict

import torch
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda, Resize, ToTensor
from torch.optim import AdamW
import torch.nn as nn

fds = None  # Cache FederatedDataset


def load_data(
    partition_id,
    num_partitions,
    scale="classic"
) -> tuple[DataLoader, DataLoader]:
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="ylecun/mnist",
            partitioners={"train": partitioner},
            trust_remote_code=True,
        )
    partition = fds.load_partition(partition_id, "train")

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
    # 20 % for on federated evaluation
    partition_full = partition.train_test_split(test_size=0.2, seed=42)
    trainloader = DataLoader(
        partition_full["train"],
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn,
    )
    valloader = DataLoader(
        partition_full["test"],
        batch_size=32,
        collate_fn=collate_fn,
    )
    return trainloader, valloader


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


def get_weights(net):
    """Convert OrderedDict[torch.tensor] which pytorch uses for weights
    into list[np.array] which can be sent through Flower."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    """Convert from list[np.array] or np.array(np.array) used by Flower
    into OrderedDict[torch.tensor] used by pytorch for weights."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def train(net, trainloader, epochs, device):
    optimizer = AdamW(net.parameters(), lr=5e-5)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            # print(batch)
            # exit()
            # batch = {k: v.to(device) for k, v in batch.items()}
            images = batch[0].to(device)
            labels = batch[1].to(device)
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch[0].to(device)
            labels = batch[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy

