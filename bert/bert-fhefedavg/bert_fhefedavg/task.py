import warnings
from collections import OrderedDict

import torch
import transformers
from datasets.utils.logging import disable_progress_bar
from evaluate import load as load_metric
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
disable_progress_bar()
transformers.logging.set_verbosity_error()


fds = None  # Cache FederatedDataset


def load_data(partition_id: int, num_partitions: int, model_name: str):
    """Load IMDB data (training and eval)"""
    # Aggresiv garbage collection.
    # gc.collect()
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="stanfordnlp/imdb",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    # Divide data: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], truncation=True, add_special_tokens=True, max_length=512
        )

    partition_train_test = partition_train_test.map(tokenize_function, batched=True)
    partition_train_test = partition_train_test.remove_columns("text")
    partition_train_test = partition_train_test.rename_column("label", "labels")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainloader = DataLoader(
        partition_train_test["train"],
        shuffle=True,
        batch_size=32,
        collate_fn=data_collator,
    )

    testloader = DataLoader(
        partition_train_test["test"], batch_size=32, collate_fn=data_collator
    )

    # Aggresiv garbage collection.
    # gc.collect()
    return trainloader, testloader


def train(net, trainloader, epochs, device):
    # Aggresiv garbage collection.
    # gc.collect()
    optimizer = AdamW(net.parameters(), lr=5e-5)
    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = net(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    # Aggresiv garbage collection.
    # gc.collect()


def test(net, testloader, device):
    # Aggresiv garbage collection.
    # gc.collect()
    metric = load_metric("accuracy")
    metric_f1 = load_metric("f1")
    loss = 0
    net.eval()
    for batch in testloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = net(**batch)
        logits = outputs.logits
        loss += outputs.loss.item()
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
        metric_f1.add_batch(predictions=predictions, references=batch["labels"])
    loss /= len(testloader.dataset)
    accuracy = metric.compute()["accuracy"]
    f1 = metric_f1.compute()["f1"]
    # Aggresiv garbage collection.
    # gc.collect()
    return loss, accuracy, f1


def get_weights(net):
    # Aggresiv garbage collection.
    # gc.collect()
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    # Aggresiv garbage collection.
    # gc.collect()
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
    # Aggresiv garbage collection.
    # gc.collect()
