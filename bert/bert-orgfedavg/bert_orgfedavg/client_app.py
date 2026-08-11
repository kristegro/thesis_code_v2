import time
from pathlib import Path

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from transformers import AutoModelForSequenceClassification

from bert_orgfedavg.task import get_weights, load_data, set_weights, test, train


# Flower client
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, testloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        print("Beginning training!")
        start_time = time.time()
        train(self.net, self.trainloader, epochs=self.local_epochs, device=self.device)
        end_time = time.time()
        print("Finished training!")
        # Lag path hvis den ikke eksisterer.
        current_path = Path("./storage/bert-orgfedavg-training-time")
        Path.mkdir(current_path,
               mode=0o777,
               parents=True, 
               exist_ok=True)
        Path.chmod(current_path, mode=0o777)
        """Må nå finne ut hvor mange runs som er gjort hittil."""
        num_dirs = len(list(current_path.iterdir()))
        """Mappe for run skal ha navn 'run{num_dirs+1}',
        altså første run for navn 'run1' osv."""
        run_dir = f"run{num_dirs+1}/"
        Path.mkdir(current_path/run_dir, mode=0o777, exist_ok=True)
        Path.chmod(current_path/run_dir, mode=0o777)
        Path.touch(current_path/run_dir/"time.txt", 0o777)
        Path.chmod(current_path/run_dir/"time.txt", mode=0o777)
        with open(Path(current_path/run_dir/"time.txt"), "a") as outfile:
            outfile.write(str(end_time-start_time))

        return get_weights(self.net), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        print("Beginning evaluation!")
        start_time = time.time()
        loss, accuracy, f1 = test(self.net, self.testloader, self.device)
        end_time = time.time()
        print("Finished evaluation!")

        # Lag path hvis den ikke eksisterer.
        current_path = Path("./storage/bert-orgfedavg-evaluation-time")
        Path.mkdir(current_path,
               mode=0o777,
               parents=True, 
               exist_ok=True)
        Path.chmod(current_path, mode=0o777)
        """Må nå finne ut hvor mange runs som er gjort hittil."""
        num_dirs = len(list(current_path.iterdir()))
        """Mappe for run skal ha navn 'run{num_dirs+1}',
        altså første run for navn 'run1' osv."""
        run_dir = f"run{num_dirs+1}/"
        Path.mkdir(current_path/run_dir, mode=0o777, exist_ok=True)
        Path.chmod(current_path/run_dir, mode=0o777)
        Path.touch(current_path/run_dir/"time.txt", 0o777)
        Path.chmod(current_path/run_dir/"time.txt", mode=0o777)
        with open(Path(current_path/run_dir/"time.txt"), "a") as outfile:
            outfile.write(str(end_time-start_time))

        return float(loss), len(self.testloader.dataset), {"accuracy": accuracy, "f1": f1}


def client_fn(context: Context):

    # Get this client's dataset partition
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    model_name = context.run_config["model-name"]
    trainloader, valloader = load_data(partition_id, num_partitions, model_name)

    # Load model
    num_labels = context.run_config["num-labels"]
    net = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(net, trainloader, valloader, local_epochs).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
