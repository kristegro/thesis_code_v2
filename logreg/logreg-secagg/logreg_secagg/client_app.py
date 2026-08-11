import time
from pathlib import Path

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flwr.client.mod import secagg_mod
import torch

from logreg_secagg.task import (load_data, 
                                   set_weights, 
                                   get_weights, 
                                   train, 
                                   test, 
                                   LogisticRegression)

# Flower client
class FlowerClient(NumPyClient):
    def __init__(self, model, trainloader, valloader, local_epochs):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def fit(self, parameters, config):
        # Update model weights.
        set_weights(self.model, parameters)
        
        print("Beginning training!")
        # Fit model to the data.
        start_time = time.time()
        train(self.model, self.trainloader, self.local_epochs, self.device)
        end_time = time.time()
        print("Finished training!")

        # Lag path hvis den ikke eksisterer.
        current_path = Path("./storage/logreg-secagg-training-time")
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
        return get_weights(self.model), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        # Update model weights.
        set_weights(self.model, parameters)

        # Evaluate model on the data
        print("Beginning evaluation!")
        start_time = time.time()
        loss, accuracy = test(self.model, self.valloader, self.device)
        end_time = time.time()
        print("Finished evaluation!")

        # Lag path hvis den ikke eksisterer.
        current_path = Path("./storage/logreg-secagg-evaluation-time")
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

        return float(loss), len(self.valloader.dataset), {"accuracy": float(accuracy)}


def client_fn(context: Context):

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    scale = context.run_config["scale"]
    local_epochs = context.run_config["local-epochs"]

    trainloader, valloader = load_data(partition_id, num_partitions, scale)

    # Load the model and initialize it with the received weights
    # 28*28 = 784, so 784 features.
    if scale == "classic":
        model = LogisticRegression(28*28, 10)
    elif scale == "larger":
        model = LogisticRegression(224*224, 10)
    elif scale == "rgb":
        model = LogisticRegression(3*224*224, 10)

    # Return Client instance
    return FlowerClient(model, trainloader, valloader, local_epochs).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
    mods=[
        secagg_mod,
    ],
)
