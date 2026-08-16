import time
from pathlib import Path

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
import torch

import orgfedavg.tasks.logreg_task as lr_task
import orgfedavg.tasks.squeezenet_task as sq_task
import orgfedavg.tasks.bert_task as b_task

# Flower client
class FlowerClient(NumPyClient):
    def __init__(self, model, trainloader, valloader, local_epochs,
                 set_weights_func, get_weights_func, train_func, test_func):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.set_weights = set_weights_func
        self.get_weights = get_weights_func
        self.train = train_func
        self.test = test_func
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def fit(self, parameters, config):
        # Update model weights.
        self.set_weights(self.model, parameters)
        
        print("Beginning training!")
        # Fit model to the data.
        start_time = time.time()
        self.train(self.model, self.trainloader, self.local_epochs, self.device)
        end_time = time.time()
        print("Finished training!")

        # Lag path hvis den ikke eksisterer.
        current_path = Path("./storage/logreg-orgfedavg-training-time")
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
        return self.get_weights(self.model), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        # Update model weights.
        self.set_weights(self.model, parameters)

        # Evaluate model on the data
        print("Beginning evaluation!")
        start_time = time.time()
        loss, accuracy = self.test(self.model, self.valloader, self.device)
        end_time = time.time()
        print("Finished evaluation!")

        # Lag path hvis den ikke eksisterer.
        current_path = Path("./storage/logreg-orgfedavg-evaluation-time")
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

    # Load the data and create model instance according to chosen model.
    model_choice = context.run_config["model"]
    if model_choice == "logreg":
        trainloader, valloader = lr_task.load_data(context)
        model = lr_task.create_model(context)
        set_weights = lr_task.set_weights
        get_weights = lr_task.get_weights
        train = lr_task.train
        test = lr_task.test
    elif model_choice == "squeezenet":
        trainloader, valloader = sq_task.load_data(context)
        model = sq_task.create_model(context)
        set_weights = sq_task.set_weights
        get_weights = sq_task.get_weights
        train = sq_task.train
        test = sq_task.test
    elif model_choice == "bert":
        trainloader, valloader = b_task.load_data(context)
        model = b_task.create_model(context)
        set_weights = b_task.set_weights
        get_weights = b_task.get_weights
        train = b_task.train
        test = b_task.test

    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(model, trainloader, valloader, local_epochs, 
                        set_weights, get_weights, train, test).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
