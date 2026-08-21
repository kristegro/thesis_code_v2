import os
import sys
import time
from logging import INFO
from random import sample
import gc # garbage collection
from math import ceil
from pathlib import Path

from flwr.common import (Context,
                         RecordDict, 
                         Message, 
                         ConfigRecord,
                         ArrayRecord)
from flwr.common.logger import log
from flwr.server import Grid, ServerApp
import torch
from prometheus_client import Gauge, start_http_server, Counter

from logreg_fhefedavg.task import get_weights, set_weights, LogisticRegression
from logreg_fhefedavg.fhe.general_server import server_fhe

# Add /app, which contains all help functions, to import path.
module_dir = os.path.abspath("/app")
if module_dir not in sys.path:
    sys.path.insert(0, module_dir)


# Create ServerApp
app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:

    # Aggresiv garbage collection.
    gc.collect()

    # Starting time.
    start_time = time.time()

    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    fraction_evaluation = 1.0

    # Initialize global model
    
    log(INFO, "Initializing global model.")
    # Initialize global model
    scale = context.run_config["scale"]
    # Load the model and initialize it with the received weights
    if scale == "classic":
        model = LogisticRegression(28*28, 10)
    elif scale == "larger":
        model = LogisticRegression(224*224, 10)
    elif scale == "rgb":
        model = LogisticRegression(3*224*224, 10)
    weights = get_weights(model)

    # old_weights = get_weights(net)
    log(INFO, f"type(weights) = {type(weights)}.")
    log(INFO, f"len(weights) = {len(weights)}.")
    log(INFO, f"type(weights[0]) = {type(weights[0])}.")
    log(INFO, f"len(weights[0]) = {len(weights[0])}.")

    """Prometheus setup:"""
    # Create labels for each round.
    labels = ["name"]

    # Define a gauge to track the global model accuracy
    accuracy_gauge = Gauge("model_accuracy", 
                           "Current accuracy of the global model",
                           labelnames=labels)
    
    # Define a gauge to track the global model loss
    loss_gauge = Gauge("model_loss", 
                       "Current loss of the global model",
                       labelnames=labels)
    
    # Define counter to track when everything is done.
    done_counter = Counter("done_counter",
                           "Is everything done?")
    # Reset in case the value lingers in Prometheus.
    done_counter.reset()
    
    # Initialize labels for all gauges
    for i in range(1, num_rounds+1):
        accuracy_gauge.labels(name=f"round-{i}")
        loss_gauge.labels(name=f"round-{i}")

    # Start Prometheus Metric server on the specified port.
    start_http_server(8000)

    
    # Loop and wait until enough nodes are available.
    min_nodes = context.run_config['num-clients']
    node_ids = []
    while len(node_ids) < min_nodes:
        node_ids = list(grid.get_node_ids())
        log(INFO, "Waiting for nodes to connect...")
        time.sleep(2)
    
    n = len(node_ids)

    log(INFO, "All nodes have connected.")

    log(INFO, "Starting computation of FedAvg.")

    # Time right before starting first round.
    log(INFO, f"Time used for initialization: {time.time()-start_time} s.")

    # FedAvg computation.
    for round in range(1, num_rounds+1):
        round_start = time.time()
        # For each round 1, ..., num_rounds.
        log(INFO, "")
        log(INFO, f"Doing computation for Round {round}.")

        # Assume that weights are decrypted at the end of 
        # each round, i.e., at this point they are always plaintexts.
        # Thus, they can be sent directly to clients.

        # Make clients do current round of training.
        # Computation should only be done by a subset of clients,
        # equal to max(fraction_fit*len(node_ids), 1).
        # Pick subset to participate.
        participating = sample(node_ids, k=max(ceil(fraction_fit*n), 1))
        num_part = len(participating)
        # Send message to each participating client.
        log(INFO, f"Round {round}; Sending messages for training.")
        messages = []
        for j in range(num_part):
            recordset = RecordDict()
            round_record = ConfigRecord({'id': j})
            """weights is a list of ndarrays, and so it cannot 
            be sent in ConfigRecord. Must instead send as ArrayRecord."""
            array_record = ArrayRecord(numpy_ndarrays=weights)
            recordset['config'] = round_record
            recordset['weights'] = array_record
            # log(INFO, f"recordset.keys() = {recordset.keys()}")
            message = Message(
                content=recordset,
                message_type="train",  
                # target train() method in ClientApp
                dst_node_id=participating[j],
                group_id=str(round),
            )
            messages.append(message)
        replies = grid.send_and_receive(messages)
        log(INFO, f"Round {round}; Received replies.")

        # Aggresiv garbage collection.
        del messages
        gc.collect()


        """Fra dette punktet blir det forskjellig om det er orgfedavg eller fhefedavg."""
        # Type of security, i.e., none/orfedavg or fhefedavg
        sec_type = context.run_config["sec-type"]

        if sec_type == "fhefedavg":
            weights = server_fhe(grid, context, node_ids, participating, replies)
        else:
            # sec_type == "orgfedavg"
            # Simply sum each array in the lists received from each client,
            # extract the size sum from the first array and divide.

            # Create list with as many slots as there are arrays to sum.
            updates = replies[0].content.array_records['updates']
            for i in range(len(updates)):
                for j in range(1, len(replies)):
                    # Add with array from each client.
                    updates[i] += replies[j].content.array_records['updates'][i]

            # Extract first element, the sum of training set sizes, and compute division.
            inv_sizes = 1/(weights[0][0])
            weights = [weights[i]*inv_sizes for i in range(1, len(weights))]

        del replies
        gc.collect()

        """At this point weights contains the proper
        aggregated global model update, and aggregation is finished."""
        # STOPP TIDTAGNING av benchmark her!
        end_time = time.time()
        # Lag path hvis den ikke eksisterer.
        current_path = Path("./storage/logreg-block-time")
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
    
        # Time after completing round i.
        log(INFO, f"Time to complete Round {round}: {time.time()-round_start} s.")

        """Evaluation should also be done each round,
        as that is how it was done in the non-FHE version."""
        log(INFO, "Doing evaluation.")
        # Only a subset, possibly, should participate,
        # denoted by fraction_evaluation.
        participating = sample(node_ids, k=max(ceil(fraction_evaluation*n), 1))
        num_part = len(participating)
        # Send message to each participating client.
        log(INFO, "Sending evaluation message to clients.")
        messages = []
        for j in range(num_part):
            recordset = RecordDict()
            round_record = ConfigRecord({'id': j})
            """weights is a list of ndarrays, and so it cannot 
            be sent in ConfigRecord. Must instead send as ArrayRecord."""
            array_record = ArrayRecord(numpy_ndarrays=weights)
            recordset['config'] = round_record
            recordset['weights'] = array_record
            message = Message(
                content=recordset,
                message_type="evaluate",  
                # target evaluate() method in ClientApp
                dst_node_id=participating[j],
                group_id="evaluate",
            )
            messages.append(message)
        replies = grid.send_and_receive(messages)
        log(INFO, "Received replies.")

        # Aggresiv garbage collection.
        del messages
        gc.collect()

        """Compute weighted average of losses as well."""
        j = 0
        # raise Exception("Right before server prints evaluation results from nodes.")
        loss_sum = 0
        accuracy_sum = 0
        size_sum = 0
        for reply in replies:
            # Log results from clients.
            loss = reply.content.config_records['results']['loss']
            size = reply.content.config_records['results']['size']
            accuracy = reply.content.config_records['results']['accuracy']

            loss_sum += loss*size
            size_sum += size
            # accuracy is just a number, not a dict.
            accuracy_sum += accuracy*size
        
        loss_average = loss_sum/size_sum
        accuracy_average = accuracy_sum/size_sum

        # Aggresiv garbage collection.
        del replies
        gc.collect()

        # Update the Prometheus gauges with the latest aggregated values
        loss_gauge.labels(f"round-{round}").set(loss_average)
        accuracy_gauge.labels(f"round-{round}").set(accuracy_average)

        log(INFO, f"Round {round}; Weighted average of loss = {loss_average}")
        log(INFO, f"Round {round}; Weighted average of accuracy = {accuracy_average}")
        # with open("result_files/benchmark-openfhe-fedavg-docker.txt", "a") as outfile:
        #     outfile.write(f"Round {round}; Weighted average of loss = {loss_average}\n")
        #     outfile.write(f"Round {round}; Weighted average of accuracy = {accuracy_average}\n")
        #     outfile.write(f"Round {round}; Runtime for weights aggregation = {benchmark_time_end-benchmark_time_start}\n\n")
        log(INFO, f"Round {round}; Evaluation completed. Round done.")

        # Save to be able to create plots later.
        set_weights(model, weights)
        storage = os.path.abspath('storage/') + "/"
        torch.save(model.state_dict(), storage+f"logreg-tmp-r{round}")
        path = Path(storage+f"logreg-tmp-r{round}")
        Path.chmod(path, mode=0o777)
    

    # Final runtime.
    end_time = time.time()
    log(INFO, f"Final runtime for FedAvg-FHE: {end_time-start_time} s.")
    # with open("result_files/benchmark-openfhe-fedavg.txt", "a") as outfile:
    #     outfile.write(f"Final runtime for FedAvg-FHE: {end_time-start_time} s.\n")

    # Increment done_counter to show that everything is done.
    done_counter.inc()


    """Various bits of code to verify correctness of results."""
    # assert len(weights) == len(new_weights)
    # for i in range(len(weights)):
    #     assert len(weights[i]) == len(new_weights[i])
    #     for j in range(len(weights[i])):
    #         if isinstance(weights[i][j], np.ndarray):
    #             assert len(weights[i][j]) == len(new_weights[i][j])
    #             assert np.allclose(new_weights[i][j], weights[i][j])
    #         else:
    #             assert np.allclose(new_weights[i], weights[i])

    """Check that the weights after all rounds
    are approximately unchanged when no training is done.
    They are. The differences are minor."""
    # log(INFO, "Loading fresh model.")
    # model_name = context.run_config["model-name"]
    # num_labels = context.run_config["num-labels"]
    # net = AutoModelForSequenceClassification.from_pretrained(
    #     model_name, num_labels=num_labels
    # )

    # org_weights = get_weights(net)

    # for i in range(len(weights)):
    #     # Both weights and org_weights are lists of arrays.
    #     # Can compare each array element.
    #     # assert np.allclose(org_weights[i], weights[i])
    #     """Get assertation error somewhere. Maybe because of
    #     np.allclose() parameters.
    #     The differences seem to be very small,
    #     so probably within accepted bounds."""

    # with open("result_files/new_weights_no_training.txt", "a") as outfile:
    #     for p in range(len(struct)):
    #         i, j, k = struct[p]
    #         if k == -1:
    #             outfile.write(f"new_weights[{i},{j}] = {weights[i][j]}\n")
    #         else:
    #             outfile.write(f"new_weights[{i},{j},{k}] = {weights[i][j,k]}\n")
    
    # with open("result_files/org_weights_no_training.txt", "a") as outfile:
    #     for p in range(len(struct)):
    #         i, j, k = struct[p]
    #         if k == -1:
    #             outfile.write(f"org_weights[{i},{j}] = {org_weights[i][j]}\n")
    #         else:
    #             outfile.write(f"org_weights[{i},{j},{k}] = {org_weights[i][j,k]}\n")