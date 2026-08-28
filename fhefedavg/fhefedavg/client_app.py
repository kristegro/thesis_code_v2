import os
import sys
from logging import INFO
import time
import gc # garbage collection
import time
from pathlib import Path

from flwr.client import ClientApp
from flwr.common import (Context, 
                         Message, 
                         ConfigRecord, 
                         RecordDict, 
                         ArrayRecord)
from flwr.common.logger import log
import torch
import numpy as np

import fhefedavg.tasks.logreg_task as lr_task
import fhefedavg.tasks.squeezenet_task as sq_task
import fhefedavg.tasks.bert_task as b_task

# Add /app, which contains all help functions, to import path.
module_dir = os.path.abspath("/app")
if module_dir not in sys.path:
    sys.path.insert(0, module_dir)

from fhefedavg.fhe.keygen import (start_forward_keygen_base, 
                    continue_forward_keygen_base, 
                    end_forward_keygen_base,
                    continue_backward_keygen_base,
                    start_forward_keygen_noek_base, 
                    continue_forward_keygen_noek_base, 
                    end_forward_keygen_noek_base,
                    continue_backward_keygen_noek_base)
from fhefedavg.fhe.general_client import (find_struct,
                                                 get_pds as get_pds_imp,
                                                 send_pds as send_pds_imp,
                                                 weights_blocks as weights_blocks_imp)
from fhefedavg.fhe.noise_flooding_client import (dist_pds_no_server as dist_pds_no_server_imp,
                                                        create_single_ciphertext as create_single_ciphertext_imp)


# Flower ClientApp
app = ClientApp()

@app.train()
def training(msg: Message, context: Context):
    
    """Handles training of the model.
    Receives weights from server which have already been 
    decrypted, due to how the weighted average is computed.
    Due to how OpenFHE can only encrypt lists of data,
    the weights must be encrypted piecemeal.
    After the model is trained, 
    the weights are saved into the state of the node,
    together with an overview of the weight structure.
    
    Only the encrypted dataset size
    and the structure overview are returned 
    to the server.
    The transfer of the weights themselves
    is handled by the method weights_blocks()
    above."""

    # Aggresiv garbage collection.
    gc.collect()

    # Get contents of message.
    id = msg.content.config_records['config']['id']
    weights = msg.content.array_records['weights']
    """weights is here a Flower ArrayRecord, 
    a dict with keys '0', '1', ..., and values 
    of Flower Array. 
    Must convert weights into a list of ndarrays again."""
    # Weights is now list of ndarrays, as it was on the server.
    weights = [weights[key].numpy() for key in list(weights.keys())] 

    # Get this client's dataset partition.
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    local_epochs = context.run_config["local-epochs"]

    # Load model depending on chosen task.
    model_choice = context.run_config["model"]
    if model_choice == "logreg":
        trainloader, valloader = lr_task.load_data(context)
        model = lr_task.create_model(context)
        set_weights = lr_task.set_weights
        get_weights = lr_task.get_weights
        train = lr_task.train
    elif model_choice == "squeezenet":
        trainloader, valloader = sq_task.load_data(context)
        model = sq_task.create_model(context)
        set_weights = sq_task.set_weights
        get_weights = sq_task.get_weights
        train = sq_task.train
    elif model_choice == "bert":
        trainloader, valloader = b_task.load_data(context)
        model = b_task.create_model(context)
        set_weights = b_task.set_weights
        get_weights = b_task.get_weights
        train = b_task.train
    
    set_weights(model, weights)
    log(INFO, f"Node {id} training model.")
    # Maybe I must call ndarray_to_parameters(weights) before I can use them?
    # parameters = ndarrays_to_parameters(weights)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    training_start = time.time()
    # Fit model to the data.
    # train(model, trainloader, local_epochs, device)
    end_time = time.time()
    weights = get_weights(model) 
    size = len(trainloader.dataset)
    log(INFO, f"Node {id} done with training. Used {end_time-training_start} s.")

    # Type of security, i.e., none/orfedavg or fhefedavg
    sec_type = context.run_config["sec-type"]
    
    # Lag path hvis den ikke eksisterer.
    current_path = Path(f"./storage/training-time")
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
        outfile.write(str(end_time-training_start))

    # Compute size*weights and pack size into the updates structure.
    updates = [np.asarray([size])] + [size*w for w in weights]
    log(INFO, f"Node {id} has computed its result.")

    if sec_type == "fhefedavg":
        log(INFO, "Doing encryption.")
        """Anta at alle clients får ca. like store vekter,
        eventuelt legge på litt ekstra for sikkerhets skyld."""
        # Find maximum and minimum weights:
        # max_weights = []
        # min_weights = []
        # for j in range(len(weights)):
        #     max_weights.append(np.max(weights[j]))
        #     min_weights.append(np.min(weights[j]))
        # true_max = np.max(max_weights)
        # true_min = np.min(min_weights)
        # with open("result_files/maximum_weight.txt", "a") as outfile:
        #     outfile.write(f"Dataset set size is size {size}\n")
        #     outfile.write(f"Testing set size is {len(valloader.dataset)}\n")
        #     outfile.write(f"The largest weight in the sum is {true_max}\n")
        #     outfile.write(f"The smallest weight in the sum is {true_min}\n\n")


        # Find the structure of the weights, so that decryption can be done.
        struct = find_struct(updates)

        # Store struct in context.state
        arr = np.asarray(struct, dtype=int)
        struct_record = ArrayRecord(numpy_ndarrays=[arr])
        context.state.array_records['struct'] = struct_record

        # Store updates as well.
        update_record = ArrayRecord(numpy_ndarrays=updates)
        context.state.array_records['weights'] = update_record

        """Probably cleaner to just return
        struct to server here and
        then use weights_blocks()
        to partition the weights."""
        recordset = RecordDict()
        recordset['struct'] = struct_record

        reply = Message(content=recordset,
                        reply_to=msg)

    else:
        # sec_type == orgfedavg.
        # Only need to return updates itself.
        update_record = ArrayRecord(numpy_ndarrays=updates)
        recordset = RecordDict()
        recordset['updates'] = update_record
        reply = Message(content=recordset,
                                reply_to=msg)

    log(INFO, f"Node {id} sending reply to server.")

    # Aggresiv garbage collection.
    gc.collect()

    return reply


@app.evaluate()
def evaluate(msg: Message, context: Context):

    """Handles the evaluation of the model.
    Receives weights from server which have already been 
    decrypted, due to how the weighted average is computed.
    Each client returns evaluation results in plaintext,
    since evaluation is only for benchmarking anyway."""

    # Aggresiv garbage collection.
    gc.collect()

    # Get contents of message.
    id = msg.content.config_records['config']['id']
    weights = msg.content.array_records['weights']
    """weights is here a Flower ArrayRecord, 
    a dict with keys '0', '1', ..., and values 
    of Flower Array. 
    Must convert weights into a list of ndarrays again."""
    # Weights is now list of ndarrays, as it was on the server.
    weights = [weights[key].numpy() for key in list(weights.keys())]
    
    # Get this client's dataset partition.
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    # Load model depending on chosen task.
    model_choice = context.run_config["model"]
    if model_choice == "logreg":
        trainloader, valloader = lr_task.load_data(context)
        model = lr_task.create_model(context)
        set_weights = lr_task.set_weights
        test = lr_task.test
    elif model_choice == "squeezenet":
        trainloader, valloader = sq_task.load_data(context)
        model = sq_task.create_model(context)
        set_weights = sq_task.set_weights
        test = sq_task.test
    elif model_choice == "bert":
        trainloader, valloader = b_task.load_data(context)
        model = b_task.create_model(context)
        set_weights = b_task.set_weights
        test = b_task.test

    # Update model weights.
    set_weights(model, weights)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Evaluate model on the data
    print("Beginning evaluation!")
    start_time = time.time()
    if model_choice == "bert":
        loss, accuracy, f1 = test(model, valloader, device)
    else:
        loss, accuracy = test(model, valloader, device)
    end_time = time.time()
    print("Finished evaluation!")

    # Lag path hvis den ikke eksisterer.
    current_path = Path("./storage/evaluation-time")
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

    # Create and return reply.
    if model_choice == "bert":
        config = {'loss': float(loss), 
                  'size': len(valloader.dataset), 
                  'accuracy': float(accuracy),
                  'f1': float(f1)}
    else:
        config = {'loss': float(loss), 
                  'size': len(valloader.dataset), 
                  'accuracy': float(accuracy)}
    reply_content = RecordDict(records={'results': ConfigRecord(config)})

    reply = Message(content=reply_content,
                    reply_to=msg)
    log(INFO, f"Node {id} sending reply to server.")

    """Force all clients to empty state just in case."""
    context.state.array_records['weights'] = ArrayRecord(numpy_ndarrays=[np.asarray([])])
    context.state.array_records['struct'] = ArrayRecord(numpy_ndarrays=[np.asarray([])])
    # Force garbage collection to run as well.
    gc.collect()

    return reply

@app.query("get_pds")
def get_pds(msg: Message, context: Context):
    return get_pds_imp(msg, context)

@app.query("send_pds")
def send_pds(msg: Message, context: Context):
    return send_pds_imp(msg, context)

@app.query("dist_pds_no_server")
def dist_pds_no_server(msg: Message, context: Context):
    return dist_pds_no_server_imp(msg, context)

@app.query("weights_blocks")
def weights_blocks(msg: Message, context: Context):
    return weights_blocks_imp(msg, context)

@app.query("create_single_ciphertext")
def create_single_ciphertext(msg: Message, context: Context):
    return create_single_ciphertext_imp(msg, context)

@app.query("start_forward_keygen")
def start_forward_keygen(msg: Message, context: Context):
    ek_needed = context.run_config['ek-needed']
    # Aggresiv garbage collection.
    gc.collect()

    # If evaluation key is needed, call function which generates it.
    # If not then call function which does not generate it.
    if ek_needed:
        reply = start_forward_keygen_base(msg, context)
    else:
        reply = start_forward_keygen_noek_base(msg, context)

    # Aggresiv garbage collection.
    # gc.collect()
    return reply

@app.query("continue_forward_keygen")
def continue_forward_keygen(msg: Message, context: Context):
    ek_needed = context.run_config['ek-needed']
    # Aggresiv garbage collection.
    gc.collect()

    # If evaluation key is needed, call function which generates it.
    # If not then call function which does not generate it.
    if ek_needed:
        reply = continue_forward_keygen_base(msg, context)
    else:
        reply = continue_forward_keygen_noek_base(msg, context)

    # Aggresiv garbage collection.
    gc.collect()
    return reply

@app.query("end_forward_keygen")
def end_forward_keygen(msg: Message, context: Context):
    ek_needed = context.run_config['ek-needed']
    # Aggresiv garbage collection.
    gc.collect()

    # If evaluation key is needed, call function which generates it.
    # If not then call function which does not generate it.
    if ek_needed:
        reply = end_forward_keygen_base(msg, context)
    else:
        reply = end_forward_keygen_noek_base(msg, context)

    # Aggresiv garbage collection.
    gc.collect()
    return reply

@app.query("continue_backward_keygen")
def continue_backward_keygen(msg: Message, context: Context):
    ek_needed = context.run_config['ek-needed']
    # Aggresiv garbage collection.
    gc.collect()

    # If evaluation key is needed, call function which generates it.
    # If not then call function which does not generate it.
    if ek_needed:
        reply = continue_backward_keygen_base(msg, context)
    else:
        reply = continue_backward_keygen_noek_base(msg, context)

    # Aggresiv garbage collection.
    gc.collect()
    return reply