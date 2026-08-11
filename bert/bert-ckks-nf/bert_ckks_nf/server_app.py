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
import openfhe as fhe
from transformers import AutoModelForSequenceClassification
import numpy as np
from prometheus_client import Gauge, start_http_server, Counter

from bert_ckks_nf.task import get_weights, set_weights

# Add /app, which contains all help functions, to import path.
module_dir = os.path.abspath("/app")
if module_dir not in sys.path:
    sys.path.insert(0, module_dir)

from openfhe_help_functions import (fhe_deserialize_file, 
                                    fhe_deserialize_string, 
                                    fhe_serialize_string,
                                    fhe_serialize_file)
from pol_encoding import (decode_real)
from keygen import server_keygen_noise_flooding, server_keygen_noise_flooding_noek

def dec_completly_without_server(grid, context, node_ids, ct, ST):
    """Decrypt a single ciphertext ct, which will be used to 
    aquire a noise estimate. The server does not recieve either
    the partial decryptions or the result.
    Use ST as serialization type where relevant.

    Args:
        grid (Grid): The Flower grid that the nodes participating in decryption
            belongs to.

        context (Context): The Flower context of the server.
            
        node_ids (list[int]): A list of ids of nodes participating in decryption.
        
        ct (openfhe.Ciphertext): Ciphertext to decrypt.

        ST (openfhe.BINARY | openfhe.JSON): Serialization type to use when relevant.
            Either binary or JSON."""
    # Aggresiv garbage collection.
    gc.collect()

    # Serialize ciphertexts.
    ct_string = fhe_serialize_string(ct, ST)

    # Loop and wait until enough nodes are available.
    min_nodes = context.run_config['num-clients']
    while len(node_ids) < min_nodes:
        node_ids = list(grid.get_node_ids())
        log(INFO, "Waiting for nodes to connect...")
        time.sleep(1)

    # Send ciphertext to each node to get
    # partial decryption.
    messages = []
    for i in range(len(node_ids)):
        recordset = RecordDict()
        dec_record = ConfigRecord({'ct_string': ct_string,
                                   'id': i})
        recordset['ciphertext'] = dec_record
        message = Message(
            content=recordset,
            message_type="query.dist_pds_no_server",  
            dst_node_id=node_ids[i],
            group_id="decryption_no_server",
        )
        messages.append(message)
    
    # Aggresiv garbage collection.
    # del ct_strings
    # gc.collect()

    """The clients send replies after they have computed the noise estimate.
    So server is done after this."""
    replies = grid.send_and_receive(messages)

    # Aggresiv garbage collection.
    del ct
    del ct_string
    del messages
    del replies
    gc.collect()

def create_noise_estimate(grid, context, node_ids, ST):
    """Do all the steps required to compute a noise estimate for CKKS noise flooding mode.
    Do keygen, make clients create ciphertext, sum ciphertexts from all clients,
    make clients decrypt sum without sharing any pds. Clients save noise estimate
    for use in keygen to create the actual keys.
    
    The point of this function is to be timed so that I can see how much extra
    time the estimation takes."""

    # Make clients do keygen in noise estimation mode.
    ek_needed = context.run_config['ek-needed']
    # If evaluation key is needed, call function which generates it.
    # If not then call function which does not generate it.
    if ek_needed:
        server_keygen_noise_flooding(grid, context, "estimate")
    else:
        server_keygen_noise_flooding_noek(grid, context, "estimate")
    log(INFO, "First key generation finished.")

    # Loop and wait until enough nodes are available.
    min_nodes = context.run_config['num-clients']
    while len(node_ids) < min_nodes:
        node_ids = list(grid.get_node_ids())
        log(INFO, "Waiting for nodes to connect...")
        time.sleep(1)

    # Make clients create a single ciphertext each.
    messages = []
    for i in range(len(node_ids)):
        recordset = RecordDict()
        dec_record = ConfigRecord({'id': i})
        recordset['ciphertext'] = dec_record
        message = Message(
            content=recordset,
            message_type="query.create_single_ciphertext",  
            dst_node_id=node_ids[i],
            group_id="single-ct",
        )
        messages.append(message)
    replies = grid.send_and_receive(messages)

    # Each reply contains a ciphertext, sum these together.
    cts = []
    for reply in replies:
        ct_string = reply.content.config_records['ct']['ct']
        ct = fhe_deserialize_string(ct_string, "Ciphertext", fhe.BINARY)
        cts.append(ct)
    
    storage = os.path.abspath('storage/') + "/"
    cc = fhe_deserialize_file("CryptoContext", storage+"cc", ST)
    fhe.ClearEvalMultKeys() # Sometimes get errors this is not done.
    ct_sum = cts[0]
    for i in range(1, len(cts)):
        ct_sum = cc.EvalAdd(ct_sum, cts[i])

    # ct_sum is now sum of ciphertexts. Make clients decrypt and save noise estimate.
    dec_completly_without_server(grid, context, node_ids, ct_sum, ST)

    """At this point clients have saved the noise estimate to their states
    and normal keygen can be done with said estimate."""


def dec_without_server_multiple(grid, context, node_ids, ct_list, ST, pt_type):
    """Decrypt all ciphertexts in ct_list using all
    nodes in node_ids which belong to grid.
    For this version, one partial decryption is distributed directly amongst
    the clients without server involvement.
    The server thus cannot decrypt directly and must instead receive the results
    from a client. This is better from a security perspective and eaiser
    to change if I later get CKKS division to work.
    Use ST as serialization type where relevant.

    Args:
        grid (Grid): The Flower grid that the nodes participating in decryption
            belongs to.

        context (Context): The Flower context of the server.
            
        node_ids (list[int]): A list of ids of nodes participating in decryption.
        
        ct_list (list[openfhe.Ciphertext]): List of ciphertexts to decrypt.

        ST (openfhe.BINARY | openfhe.JSON): Serialization type to use when relevant.
            Either binary or JSON.

        pt_type (str): 'int' or 'real'. This indicates whether the ciphertexts
            encrypt integers or real numbers.
            This is not a security concern since the server and clients
            already know that dataset sizes are integers and 
            weights are real numbers.
            
    Returns:
        list[openfhe.Plaintext]: A list containing decrypted results of the ciphertexts."""
    
    # Aggresiv garbage collection.
    gc.collect()

    # Serialize ciphertexts.
    ct_strings = [fhe_serialize_string(ct, ST) for ct in ct_list]

    # Aggresiv garbage collection.
    del ct_list
    gc.collect()

    """Den enkleste løsningen er vel egentlig hvis 
    alle partial decryptions utvekles
    mellom klientene direkte,
    istedenfor at hovedserver skal distribuere
    alle unntatt 1.
    
    Egentlig så burde vel clients rekonstruere sine 
    egne weights strukturer med dekrypterte resultater
    og kun sende slutt resultatet til server?
    Da må jeg isåfall gjøre store endringer til 
    dec_with_server_multiple funksjonen."""

    # Loop and wait until enough nodes are available.
    min_nodes = context.run_config['num-clients']
    while len(node_ids) < min_nodes:
        node_ids = list(grid.get_node_ids())
        log(INFO, "Waiting for nodes to connect...")
        time.sleep(1)

    # Send ciphertext to each node to get
    # partial decryption.
    messages = []
    for i in range(len(node_ids)):
        recordset = RecordDict()
        dec_record = ConfigRecord({'ct_list': ct_strings,
                                   'id': i,
                                   'pt_type': pt_type})
        recordset['ciphertext'] = dec_record
        message = Message(
            content=recordset,
            message_type="query.get_pds",  
            dst_node_id=node_ids[i],
            group_id="decryption",
        )
        messages.append(message)
    
    # Aggresiv garbage collection.
    del ct_strings
    gc.collect()

    """Assume that the returned messages contain the partial decryptions from 
    each client, except the one which launched the subprocess server."""
    replies = grid.send_and_receive(messages)

    # Aggresiv garbage collection.
    del messages
    gc.collect()

    struct = []
    for reply in replies:
        # No keys means empty reply from subserver hosting client.
        if len(reply.content.keys()) == 0:
            continue
        struct.append(reply.content.config_records['pds']['pds'])
    
    # Aggresiv garbage collection.
    del replies
    gc.collect()

    # Server now has partial decryptions from all clients
    # except the one which hosted the subserver.
    # Thus the server cannot complete decryption itself, as wanted.

    # Loop and wait until enough nodes are available.
    min_nodes = context.run_config['num-clients']
    while len(node_ids) < min_nodes:
        node_ids = list(grid.get_node_ids())
        log(INFO, "Waiting for nodes to connect...")
        time.sleep(1)

    # Send struct to all clients.
    """Må dele opp structs før sending slik jeg gjorde i help_code/flwr-decryption"""
    messages = []
    for i in range(len(node_ids)):
        recordset = RecordDict()
        config_record = ConfigRecord({'id': i})
        pds_record = ConfigRecord()
        for j in range(len(struct)):
            pds_record[f'pds{j}'] = struct[j]
        recordset['config'] = config_record
        recordset['pds'] = pds_record
        message = Message(
            content=recordset,
            message_type="query.send_pds",  
            # target query('send_pds') method in ClientApp
            dst_node_id=node_ids[i],
            group_id="decryption",
        )
        messages.append(message)

    replies = grid.send_and_receive(messages)

    # Aggresiv garbage collection.
    del messages
    gc.collect()

    # All clients except client 0 will send empty replies.
    # Hopefully, the first reply corresponds to the first client.
    for reply in replies:
        # Most replies are empty
        if len(reply.content.keys()) == 0:
            continue
        weights = reply.content.array_records['weights'].to_numpy_ndarrays()
        break
    # Weights is now list of a single array, but only need the array itself

    # Aggresiv garbage collection.
    del replies
    gc.collect()

    return weights[0]

# Create ServerApp
app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:

    # Aggresiv garbage collection.
    gc.collect()

    # Starting time.
    start_time = time.time()

    # Scheme to be used.
    SCHEME = context.run_config['scheme']
    log(INFO, f"Running FedAVG with FHE using {SCHEME}.")

    # Serialization type
    st = context.run_config['ser-type']
    if st == "BINARY":
        ST = fhe.BINARY
    elif st == "JSON":
        ST = fhe.JSON

    # Chunk size to use when encryption weights.
    CS = context.run_config['chunk-size']
    """Know that atleast 100 and 500 works."""

    """Maybe all these constants should be put into the pyproject.toml
    file and read from there instead?
    Then the clients can read them directly from there 
    as well I think, instead of needing the server the share them."""

    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    fraction_evaluation = 1.0

    # Initialize global model
    
    log(INFO, "Initializing global model.")
    model_name = context.run_config["model-name"]
    num_labels = context.run_config["num-labels"]
    net = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    weights = get_weights(net)
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

    # Define a gauge to track the global model F1 score
    f1_gauge = Gauge("model_f1", 
                     "Current F1 score of the global model",
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
        f1_gauge.labels(name=f"round-{i}")

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

        # Receive struct from the clients to know the structure
        # of the weights.
        # All replies should contain the same value
        # since all nodes should have weights with the same structure.
        reply = replies[0]
        struct_arr = reply.content.array_records['struct']
        struct = struct_arr.to_numpy_ndarrays()[0]

        # Aggresiv garbage collection.
        del replies
        gc.collect()
        
        # Add array to first position in weights to get same structure
        # as client updates.
        weights = [np.asarray([0])] + weights

        """Do key generation."""
        start_time_noise = time.time()
        create_noise_estimate(grid, context, node_ids, ST)
        end_time_noise = time.time()
        current_path = Path("./storage/noise-estimate-time")
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
            outfile.write(str(end_time_noise-start_time_noise))

        ek_needed = context.run_config['ek-needed']
        # If evaluation key is needed, call function which generates it.
        # If not then call function which does not generate it.
        if ek_needed:
            start_time = time.time()
            server_keygen_noise_flooding(grid, context, "evaluation")
            end_time = time.time()
        else:
            start_time = time.time()
            server_keygen_noise_flooding_noek(grid, context, "evaluation")
            end_time = time.time()
        log(INFO, "Key generation finished.")

        # Lag path hvis den ikke eksisterer.
        storage = os.path.abspath('storage/') + "/"
        current_path = Path(storage+"bert-keygen-time")
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

        # Prepare crypto context.
        storage = os.path.abspath('storage/') + "/"
        cc = fhe_deserialize_file("CryptoContext", storage+"cc", ST)
        fhe.ClearEvalMultKeys() # Sometimes get errors this is not done.
        if ek_needed:
            jek = fhe_deserialize_file("EvalKey", storage+"jek", ST)
            cc.InsertEvalMultKey([jek])

        """Use while loop to get
        chunks of weights for aggregation."""
        pos = 0
        log(INFO, f"Round {round}; Starting transfer of ciphertext blocks.")
        # TA TIDA HERFRA for benchmark
        # new_weights = weights
        start_time = time.time()
        while pos < len(struct):
            block_start = time.time()
            """Receive chunks of size
            CS and do aggregation on them.
            Decrypt and store in new weights
            structure."""

            """Need to ensure
            that pos+CS < len(struct),
            and if not must use 
            a smaller CS
            for the last chunk."""
            cs = CS
            
            """If CS == -1, send all weights at once.
            Thus, don't need to block up if """
            if CS == -1:
                cs = len(struct)

            if pos + CS >= len(struct):
                cs = len(struct) - pos
            log(INFO, f"Round {round}; Aggregating chunk starting at pos {pos} with size {cs}")

            log(INFO, f"Round {round}; Sending messages to get chunk from pos {pos}.")
            messages = []
            for j in range(num_part):
                recordset = RecordDict()
                # cs can change, so still need to send it as a message.
                round_record = ConfigRecord({'id': j,
                                             'cs': cs,
                                             'pos': pos})
                """weights is a list of ndarrays, and so it cannot 
                be sent in ConfigRecord. Must instead send as ArrayRecord."""
                array_record = ArrayRecord(numpy_ndarrays=weights)
                recordset['config'] = round_record
                recordset['weights'] = array_record
                # log(INFO, f"recordset.keys() = {recordset.keys()}")
                message = Message(
                    content=recordset,
                    message_type="query.weights_blocks",  
                    # target query("weights_blocks") method in ClientApp
                    dst_node_id=participating[j],
                    group_id=str(round),
                )
                messages.append(message)
            replies = grid.send_and_receive(messages)
            # Aggresiv garbage collection.
            del messages
            gc.collect()
            log(INFO, f"Round {round}; Received chunk from each node.")

            """Store the ciphertexts received from each client,
            after deserializing them."""
            log(INFO, f"Round {round}; Deserialize chunks.")
            weights_storage = []
            for reply in replies:
                # Weights are stored in a config record.
                weights_lst = reply.content.config_records['weights']['weights']
                # weights is a ndarray of serialized ciphertexts. Deserialize.
                # log(INFO, f"weights_lst[0] = {weights_lst[0]}, type = {type(weights_lst[0])}")
                weights_lst = [fhe_deserialize_string(weight, 
                                                      "Ciphertext", 
                                                      fhe.BINARY)
                                for weight in weights_lst]
                weights_storage.append(weights_lst)
            
            # Also store recieved indices.
            ciphertext_indices_arr = replies[0].content.array_records['indices']
            ciphertext_indices = ciphertext_indices_arr.to_numpy_ndarrays()[0]

            # Aggresiv garbage collection.
            del replies
            gc.collect()

            # Prepare for summation.
            elem = cc.MakeCKKSPackedPlaintext([0])
            weights_sum = [elem for _ in range(len(weights_storage[0]))]
            # weights_sum contains a 0-plaintext for each weigth to be summed.

            # Do summation.
            log(INFO, f"Round {round}; Do summation of received weigths.")
            for j in range(len(weights_sum)):
                for weight in weights_storage:
                    """Add the j-th weight from all nodes
                    to the sum."""
                    weights_sum[j] = cc.EvalAdd(weight[j], weights_sum[j])

            # Clear contents of weights_storage to free memory,
            # since the contents are not needed anymore.
            # Does it work like I want it to?
            # weights_storage.clear() does not seem to help.
            # Try to just set to None instead and call garbage collection.
            del weights_storage
            gc.collect()


            # Must now decrypt each of the sums.
            log(INFO, f"Round {round}; Requesting decryption for each of the {len(weights_sum)} sums.")
            dec_start = time.time()
            # weights_sum = [dec_with_server(grid, node_ids, sums, ST)
            #                for sums in weights_sum]
            # weights_sum = dec_with_server_multiple(grid, context, node_ids, weights_sum, cc, ST)
            weights_sum = dec_without_server_multiple(grid, context, node_ids, weights_sum, ST, "real")
            dec_stop = time.time()
            current_path = Path(storage+"bert-decryption-time")
            Path.mkdir(current_path,
                    mode=0o777, 
                    parents=True, 
                    exist_ok=True)
            Path.chmod(current_path, mode=0o777)
            """Mappe for run skal ha navn 'run{num_dirs+1}',
            altså første run for navn 'run1' osv."""
            run_dir = f"run{round}/"
            Path.mkdir(current_path/run_dir, mode=0o777, exist_ok=True)
            Path.chmod(current_path/run_dir, mode=0o777)
            Path.touch(current_path/run_dir/"time.txt", 0o777)
            Path.chmod(current_path/run_dir/"time.txt", mode=0o777)
            with open(Path(current_path/run_dir/"time.txt"), "a") as outfile:
                outfile.write(str(dec_stop-dec_start)+"\n")
            log(INFO, f"Round {round}; All sum decryptions received.")
            log(INFO, f"Round {round}; Decryption took {dec_stop-dec_start} s.")


            """At this point, weights_sum contains
            the summed results for this block of weights.
            Must then insert these values into 
            the weight structure."""
            log(INFO, f"Round {round}; Inserting computed averages into weights structure.")
            for w, indices in zip(weights_sum, ciphertext_indices):
                """struct uses the format (i, j, k)
                which indicates that ciphertext at struct[x]
                comes from either weights[i][j][k] or
                weights[i][j] if k == -1 or
                weights[i] if j == k == -1.
                """

                """indices is a list of which indices are used in a 
                specific ciphertext.
                w is a list of weights from that same ciphertext."""
                for i in range(len(w)):
                    """The number of elements in w is equal to the number
                    of elements in indices which are not -1, i.e., 
                    without padding they have equal length."""
                    # Get the indices in struct[indices[i]] which are not -1, i.e., padding.
                    idx = [j for j in struct[indices[i]] if j != -1]
                    """The list index must be handled separetly, but arrays
                    can be indexed by a tuple. If len(idx) is 0 then
                    the tuple is simply empty and the whole array is returned.
                    Thus, there is no need for an if-test to check if it is empty,
                    as I had previously."""
                    weights[idx[0]][tuple(idx[1::])] = w[i]

            # Clear averages and ciphertext_indices as well since they are not needed.
            del weights_sum
            del ciphertext_indices
            gc.collect()

            """Now the weights structure
            has been updated with 
            the weighted average of the block
            that the server received.
            Repeat for all blocks."""
            log(INFO, f"Round {round}; Insertion of chunk from pos {pos} with size {cs} done.")
            log(INFO, f"Round {round}; Insertion took {time.time() - block_start} s.")
            pos += cs
            
            """The following code comment can be used to verify the new
            method of reconstructing the weights gives the same results as the original method.
            To use it, lines 203 and 464-469 must be commented in as well."""
            # atol = 1e-10
            # rtol = 0
            # assert len(old_weights) == len(weights)
            # for k in range(len(old_weights)):
            #     try:
            #         assert len(old_weights[k]) == len(weights[k])
            #     except AssertionError:
            #         log(INFO, f"len(old_weights[{k}]) = {len(old_weights[k])}; len(weights[{k}]) = {len(weights[k])}")
            #     for j in range(len(old_weights[k])):
            #         if isinstance(old_weights[k][j], np.ndarray):
            #             try:
            #                 assert len(old_weights[k][j]) == len(weights[k][j])
            #                 assert np.allclose(weights[k][j], old_weights[k][j], atol=atol, rtol=rtol)
            #             except AssertionError:
            #                log(INFO, f"len(old_weights[{k}][{j}]) = {len(old_weights[k][j])};" 
            #                    f" len(weights[{k}][{j}]) = {len(weights[k][j])}") 
            #                log(INFO, f"np.allclose(weights[{k}][{j}], old_weights[{k}][{j}]) = " 
            #                    f"{np.allclose(weights[k][j], old_weights[k][j], atol=atol, rtol=rtol)}")
            #         else:
            #             try:
            #                 assert np.allclose(weights[k], old_weights[k], atol=atol, rtol=rtol)
            #             except AssertionError:
            #                 log(INFO, f"np.allclose(weights[{k}], old_weights[{k}]) = "
            #                     f"{np.allclose(weights[k], old_weights[k], atol=atol, rtol=rtol)}")
        

        """At this point, weights contains the summed results
        of ciphertext addition.
        The first element, weights[0] is an array containing
        the summed training set sizes of the clients. 
        This must be extracted and used to compute weights/size 
        for the rest of the weights structure."""
        print(f"weights[0][0] = {weights[0][0]}")
        inv_sizes = 1/(weights[0][0])
        weights = [weights[i]*inv_sizes for i in range(1, len(weights))]

        """At this point weights contains the proper
        aggregated global model update, and aggregation is finished."""
        # STOPP TIDTAGNING av benchmark her!
        end_time = time.time()
        # Lag path hvis den ikke eksisterer.
        current_path = Path(storage+"bert-block-time")
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
        f1_sum = 0
        for reply in replies:
            # Log results from clients.
            loss = reply.content.config_records['results']['loss']
            size = reply.content.config_records['results']['size']
            accuracy = reply.content.config_records['results']['accuracy']
            f1 = reply.content.config_records['results']['f1']

            loss_sum += loss*size
            size_sum += size
            # accuracy is just a number, not a dict.
            accuracy_sum += accuracy*size
            f1_sum += f1*size
        
        loss_average = loss_sum/size_sum
        accuracy_average = accuracy_sum/size_sum
        f1_average = f1_sum/size_sum

        # Aggresiv garbage collection.
        del replies
        gc.collect()

        # Update the Prometheus gauges with the latest aggregated values
        loss_gauge.labels(f"round-{round}").set(loss_average)
        accuracy_gauge.labels(f"round-{round}").set(accuracy_average)
        f1_gauge.labels(f"round-{round}").set(f1_average)

        log(INFO, f"Round {round}; Weighted average of loss = {loss_average}")
        log(INFO, f"Round {round}; Weighted average of accuracy = {accuracy_average}")
        log(INFO, f"Round {round}; Weighted average of f1 = {f1_average}")
        # with open("result_files/benchmark-openfhe-fedavg-docker.txt", "a") as outfile:
        #     outfile.write(f"Round {round}; Weighted average of loss = {loss_average}\n")
        #     outfile.write(f"Round {round}; Weighted average of accuracy = {accuracy_average}\n")
        #     outfile.write(f"Round {round}; Weighted average of f1 = {f1_average}\n")
        #     outfile.write(f"Round {round}; Runtime for weights aggregation = {benchmark_time_end-benchmark_time_start}\n\n")
        log(INFO, f"Round {round}; Evaluation completed. Round done.")

        # Save to be able to create plots later.
        set_weights(net, weights)
        storage = os.path.abspath('storage/') + "/"
        net.save_pretrained(storage+f"bert-tmp-r{round}")
        path = Path(storage+f"bert-tmp-r{round}")
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