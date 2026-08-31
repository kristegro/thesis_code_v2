import os
import sys
import time
from logging import INFO
from random import sample
import gc # garbage collection
from pathlib import Path

from flwr.common import (RecordDict, 
                         Message, 
                         ConfigRecord,
                         ArrayRecord)
from flwr.common.logger import log
import openfhe as fhe
import numpy as np

# Add /app, which contains all help functions, to import path.
module_dir = os.path.abspath("/app")
if module_dir not in sys.path:
    sys.path.insert(0, module_dir)

from fhefedavg.fhe.openfhe_help_functions import (fhe_deserialize_file,
                                                  fhe_serialize_string, 
                                                  fhe_deserialize_string)
from fhefedavg.fhe.keygen import (server_keygen, 
                                  server_keygen_noek, 
                                  server_keygen_noise_flooding, 
                                  server_keygen_noise_flooding_noek)
from fhefedavg.fhe.noise_flooding_server import create_noise_estimate


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
            # target query('dec_without_server') method in ClientApp
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


def server_fhe(grid, context, node_ids, participating, replies, weights, round):
    """Handles all server-side fhe work.
    Starts by receiving update structure from all clients,
    does computation on each block, before reconstructing the weights structure.
    
    Args:
        grid (Grid): The Flower grid used in fhe operations.

        context (Context): The Flower context of the server.
            
        node_ids (list[int]): A list of ids of all nodes.

        participating (list[int]): A list of ids of all nodes which participate in training.

        replies (list[Message]): List of Flower messages, replies from each client.

        weights (list[np.ndarray]): Current weights.

        round (int): Which round it is.

    Returns:
        list[np.ndarray]: New aggregated weights.
    """

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
    ek_needed = context.run_config['ek-needed']
    # If evaluation key is needed, call function which generates it.
    # If not then call function which does not generate it.
    if SCHEME == "CKKS" or SCHEME == "CKKS-NF":
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
        # If evaluation key is needed, generate it.
        if ek_needed:
            start_time = time.time()
            server_keygen_noise_flooding(grid, context, "evaluation")
            end_time = time.time()
        else:
            start_time = time.time()
            server_keygen_noise_flooding_noek(grid, context, "evaluation")
            end_time = time.time()
    else:
        # No noise flooding.
        # If evaluation key is needed, generate it.
        if ek_needed:
            start_time = time.time()
            server_keygen(grid, context)
            end_time = time.time()
        else:
            start_time = time.time()
            server_keygen_noek(grid, context)
            end_time = time.time()
    log(INFO, "Key generation finished.")

    # Lag path hvis den ikke eksisterer.
    current_path = Path("./storage/keygen-time")
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
        num_part = len(participating)
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
        if SCHEME == "CKKS" or SCHEME == "CKKS-NF":
            elem = cc.MakeCKKSPackedPlaintext([0])
        else:
            elem = cc.MakeCoefPackedPlaintext([0])
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
        current_path = Path(storage+"decryption-time")
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
    inv_sizes = 1/(weights[0][0])
    weights = [weights[i]*inv_sizes for i in range(1, len(weights))]
    return weights