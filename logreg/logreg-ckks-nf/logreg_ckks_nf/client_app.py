import os
import sys
from logging import INFO
import time
import gc # garbage collection
import subprocess
import signal
import glob
import multiprocessing.connection as mpc
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
import openfhe as fhe

from logreg_ckks_nf.task import (load_data, 
                                   set_weights, 
                                   get_weights, 
                                   train, 
                                   test, 
                                   LogisticRegression)

# Add /app, which contains all help functions, to import path.
module_dir = os.path.abspath("/app")
if module_dir not in sys.path:
    sys.path.insert(0, module_dir)

from openfhe_help_functions import (fhe_deserialize_file, 
                                              fhe_serialize_file, 
                                              fhe_deserialize_string, 
                                              fhe_serialize_string)
from pol_encoding import (encode_real, decode_real, encode_int, decode_int)
from keygen import (start_forward_keygen_base, 
                    continue_forward_keygen_base, 
                    end_forward_keygen_base,
                    continue_backward_keygen_base,
                    start_forward_keygen_noek_base, 
                    continue_forward_keygen_noek_base, 
                    end_forward_keygen_noek_base,
                    continue_backward_keygen_noek_base)


def weights_to_ciphertext_bytes(weights: list[list[float]], cc, jpk, scheme):
    """Convert a list of of lists of numbers to a list of byte serialized 
    openfhe ciphertexts.
    
    Args:
        weights (list[list[float]]): List of lists to encrypt.
        cc (openfhe.CryptoContext): The crypto context for encryption.
        jpk (openfhe.PublicKey): The public key to use for encryption.
        scheme (str): Either 'CKKS' or 'BFV', the encryption scheme to use.
        
    Returns:
        list[str]: A list of ciphertext byte serializations. 
            The serialization on index i is the encryption of weights[i]."""
    
    """Each element of weights is a ndarray,
    so convert each to list with .tolist()
    before encoding to plaintext and encrypting.
    
    Also store the lenght of each weight to aid in
    decryption."""

    """The weights argument is a subset of the actual weights,
    which need to be divided into chunks."""
    """Since most plaintexts need length 128, it is 
    better to only store the lenghts of those which differ.
    lengths should thus store a tuple (l, i, j)
    where l is the length and weights[i][j]
    is the position of the ciphertext in the structure.
    If j == -1, the ciphertext
    belongs at weights[i] instead."""
    """Probably also better to just have
    ciphertexts as a plain list instead of
    nested, and then use the 'indicies' list
    (see plan_ciphertext_partitioning.txt)
    to reconstruct the weights structure instead."""
    # Aggresiv garbage collection.
    gc.collect()
    ciphertexts = []

    for i in range(len(weights)):
        # Encode to plaintext.
        weight_pt = cc.MakeCKKSPackedPlaintext(weights[i])
        # Encrypt plaintext.
        weight_ct = cc.Encrypt(jpk, weight_pt)
        # Serialize for use with Flower.
        weight_string = fhe_serialize_string(weight_ct, fhe.BINARY)
        ciphertexts.append(weight_string)

        # Garbage collection for å forhåpentligvis redusere minnebruk.
        del weight_pt
        del weight_ct
        del weight_string
        gc.collect()

    return ciphertexts

def recursive_find_struct(weights, dims):
    """Determine the structure of weights
    by recursively iterating through
    elements until an array of numbers is found.
    
    Args:
        weights (list[np.ndarray] | np.ndarray):
            The weights to find the structure of.
        dims (list[int]):
            A list of dimensions for weights.
            If it is non-empty, the
            values indiciate the path taken from
            the start of the original weights,
            i.e., the current position is 
            weights[dims[0]][dims[1]][...][dims[-1]].
            
    Returns:
        list[tuple[int]]: A list of tuples,
            each of the form (i, j, ..., k)
            where i, j, ..., k show the element's position in weights,
            i.e., element = weights[i][j][...][k]."""
    # print(dims)
    struct = []
    """Struct should have elements of the form
    (i, j, ..., k)
    where i, j, ..., k show the element's position in weights,
    i.e., element = weights[i][j][...][k]."""

    # Basis state of recursion,
    # when weights is a number.
    if not isinstance(weights, (np.ndarray, list)):        
        # dims now has form (i, j, ..., k),
        # i.e., the position of a weight.
        return tuple(dims)
    
    # weights is an array of arrays, so can go deeper.
    for i in range(len(weights)):
        elem = weights[i]
        dims.append(i)
        values = recursive_find_struct(elem, dims)
        dims.pop(-1)
        if isinstance(values, tuple):
            # values is coords for a weight
            # so it can be appended directly.
            struct.append(values)
        else:
            # values is struct for multiple weights
            # so add together lists.
            struct += values

    return struct

def find_struct(weights):
    """Find structure of weights using recursive_find_struct().
    Pad elements of output with -1 to give all elements same lenght.
    This ensures that the final return value can be 
    converted to a numpy array for storing in the client state.
    
    Args:
        weights (list[np.ndarray] | np.ndarray):
            The weights to find the structure of.
            
    Returns:
        list[tuple[int]]: A list of tuples,
            each of the form (i, j, ..., k)
            where i, j, ..., k show the element's position in weights,
            i.e., element = weights[i][j][...][k].
            Unused indices are padded with -1's to ensure that 
            all elements have same lenght to allow conversion to ndarray."""
    # Find the structure with recursive_find_struct()
    struct = recursive_find_struct(weights, [])

    # Pad all elements with -1 until they all have the same length.
    max_length = max([len(struct[i]) for i in range(len(struct))])
    for i in range(len(struct)):
        s = list(struct[i])
        for j in range(max_length-len(s)):
            # Append enough -1 to fill s.
            # If len(s) == max_length, then nothing is appended.
            s.append(-1)
        struct[i] = tuple(s)
    
    return struct



# Flower ClientApp
app = ClientApp()

@app.query("get_pds")
def get_pds(msg: Message, context: Context):
    """This method handles launching of subserver / subclients
    and the distribution of the partial decryptions from client
    with partition_id 0.
    Further, all other clients send their partial decryptions to
    the server."""

    # Aggresiv garbage collection.
    gc.collect()

    # Serialization type
    st = context.run_config['ser-type']
    if st == "BINARY":
        ST = fhe.BINARY
    elif st == "JSON":
        ST = fhe.JSON

    # Get contents of message.
    ct_strs = msg.content.config_records['ciphertext']['ct_list']
    id = msg.content.config_records['ciphertext']['id']

    # log(INFO, f"Node {id} computing partial decryption.")
    # Assume that crypto context and secret key 
    # was serialized with same type as ST.
    storage = os.path.abspath('storage/') + "/"
    cc = fhe_deserialize_file("CryptoContext", storage+"cc", ST)
    sk = fhe_deserialize_file("PrivateKey", storage+"sk", ST)

    """Compute partial decryption.
    One client needs to start decryption,
    so might as well be the first one."""
    ct_list = [fhe_deserialize_string(ct_str, "Ciphertext", ST)
               for ct_str in ct_strs]
    len_ct_list = len(ct_list)
    
    # Aggresiv garbage collection.
    del ct_strs
    gc.collect()

    if id == 0:
        pd_list = cc.MultipartyDecryptLead(ct_list, sk)
    else:
        pd_list = cc.MultipartyDecryptMain(ct_list, sk)

    for i in range(len(pd_list)):
        pd_list[i] = fhe_serialize_string(pd_list[i], ST)

    # Aggresiv garbage collection.
    del ct_list
    gc.collect()

    # Write parital decryptions to files:
    # for i in range(len(pd_list)):
    #     fhe_serialize_file(pd_list[i], storage+f"pd{i}", ST)
    
    # For ease of programming, hardcode that the 
    # client with partition 0 launches the server.
    # This way I can open a specific port on the container.
    partition_id = context.node_config["partition-id"]
    addr = context.node_config["addr"]
    # Num-partions is equivalent to total number of clients,
    # but one client launches server, so only need num-partitions-1 clients.
    num_clients = context.node_config['num-partitions'] - 1

    if partition_id == 0:
        """Create Server which reads and then deletes
        partial decryption files."""
        # Launch superlink.
        args_1 = f'flower-superlink --insecure '
        args_2 = f'--serverappio-api-address {addr}:10091 ' 
        args_3 = f'--fleet-api-address {addr}:10092 '
        args_4 = f'--exec-api-address {addr}:10093'
        args = args_1 + args_2 + args_3 + args_4
        log(INFO, f"Command to run to launch superlink: {args}")
        super = subprocess.Popen(args=args,
                                     shell=True,
                                     start_new_session=True,
                                     )
        time.sleep(2) # To avoid flwr run being done to soon.
        """Kanskje bedre å sjekke om porten 10093
        er åpen, slik som jeg gjorde
        i essaykoden med funksjonen is_port_in_use."""
        args1 = f'flwr run flwr-decryption/ '
        args2 = f"-c 'num-ct={len_ct_list} num-clients={num_clients}' "
        args3 = f"--federation-config 'options.num-supernodes={num_clients}'"
        args = args1 + args2 + args3
        log(INFO, f"Command for flwr run: {args}")
        flwrrun = subprocess.Popen(args=args,
                                   shell=True,
                                   start_new_session=True,
                                   )
        
        # Start multiprocess.connection Listener.
        # Cannot pass any information to superlink,
        # so must give unique port.
        listener = mpc.Listener(("127.0.0.1", 12360))
        # accept waits until a mpc.Client connects.
        connection = listener.accept()
        # When client is connected, can send partial decryptions.
        connection.send(pd_list)
        # Close connection after sending.
        connection.close()
        listener.close()

    else:      
        """Create Client which reads and then deletes
        partial decryption files."""

        # Launch supernode.        
        args_1 = f'flower-supernode --insecure '
        args_2 = f'--superlink {addr}:10092 '
        args_3 = f"--node-config 'id={id}' "
        args_4 = f'--clientappio-api-address 0.0.0.0:{10000+id}'
        print(f"Command to launch supernode: \n{args_1+args_2+args_3+args_4}")
        args = args_1 + args_2 + args_3 + args_4
        super = subprocess.Popen(args=args,
                                     shell=True,
                                     start_new_session=True,
                                     )

    if partition_id == 0:
        listener = mpc.Listener(("127.0.0.1", 12360))
        # Wait until subserver open connections.
        # Then the server can be closed.
        connection = listener.accept()
        connection.close()
        listener.close()
        os.killpg(os.getpgid(super.pid), signal.SIGTERM)

        # save own partial decryptions in the same way that others do.
        context.state.config_records['pds_subserver'] = ConfigRecord({'pds_subserver': pd_list})

        # Aggresiv garbage collection.
        del pd_list

        # Give empty reply to server.
        # This is simply to avoid Flower crashing before the code is even run.
        recordset = RecordDict()
        reply = Message(content=recordset,
                        reply_to=msg)
        
        # Aggresiv garbage collection.
        gc.collect()

        return reply
    else:
        listener = mpc.Listener(("127.0.0.1", 12340+id))
        # This one waits until subprocess is ready to send partial decryptions back.
        # accept waits until a mpc.Client connects.
        connection = listener.accept()
        # When client is connected, can receive partial decryptions.
        pds_subserver = connection.recv()
        # Close connection after sending.
        connection.close()
        listener.close()
        os.killpg(os.getpgid(super.pid), signal.SIGTERM)

        # save pds_subserver in state.
        context.state.config_records['pds_subserver'] = ConfigRecord({'pds_subserver': pds_subserver})

        del pds_subserver

        # Return own partial decryptions to server.
        recordset = RecordDict()
        config_record = ConfigRecord({'pds': pd_list})
        recordset['pds'] = config_record
        reply = Message(content=recordset,
                        reply_to=msg)
        
        # Aggresiv garbage collection.
        gc.collect()
        return reply

@app.query("send_pds")
def send_pds(msg: Message, context: Context):
    """This method gives the clients access to all partial decryptions
    so that they can decrypt properly.
    In this setup one of them, say client with id 0,
    should also send results back to server."""

    # Aggresiv garbage collection.
    gc.collect()

    # Extract message contents.
    id = msg.content.config_records['config']['id']
    pds = []
    num_pds = context.node_config['num-partitions'] - 1
    for i in range(num_pds):
        pds.append(msg.content.config_records['pds'][f'pds{i}'])
    
    # Serialization type
    st = context.run_config['ser-type']
    if st == "BINARY":
        ST = fhe.BINARY
    elif st == "JSON":
        ST = fhe.JSON

    # Batch size, number of elements in each ciphertext.
    # Must correspond to the batchsize of the keys,
    # and the batchsize used in clientapp.
    BATCHSIZE = context.run_config['ring-dim'] // 2
    storage = os.path.abspath('storage/') + "/"
    cc = fhe_deserialize_file("CryptoContext", storage+"cc", ST)

    # Get pds_subserver from state.
    pds.append(context.state.config_records['pds_subserver']['pds_subserver'])

    # Now have all pds, can finish decryption.
    plaintexts = [[] for _ in range(len(pds[0]))]
    for i in range(len(pds)):
        for j in range(len(pds[i])):
            plaintexts[j].append(fhe_deserialize_string(pds[i][j], "Ciphertext", ST))
        # Aggresiv garbage collection.
        pds[i] = []
        gc.collect()
    
    # Aggresiv garbage collection.
    del pds
    gc.collect()

    for i in range(len(plaintexts)):
        tmp = plaintexts[i]
        # Fuse together to get plaintext.
        tmp = cc.MultipartyDecryptFusion(tmp)
        # Set correct size.
        tmp.SetLength(BATCHSIZE)
        plaintexts[i] = tmp
        # Aggresiv garbage collection.
        del tmp
        gc.collect()

    # For CKKS, each slot in the plaintext contains a weight,
    # so the weights can inserted as is.
    weights_sum = np.asarray([sums.GetRealPackedValue()
                    for sums in plaintexts])
        
    # Aggresiv garbage collection.
    del plaintexts
    gc.collect()

    # Remove any partial decryption files which might remain.
    for f in glob.glob(storage+"pd*"):
        os.remove(f)

    if id == 0:
        """Send decrypted weights back to server."""
        recordset = RecordDict()
        array_record = ArrayRecord(numpy_ndarrays=[weights_sum])
        recordset['weights'] = array_record
        reply = Message(content=recordset,
                        reply_to=msg)
        log(INFO, f"Node {id} sending reply to server.")

        # Aggresiv garbage collection.
        gc.collect()

        return reply
    else:
        """Send empty reply."""
        recordset = RecordDict()
        reply = Message(content=recordset,
                        reply_to=msg)
        log(INFO, f"Node {id} sending reply to server.")

        # Aggresiv garbage collection.
        gc.collect()

        return reply
    
@app.query("dist_pds_no_server")
def dist_pds_no_server(msg: Message, context: Context):
    """Receive a single ciphertext from the server,
    decrypt it, compute and save noise estimate to state."""
    # Aggresiv garbage collection.
    gc.collect()

    # Serialization type
    st = context.run_config['ser-type']
    if st == "BINARY":
        ST = fhe.BINARY
    elif st == "JSON":
        ST = fhe.JSON

    # Get contents of message.
    ct_string = msg.content.config_records['ciphertext']['ct_string']
    id = msg.content.config_records['ciphertext']['id']

    # log(INFO, f"Node {id} computing partial decryption.")
    # Assume that crypto context and secret key 
    # was serialized with same type as ST.
    storage = os.path.abspath('storage/') + "/"
    cc = fhe_deserialize_file("CryptoContext", storage+"cc", ST)
    sk = fhe_deserialize_file("PrivateKey", storage+"sk", ST)

    """Compute partial decryption.
    One client needs to start decryption,
    so might as well be the first one."""
    ct = fhe_deserialize_string(ct_string, "Ciphertext", ST)
    
    # Aggresiv garbage collection.
    del ct_string
    gc.collect()

    if id == 0:
        pd = cc.MultipartyDecryptLead([ct], sk)[0]
    else:
        pd = cc.MultipartyDecryptMain([ct], sk)[0]

    pd_string = fhe_serialize_string(pd, ST)

    # Write parital decryptions to files:
    # for i in range(len(pd_list)):
    #     fhe_serialize_file(pd_list[i], storage+f"pd{i}", ST)
    
    # For ease of programming, hardcode that the 
    # client with partition 0 launches the server.
    # This way I can open a specific port on the container.
    partition_id = context.node_config["partition-id"]
    addr = context.node_config["addr"]
    # Num-partions is equivalent to total number of clients,
    # but one client launches server, so only need num-partitions-1 clients.
    num_clients = context.node_config['num-partitions'] - 1

    if partition_id == 0:
        """Create Server which reads and then deletes
        partial decryption files."""
        # Launch superlink.
        args_1 = f'flower-superlink --insecure '
        args_2 = f'--serverappio-api-address {addr}:10091 ' 
        args_3 = f'--fleet-api-address {addr}:10092 '
        args_4 = f'--exec-api-address {addr}:10093'
        args = args_1 + args_2 + args_3 + args_4
        log(INFO, f"Command to run to launch superlink: {args}")
        super = subprocess.Popen(args=args,
                                     shell=True,
                                     start_new_session=True,
                                     )
        time.sleep(2) # To avoid flwr run being done to soon.
        """Kanskje bedre å sjekke om porten 10093
        er åpen, slik som jeg gjorde
        i essaykoden med funksjonen is_port_in_use."""
        args1 = f'flwr run flwr-decryption-no-server/ '
        args2 = f"-c 'num-ct=1 num-clients={num_clients}' "
        args3 = f"--federation-config 'options.num-supernodes={num_clients}'"
        args = args1 + args2 + args3
        log(INFO, f"Command for flwr run: {args}")
        flwrrun = subprocess.Popen(args=args,
                                   shell=True,
                                   start_new_session=True,
                                   )
        
        # Start multiprocess.connection Listener.
        # Cannot pass any information to superlink,
        # so must give unique port.
        listener = mpc.Listener(("127.0.0.1", 12360))
        # accept waits until a mpc.Client connects.
        connection = listener.accept()
        # When client is connected, can send partial decryptions.
        connection.send(pd_string)
        # Close connection after sending.
        connection.close()
        listener.close()

    else:      
        """Create Client which reads and then deletes
        partial decryption files."""

        # Launch supernode.        
        args_1 = f'flower-supernode --insecure '
        args_2 = f'--superlink {addr}:10092 '
        args_3 = f"--node-config 'id={id}' "
        args_4 = f'--clientappio-api-address 0.0.0.0:{10000+id}'
        print(f"Command to launch supernode: \n{args_1+args_2+args_3+args_4}")
        args = args_1 + args_2 + args_3 + args_4
        super = subprocess.Popen(args=args,
                                     shell=True,
                                     start_new_session=True,
                                     )

        listener = mpc.Listener(("127.0.0.1", 12340+id))
        # accept waits until a mpc.Client connects.
        connection = listener.accept()
        # When client is connected, can send partial decryptions.
        connection.send(pd_string)
        # Close connection after sending.
        connection.close()
        listener.close()

    if partition_id == 0:
        listener = mpc.Listener(("127.0.0.1", 12360))
        # Wait until subserver open connections.
        # Then the server can be closed.
        connection = listener.accept()
        # When client is connected, can receive partial decryptions.
        pds = connection.recv()
        connection.close()
        listener.close()
        os.killpg(os.getpgid(super.pid), signal.SIGTERM)

        # Deserialize pds and fuse to get decryption.
        pd_list = [fhe_deserialize_string(pd, "Ciphertext", ST) for pd in pds]
        plaintext = cc.MultipartyDecryptFusion(pd_list)
        noise_estimate = plaintext.GetLogError()
        # Save noise estimate to client state.
        context.state.config_records['noise-estimate'] = ConfigRecord({'noise-estimate': noise_estimate})
        
        # Give empty reply to server.
        # This is simply to avoid Flower crashing before the code is even run.
        recordset = RecordDict()
        reply = Message(content=recordset,
                        reply_to=msg)
        
        # Aggresiv garbage collection.
        gc.collect()

        return reply
    else:
        listener = mpc.Listener(("127.0.0.1", 12340+id))
        # This one waits until subprocess is ready to send partial decryptions back.
        # accept waits until a mpc.Client connects.
        connection = listener.accept()
        # When client is connected, can receive partial decryptions.
        pds = connection.recv()
        # Close connection after sending.
        connection.close()
        listener.close()
        os.killpg(os.getpgid(super.pid), signal.SIGTERM)

        # Deserialize pds and fuse to get decryption.
        pd_list = [fhe_deserialize_string(pd, "Ciphertext", ST) for pd in pds]
        plaintext = cc.MultipartyDecryptFusion(pd_list)
        noise_estimate = plaintext.GetLogError()
        # Save noise estimate to client state.
        context.state.config_records['noise-estimate'] = ConfigRecord({'noise-estimate': noise_estimate})

        # Give empty reply to server.
        # This is simply to avoid Flower crashing before the code is even run.
        recordset = RecordDict()
        reply = Message(content=recordset,
                        reply_to=msg)
        
        # Aggresiv garbage collection.
        gc.collect()

        return reply

@app.query("create_single_ciphertext")
def create_single_ciphertext(msg: Message, context: Context):
    """Create a single ciphertext which will be used for noise estimation.
    Values of the ciphertext must be representative of the actual values,
    so the easiest is probably to randomly samples values from the interval
    of scaled values."""
    id = msg.content.config_records['ciphertext']['id']

    # Load scaled weights from state.
    weights_array = context.state.array_records['weights']
    # Need to convert from Flower Array to lists of arrays.
    weights = [weights_array[key].numpy() for key in list(weights_array.keys())]

    # Det aller enkleste er kanksje å bare bruke maksverdien i alle slots?
    max_weights = []
    for weight in weights:
        max_weights.append(np.max(weight))
    true_max = np.max(max_weights)

    # Get batchsize from ring dimension.
    batchSize = context.run_config['ring-dim']//2
    # Get cryptocontext and jpk.
    st = context.run_config['ser-type']
    if st == "BINARY":
        ST = fhe.BINARY
    elif st == "JSON":
        ST = fhe.JSON
    storage = os.path.abspath('storage/') + "/"
    cc = fhe_deserialize_file("CryptoContext", storage+"cc", ST)
    jpk = fhe_deserialize_file("PublicKey", storage+"jpk", ST)

    # Create ciphertext and serialize.
    # Fill whole ciphertext with max value.
    pt = cc.MakeCKKSPackedPlaintext([true_max] * batchSize)
    ct = cc.Encrypt(jpk, pt)
    ct_string = fhe_serialize_string(ct, fhe.BINARY)

    # Construct and return reply.
    recordset = RecordDict()
    dec_record = ConfigRecord({'ct': ct_string})
    recordset['ct'] = dec_record
    reply = Message(content=recordset,
                    reply_to=msg)
    return reply



@app.query("weights_blocks")
def weights_blocks(msg: Message, context: Context):

    """The purpose of this method is to 
    send small chunks of encrypted weights to the server for
    aggregation.
    The weights generated in the 'training' method are stored in context
    alongside a list of indicies to keep track of where 
    ciphertexts fit into the weights list arrays,
    and metadata for which round the weights are for.
    Only need to keep the weights for one round,
    so can replace the weights of round i 
    with fresh weights when working round i+1.
    The refreshing part will be handled in 'training' method,
    this method is only concerned with sending smaller chunks of 
    the weights to the server.
    Must do encryption inside here, since being 
    unable to transfer ciphertexts is the whole reason
    this is being done. 
    """
    # Aggresiv garbage collection.
    gc.collect()

    # Serialization type
    st = context.run_config['ser-type']
    if st == "BINARY":
        ST = fhe.BINARY
    elif st == "JSON":
        ST = fhe.JSON
    # Scheme to be used.
    SCHEME = context.run_config['scheme']

    # Assume that server sends info about how many chunks it has received so far.
    # Parse message contents.
    id = msg.content.config_records['config']['id']
    # log(INFO, f"config_records.keys() = {msg.content.config_records.keys()}.")
    # pos marks the first position in 'struct' which has not been processed yet.
    # I.e., the server has processed struct[0], ..., struct[pos-1].
    pos = msg.content.config_records['config']['pos']
    # Use variable chunk size provided by server.
    # Need to get it from server since cs can differ depending
    # on what pos is.
    cs = msg.content.config_records['config']['cs']

    # Load cc and jpk.
    storage = os.path.abspath('storage/') + "/"
    cc = fhe_deserialize_file("CryptoContext", storage+"cc", ST)
    jpk = fhe_deserialize_file("PublicKey", storage+"jpk", ST)

    # Load weights and struct from context.
    weights_array = context.state.array_records['weights']
    struct_array = context.state.array_records['struct']

    # Need to convert from Flower Array to lists of arrays.
    weights = [weights_array[key].numpy() for key in list(weights_array.keys())]
    struct = struct_array.to_numpy_ndarrays()[0] # ?
    """Actually, what structure will the struct array have?
    The plan is for struct to be a 1D list of tuples, 
    so need to figure out how to convert from Array.
    Assume it is converted correctly for now."""

    """struct uses the format (i, j, k)
    which indicates that ciphertext at struct[x]
    comes from either weights[i][j][k] or
    weights[i][j] if k == -1 or
    weights[i] if j == k == -1."""

    """The various indices lists might have different lengths,
    since the lengths of weights differ.
    Thus, the lists must be padded to some fixed length
    to ensure they can be converted to ndarrays.
    Pad with -1's and have a check on the server
    to ignore those indices."""

    # Now need to create the next chunk of weights.
    plaintexts = []
    ciphertext_indices = []
    weights_chunk = []
    indices = []
    BATCHSIZE = context.run_config['ring-dim'] // 2
    log(INFO, f"pos = {pos}, pos+cs = {pos+cs}")

    for p in range(pos, pos+cs):
        # log(INFO, f"sum = {sum([struct[x][0] for x in indices])+struct[p][0]}")
        if (len(indices)+1) <= BATCHSIZE:
            # Capacity has not yet been reached. Can add more to ciphertext.
            # Concatanate the weight list with additional weights.
            """struct[p] might contain one or more -1 entries
            which should be ignored."""
            idx = [j for j in struct[p] if j != -1]
            """The list index must be handled separetly, but arrays
            can be indexed by a tuple. If len(idx) is 0 then
            the tuple is simply empty and the whole array is returned.
            Thus, there is no need for an if-test to check if it is empty,
            as I had previously."""
            # For CKKS the weights can be added as is.
            weights_chunk.append(weights[idx[0]][tuple(idx[1::])])
            indices.append(p)
        else:
            # Capacity has been reached.
            # Append to plaintexts and empty storages.

            # Pad indices if it is too small.
            if len(indices) < BATCHSIZE:
                # Pad using list concatanation
                indices += [-1 for _ in range(BATCHSIZE-len(indices))]

            plaintexts.append(weights_chunk)
            ciphertext_indices.append(indices)
            weights_chunk = []
            indices = []
            # # Aggresiv garbage collection.
            gc.collect()
        """Must combine multiple weights into a single
        ciphertext.
        Can combine 8 weights with length 128,
        or 2 weights with length 512
        with current batch size of 1024.
        For encryption, I can probably just
        concatanate lists with list addition,
        but I am not sure how I should do insertion
        into the weights structure after decryption.
        
        Probably best to just send a list 
        which contains the p-values used
        in each ciphertext.
        I.e., if ciphertext[i] contains
        the weights from struct[x], struct[x+1], struct[x+2],
        then list[i] = [x, x+1, x+2].
        Can then use the lengths l and list
        slicing to divide the result into the proper sizes."""

    """The loop only saves weigths_chunk and indices if there 
    are enough weights to fill a whole ciphertext.
    So the last weights_chunk and indices will not be
    saved if there are not enough weights.
    Thus, the last weights_chunk and indices should always be saved.
    
    This also handles the case when CS is small enough
    that not even a single ciphertext can be filled,
    which was previously handled by its own if test.
    
    Must also do padding in case."""
    if len(weights_chunk) > 0:
        # Pad indices if it is too small.
        if len(indices) < BATCHSIZE:
            # Pad using list concatanation
            indices += [-1 for _ in range(BATCHSIZE-len(indices))]
        plaintexts.append(weights_chunk)
        ciphertext_indices.append(indices)

    # Convert chunk from weights to ciphertext byte serializations.
    ciphertexts = weights_to_ciphertext_bytes(plaintexts, 
                                              cc,
                                              jpk,
                                              SCHEME)
    
    # Aggresiv garbage collection.
    del plaintexts
    gc.collect()

    # Convert ciphertexts list to ndarray, to use ArrayRecord
    # which is recommended for large data structures.
    """I experienced problems when trying to deserialize
    ciphertexts which have been stored in an ArrayRecord.
    Using a ConfigRecord seems to work fine,
    as the individual chunks are not that large."""
    recordset = RecordDict()
    # array_record = ArrayRecord(numpy_ndarrays=[np.asarray(ciphertexts, dtype=str)])

    # Must convert ciphertext_indices to ArrayRecord,
    # since it is a list of lists, which is not supported by ConfigRecord.
    array_record = ArrayRecord(numpy_ndarrays=[np.asarray(ciphertext_indices)])

    config_record = ConfigRecord({'weights': ciphertexts, 
                                #   'indices': ciphertext_indices
                                  })
    # recordset['weights'] = array_record
    recordset['weights'] = config_record
    recordset['indices'] = array_record

    reply = Message(content=recordset,
                    reply_to=msg)
    log(INFO, f"Node {id} sending reply to server.")
    
    """If all weights have been sent, the weights and struct
    can be removed from state to save memory.

    Currently, running the docker version ends with the at least one container,
    usually the ServerApp being shut down after 1 round.

    The shut down container has error code 137 which seems to indicate 
    that it received SIGKILL from OS, which might happen because 
    the system is running low of memory.

    Hopefully, freeing up memory will resolve this.
    """

    # Aggresiv garbage collection.
    gc.collect()

    """If pos+cs == len(struct), all elements of struct must have been handled.
    So in that case, clean up the client's state and force garbage collection.
    The values must be ArrayRecords, but hopefully it helps to replace them with empty arrays."""
    if pos+cs == len(struct):
        context.state.array_records['weights'] = ArrayRecord(numpy_ndarrays=[np.asarray([])])
        context.state.array_records['struct'] = ArrayRecord(numpy_ndarrays=[np.asarray([])])
        # Force garbage collection to run as well.
        gc.collect()


    return reply

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

    # Load the model and initialize it with the received weights
    # 28*28 = 784, so 784 features.
    scale = context.run_config["scale"]
    if scale == "classic":
        model = LogisticRegression(28*28, 10)
    elif scale == "larger":
        model = LogisticRegression(224*224, 10)
    elif scale == "rgb":
        model = LogisticRegression(3*224*224, 10)
    set_weights(model, weights)
    trainloader, valloader = load_data(partition_id, num_partitions)
    log(INFO, f"Node {id} training model.")
    # Maybe I must call ndarray_to_parameters(weights) before I can use them?
    # parameters = ndarrays_to_parameters(weights)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    training_start = time.time()
    # Fit model to the data.
    train(model, trainloader, local_epochs, device)
    end_time = time.time()
    weights = get_weights(model) 
    size = len(trainloader.dataset)
    log(INFO, f"Node {id} done with training. Used {end_time-training_start} s.")

    
    # Lag path hvis den ikke eksisterer.
    current_path = Path("./storage/logreg-fhefedavg-training-time")
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
    log(INFO, f"Node {id} has computed its result. Doing encryption.")

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
    log(INFO, f"Node {id} sending reply to server.")

    # Aggresiv garbage collection.
    gc.collect()

    return reply
    


    """When I get the training results,
    I can compute get_weights(self.net) * len(self.trainloader)
    and then encrypt,
    rather than encrypt both and make the server do multiplication
    on ciphertexts. Saves some computation that way.
    Don't actually need to do any multiplication on ciphertexts
    then? So I can set multiplicative depth / level to be only 1."""

    """Should trainloader, valloader be created each time?
    Or must I find some other wayt to do it."""


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
    trainloader, valloader = load_data(partition_id, num_partitions)

    # Load the model and initialize it with the received weights
    # 28*28 = 784, so 784 features.
    scale = context.run_config["scale"]
    if scale == "classic":
        model = LogisticRegression(28*28, 10)
    elif scale == "larger":
        model = LogisticRegression(224*224, 10)
    elif scale == "rgb":
        model = LogisticRegression(3*224*224, 10)
    set_weights(model, weights)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Evaluate model on the data
    print("Beginning evaluation!")
    start_time = time.time()
    loss, accuracy = test(model, valloader, device)
    end_time = time.time()
    print("Finished evaluation!")

    # Lag path hvis den ikke eksisterer.
    current_path = Path("./storage/logreg-fhefedavg-evaluation-time")
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