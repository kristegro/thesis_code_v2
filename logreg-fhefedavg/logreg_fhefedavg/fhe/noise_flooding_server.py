import os
import sys
import time
from logging import INFO
from random import sample
import gc # garbage collection

from flwr.common import (RecordDict, 
                         Message, 
                         ConfigRecord)
from flwr.common.logger import log
import openfhe as fhe

# Add /app, which contains all help functions, to import path.
module_dir = os.path.abspath("/app")
if module_dir not in sys.path:
    sys.path.insert(0, module_dir)

from openfhe_help_functions import (fhe_deserialize_file, 
                                    fhe_deserialize_string, 
                                    fhe_serialize_string)
from keygen import (server_keygen_noise_flooding,
                    server_keygen_noise_flooding_noek)


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
            # target query('dec_without_server') method in ClientApp
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
    print("Calling noise flooding from create_noise estimate")
    if ek_needed:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("calling noise_flooding")
        server_keygen_noise_flooding(grid, context, "estimate")
    else:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("calling noise_flooding_noek")
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
            # target query('dec_without_server') method in ClientApp
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