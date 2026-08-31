import os
from logging import INFO
import time
import gc # garbage collection
import subprocess
import signal
import multiprocessing.connection as mpc
import time

from flwr.app import (Context, 
                         Message, 
                         ConfigRecord, 
                         RecordDict)
from flwr.common.logger import log
import numpy as np
import openfhe as fhe

from fhefedavg.fhe.openfhe_help_functions import (fhe_deserialize_file,
                                                  fhe_deserialize_string, 
                                                  fhe_serialize_string)


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
        args_2 = f'--host {addr} --port 10091 ' 
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
        args_4 = f'--host 0.0.0.0 --port {10000+id}'
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