import time
from logging import INFO
import multiprocessing.connection as mpc

from flwr.app import (Context,
                         RecordDict, 
                         Message, 
                         ConfigRecord)
from flwr.common.logger import log
from flwr.serverapp import Grid, ServerApp


# Create ServerApp
app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    # Loop and wait until enough nodes are available.
    min_nodes = context.run_config["num-clients"]
    node_ids = []
    while len(node_ids) < min_nodes:
        node_ids = list(grid.get_node_ids())
        log(INFO, "Waiting for nodes to connect...")
        time.sleep(2)

    print("Receiving partial decryptions from parents.")
    client = mpc.Client(("127.0.0.1", 12360))
    """pds is list of bytes.
    Each bytes object corresponds to a different partial decryption."""
    pds = client.recv()

    print("Sending partial decryptions to all clients.")
    # pds must be distributed to all clients.
    recordset = RecordDict()
    config_record = ConfigRecord({})
    config_record['pds'] = pds
    # recordset['weights'] = array_record
    recordset['pds'] = config_record
    messages = []
    for i in range(len(node_ids)):
        message = Message(
            content=recordset,
            message_type="query.send_pds",  
            # target query('send_pds') method in ClientApp
            dst_node_id=node_ids[i],
            group_id="decryption",
        )
        messages.append(message)
    # Don't actually care about replies.
    grid.send_and_receive(messages)

    print("Opening connection to parent.")    
    client = mpc.Client(("127.0.0.1", 12360))
    """When connection is opened, parent client
    knows that subserver is done and can be shut down."""

    