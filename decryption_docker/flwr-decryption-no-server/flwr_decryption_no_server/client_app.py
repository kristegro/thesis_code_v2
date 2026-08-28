import multiprocessing.connection as mpc

from flwr.client import ClientApp
from flwr.common import (Context, 
                         Message,
                         RecordDict,
                         ConfigRecord)



# Flower ClientApp
app = ClientApp()

@app.query("pds_to_server")
def pds_to_server(msg: Message, context: Context):
    print("Receiving partial decryption from parent.")
    id = context.node_config['id']
    client = mpc.Client(("127.0.0.1", 12340+id))
    pd = client.recv()
    
    print("Sending reply to server.")
    recordset = RecordDict()
    config_record = ConfigRecord({'pd': pd})
    recordset['pd'] = config_record
    reply = Message(content=recordset,
                    reply_to=msg)
    return reply


@app.query("pds_from_server")
def pds_from_server(msg: Message, context: Context):
    """Receive all pds from server.
    Serialize to file and give "empty" reply."""

    print("Receiving partial decryptions from server.")
    id = context.node_config['id']
    pds = msg.content.config_records['pds']['pds']
    
    print("Sending pds to parent and reply to server.")
    client = mpc.Client(("127.0.0.1", 12340+id))
    client.send(pds)

    # Give empty reply to server.
    # This is simply to avoid Flower crashing before the code is even run.
    recordset = RecordDict()
    reply = Message(content=recordset,
                    reply_to=msg)
    return reply