import multiprocessing.connection as mpc

from flwr.client import ClientApp
from flwr.common import (Context, 
                         Message,
                         RecordDict)



# Flower ClientApp
app = ClientApp()

@app.query("send_pds")
def send_pds(msg: Message, context: Context):
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