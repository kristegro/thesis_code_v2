# openfhe-fedavg-docker-scraping directory

Project to run Federated Averaging with OpenFHE through Flower. <br>
This version is intended to run through docker.
This version also scrapes metrics from the containers by using cAdvisor, Prometheus and Grafana.

## Required directories 

### storage/key_store

Requires that storage/key_store directory exists at the root directory of the repo.
storage/key_store must contain a sub directory for the server and one for each client.
The subdirectories for the clients should be named 

> node0, node1, ..., node{N-1}

where N is the total number of clients needed.

Additionally, the ownership of storage/key_store and all sub directories must be set to
user and group 

> 49999

This can be done by running

> sudo chown 49999:49999 -R storage/key_store

The 'other' category must also gain write access to the dir so that key generation can work.
This can be given by running

> sudo chmod 777 -R storage/key_store

### storage/results_docker

Also requires that a directory storage/results_docker exists.
Run results will be written to a file 

> benchmark-openfhe-fedavg-docker.txt

in this directory.

The ownership of storage/results_docker and all sub directories should either be set to
user and group 

> 49999

which can be done by running

> sudo chown 49999:49999 -R storage/results_docker

Or the 'other' category must also gain write access to the dir.
This can be given by running

> sudo chmod 777 -R storage/results_docker

Either of these options must be done to ensure that the containers have write access to write to

> benchmark-openfhe-fedavg-docker.txt

The safest option is to do both.

### storage/grafana-storage

Storage directory for Grafana. Grafana needs permission to write to it.
From the docker image it seems that Grafana runs as user '472',
so can either change ownership or give 'other' category write access.

> sudo chown 472:472 -R storage/grafana-storage

or

> sudo chmod 777 -R storage/grafana-storage

### storage/prometheus-storage

Storage for Prometheus for data persistency. Prometheus needs permision to write to it.
Either set ownership to 65534:65534 (user:group) or give 'Other category write access.

> sudo chown 65534:65534 -R storage/prometheus-storage

or 

> sudo chmod 777 -R storage/prometheus-storage

## Install depencies, build and run

### Install 

All dependencies will be installed directly into the docker containers.

Flower itself still needs to be installed to get access to the CLI.

It can be convinent to install dependencies into a virtual environment.
To create a virtual environment '.venv' inside this program's directory do

> python3 -m venv openfhe-fedavg-docker-scraping/.venv

To activate this env do:

> source openfhe-fedavg-docker-scraping/.venv/bin/activate

To deactivate the environment, run in a terminal:

> deactivate

To install Flower run

> pip install flwr

### Build

First, the ClientApp and ServerApp images must be built. This can be done by:

> docker compose -f dockerfiles/compose-base.yaml build

However, it might be too resource intensive to build both at once.
In this case, they can be built one at a time by 

> docker compose -f dockerfiles/compose-base.yaml build serverapp-task-build
>
> docker compose -f dockerfiles/compose-base.yaml build clientapp-task-build

Afterwards the containers can be started by

> docker compose -f dockerfiles/compose-scraping.yaml up

If some other number than 5 clients are needed, the 'dockerfiles/compose-scraping.yaml'
must be modified, either by adding more services or by commenting out uneeded clients. <br>
In 'pyproject.toml' inside this directory, the value of 
 
> options.num-supernodes

must also be changed to this new number of clients.

### Run

To run, first start the services:

> docker compose -f dockerfiles/compose-scraping.yaml up

Then use flwr run as normal:

> flwr run openfhe-fedavg-docker-scraping

When running, can also adjust the run configuration or federation configuration for that specific run.
The option 

> -c 'key1=value1 key2=value2'

adjusts run configuration. keyi must be a valid key in pyproject.toml and valuei must be valid.

The option 

> --federation-config 'key1=value1 key2=value2'

adjusts federation configuration. keyi must be a valid key in pyproject.toml and valuei must be valid.

Both can be used together:

> flwr run openfhe-fedavg-docker -c 'key1=value1 key2=value2' --federation-config 'key1=value1 key2=value2'

Valid choices for keys can be found below under the pyproject.toml header.

To stop services afterwards:

> docker compose -f dockerfiles/compose-scraping.yaml down

### Access Grafana Dashboard:

To access the grafana dashboard, simply open a localhost page to the port that Grafana runs on in a browser. E.g, if the Grafana docker container opens port 3000, open `http://localhost:3000/` in a browser.

## File overview

### client_app.py

Contains methods which are executed by a ClientApp according <br>
to the type of message received from the server.

Also contains some help functions.

#### Function overview

##### weights_to_ciphertext_bytes

This function converts a list of lists into a list of ciphertexts. It takes the following arguments:

- weights, the list of arrays to encrypts.
- cc, the crypto context to use for encryption.
- jpk, the joint public key to use for encryption.
- scheme, a string which indicates which scheme is used, either 'CKKS' or 'BFV'. This is needed since CKKS and BFV use different ways to convert vectors to plaintexts.

The return value is a list of byte serialized ciphertexts which encrypt each of the arrays. These bytes can then be sent directly to the server.

##### recursive_find_struct

One argument is a list of numpy arrays, which themselves might contain numbers or numpy arrays of numbers, i.e. there might be 2 dimensions or 3 dimensions depending on the indices. This list is known as the weights.
The other argument is a list of integers which is used internaly to track the indices needed to get to a specific point in the recursion.

This is a recursive method for discovering the structure of the weights argument, which will allow the clients to parse weights of any structure. The function recursively explores the numpy arrays until it encounters an array of numbers. The index path taken to reach this point and the length of the number array is then added as an element to a list 'struct'. These elements have the form (l, i, j, ..., z)

- l is the length of the deepest dimension, e.g. len(weights\[i\]\[j\]) if the weights are 3D at those indices. l is used by the server to know where one weight ends and another begins when multiple weights are packed into the same ciphertext.
- (i, j, ..., z) are the indices used to get to the element, i.e., the element is positioned at weights\[i\]\[j\]...\[z\].

The mentioned 'struct' list is returned.

##### find_struct

This function wraps recursive_find_struct (above) so that clients can call it. It also pads out each element of the returned struct list to the same length to ensure that the list can be converted to a numpy array to save it in the client state.

It takes the weights, a list of numpy arrays, which themselves might contain numbers or numpy arrays of numbers, i.e. there might be 2 dimensions or 3 dimensions depending on the indices, as an argument.

The padded struct list is returned.

##### dec_with_server

This method is called at orders from the server. The message should contain

- id, an integer between 0 and N-1 where N is the number of clients. 
- ct, a ciphertext to decrypt.

Each client decrypts the ciphertext with their own secret key and send the partial decryption back to the server.

##### dec_with_server_multiple

This method is similar to dec_with_server (above), but the server send a list of ciphertexts instead of just one.
The message from the server should contain 

- ct_list, a list of ciphertexts to decrypt
- id, the id of the client.

In total, using dec_with_server_multiple for both client and server is more efficient since the OpenFHE functions for partial decryption take lists anyway, so decrypting larger lists is more efficient. The decryption methods likely use some degree of parallelization.

Each client decrypt all the ciphertexts and reply to the server with a list of partial decryptions.

##### weights_blocks

This method is responsible for partitioning the weights structure into smaller blocks which can be aggregated by the server on at a time.

The client loads the previously created 'struct' list and weights from its state.

The message from the server should contain:

- id, the client's id.
- pos, the index of the struct list to start from.
- cs, the number of elements in struct list to handle at once.

The weights are parsed to pack as many as possible into each ciphertext,
and in total all weights recorded in struct\[pos\] to struct\[pos+cs\]
will be packed into lists. These lists are then sent through the weights_to_ciphertext_bytes() function (see above) before the bytes are sent back to the server.

##### training

This method handles model training. The client receives weights from the server which it uses to train the model on its local data. When training is done, the structure of the new weights are parsed by the find_struct() function (above) and the returned struct list and the weights themselves are saved in the client's state.

The message from the server should include:

- id, the id of the client.
- weights, the weights to be used for training.

The size of the data set used for traning is encrypted and returned to the server.
The struct list is also returned to the server.

##### evaluate

Evaluate the model based on weights received from the server.

The message from the server should include:

- id, the client's id.
- weights, the weights to evaluate the model with.

The metrics returned from evaluation are sent back to the server.

### server_app.py

Contains code which is run on a ServerApp on the server.

Also contains some help functions.

#### Function overview

##### dec_with_server

This function is used by the server to request the decryption of a single ciphertext.
Each client replies which a partial decryption which this function returns a list of.
The partial decryptions are then fused together in main (below).

Arguments are:

- grid, the Flower Grid instance which represents the server.
- node_ids, the ids of the clients which should participate in decryption, which are all of them.
- ct, the ciphertext to request decryption of.
- ST, the serialization type that will be used to communicate ciphertext and partial decryptions back and forth.

##### dec_with_server_multiple

This function is used by the server to request the decryption of a list of ciphertexts.
Each client replies which a list of partial decryptions, one for each ciphertext.
This function returns a nested list where the first dimension contains a list for each client, which again contains partial decryptions for each ciphertext.
The partial decryptions of each ciphertext are then fused together in main (below).

Arguments are:

- grid, the Flower Grid instance which represents the server.
- node_ids, the ids of the clients which should participate in decryption, which are all of them.
- ct_list, list of the ciphertext to request decryptions of.
- ST, the serialization type that will be used to communicate ciphertext and partial decryptions back and forth.

##### main

This method handles the majority of the server's work.
The server initializes a model and engages in Federated Aggregation with the clients.
Decryption is handled by dec_with_server and dec_with_server_multiple (above).

The method also contains outcommented code to verify that weights are properly reconstructed based on the struct list and received chunks.

### task.py

Contains functions to handle training and evaluation <br>
of the model, independently of Flower communication.

Everything in this file comes from running

> flwr new

with the HuggingFace option, unless stated otherwise.

#### Function overview

##### load_data

This function loads the chosen dataset as specified by the 'model-name' run configuration option.

##### train

This function trains the model. It is called by the clients in the training method.

##### test

This function evaluates the result of training. It is called by the clients in the evaluate method.
This function is modified from its original form to also extract the F1-score of the model.
So in addition to returning loss and accuracy, it also returns the F1-score.

##### get_weights

This function retrieves the model weights.

##### set_weights

This function updates the model weights with new values.

### pyproject.toml

Contains dependencies and configuration options for the program. <br>
Parameters that can be changed are:

- Run configurations:

    - num-server-rounds:

        Used by Flower to indicate how many rounds of training etc that should be done.

    - fraction-fit

        The fraction of clients which should participate in training.
        The number of participating clients becomes 

        > max(1, fraction-fit * number-of-clients)

    - local-epochs

        The number of local training rounds a client should do before sending the weights back to the server.

    - model-name

        The name of the model to use, as it is named on the HuggingFace website.

    - num-labels

        Something related to either the dataset being used or the model. I am not sure. <br>
        Best to leave it as the default value of 2.

    - scheme

        The type of FHE scheme to use for encryption. Currently only "CKKS" is really supported, <br>
        since "BFV" and "BGV" only works on integers and would need additional methods to work.

    - ser-type

        The type of serialization to use when deserializing keys, <br>
        and serializing/deserializing ciphertexts for sending/receiving. <br>
        Supported values are "BINARY" and "JSON".

    - batchsize

        The batch size used in the keys. This is the maximum amount of number
        which can be packed into the same ciphertext.

    - chunk-size

        The amount of weights being encrypted and processed at once.
        Should be set according to batchsize, so that all ciphertexts are filled as much as possible. <br>
        If set to -1, all weights will be processed at once.

- Federation configurations: 

    - options.num-supernodes 

        The total number of clients to participate.

    - address 

        A string indicating the address and port that 'flwr run' communicates with.
        Must match an open port on the machine running the SuperLink.
        Defaults to "127.0.0.1:9093"

    - insecure

        'true' or 'false'. Defaults to 'true'. Indicates whether or not the SuperLink and SuperNodes should
        communicate with HTTPS or not. Must supply certificates and such if 'false'.