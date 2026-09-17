import multiprocessing.connection as mpc
import time
import subprocess
import requests
from ast import literal_eval
import pprint
import os
from pathlib import Path
import shutil
import gc
import argparse

from prometheus import request_to_dict, dict_to_csv, runtime_to_csv_single_pc, move_models
import numpy as np

"""Get arguments from commandline."""
parser = argparse.ArgumentParser(description='Run benchmarking for chosen parameter combination.')
parser.add_argument('program', metavar='program', type=str,
                    help='Program to benchmark',
                    choices=['orgfedavg', 'fhefedavg', 'secagg'])
parser.add_argument('num_clients', metavar='num_clients', type=int,
                    help='Number of clients',
                    choices=[3, 5, 7])
parser.add_argument('model', metavar='model', type=str,
                    help='Model to use',
                    choices=['squeezenet', 'bert', 'logreg'])
parser.add_argument('num_tests', metavar='num_tests', type=int,
                    help='Number of tests to run for the same parameters',
                    choices=range(1,11))
parser.add_argument('scheme', metavar='scheme', type=str,
                    help='Encryption scheme to use',
                    choices=['CKKS', 'CKKS-NF', 'BGV', 'BFV', 'None'],
                    default=None)
parser.add_argument('sec_level', metavar='sec_level', type=str,
                    help='Security level to achieve',
                    choices=['128c', '256c', 'None'],
                    default=None)
def power_of_two(string):
    if string == "None":
        return string
    lst = string.split("*")
    """For string '2**X'
    lst becomes ['2', '', 'X']"""
    number = int(lst[0])**int(lst[2])
    if number < 2**13 or number > 2**17:
        msg = f"Ring dimension must be atleast 2**13 and at most 2**17."
        raise argparse.ArgumentTypeError(msg)
    return number
parser.add_argument('ring_dim', metavar='ring_dim', type=power_of_two,
                    help='Ring dimension. For sec_level 128c must be 2**13, ' \
                    'for sec_level 256c must be atleast 2**14.',
                    default=None)
def multiple_of_two(string):
    if string == "None":
        return string
    if "*" not in string:
        return int(string)
    lst = string.split("*")
    """For string 'X*131072'
    lst becomes ['X', '131072']"""
    return int(lst[0])*int(lst[1])
parser.add_argument('chunk_size', metavar='chunk_size', type=multiple_of_two,
                    help='The size of the blocks to divide the weights into when encryting.',
                    choices=[i*131072 for i in range(1,21)] + ['None'])
parser.add_argument('num_coeffs', metavar='num_coeffs', type=int,
                    help='Number of coefficients to use in polynomial encoding.',
                    choices=[16,32, 'None'])
parser.add_argument('mult_tech', metavar='mult_tech', type=str,
                    help='Multiplication techinque for BFV',
                    choices=['HPS', 'BEHZ', 'None'],
                    default=None)
parser.add_argument('delta_bits', metavar='delta_bits', type=str,
                    help='Should high (50) or low (15) bit count be used for delta.',
                    choices=['high', 'low', 'None'],
                    default=None)

parameters = dict(vars(parser.parse_args()))

# Configurer integer polynomial degree for ulike modeller.
if parameters['model'] == 'bert':
    if parameters["num_clients"] == 3:
        parameters["max_deg_int"] = 9
    elif parameters["num_clients"] in [5,7]:
        parameters["max_deg_int"] = 8
elif parameters["model"] == "squeezenet":
    if parameters["num_clients"] == 3:
        parameters["max_deg_int"] = 9
    elif parameters["num_clients"] in [5,7]:
        parameters["max_deg_int"] = 8
elif parameters['model'] == 'logreg':
    if parameters["num_clients"] == 3:
        parameters["max_deg_int"] = 9
    elif parameters["num_clients"] in [5,7]:
        parameters["max_deg_int"] = 8

# Konfigurer max-weight parameter for secagg.
if parameters['program'] == "secagg":
    if parameters['model'] == "bert":
        if parameters['num_clients'] == 3:
            parameters['max_weight'] = 15001
        elif parameters['num_clients'] == 5:
            parameters['max_weight'] = 9000
        elif parameters['num_clients'] == 7:
            parameters['max_weight'] = 6429
    elif parameters['model'] == "squeezenet":
        if parameters['num_clients'] == 3:
            parameters['max_weight'] = 12000
        elif parameters['num_clients'] == 5:
            parameters['max_weight'] = 7200
        elif parameters['num_clients'] == 7:
            parameters['max_weight'] = 5143
    elif parameters['model'] == "logreg":
        if parameters['num_clients'] == 3:
            parameters['max_weight'] = 12000
        elif parameters['num_clients'] == 5:
            parameters['max_weight'] = 4800
        elif parameters['num_clients'] == 7:
            parameters['max_weight'] = 3429

# Konfigurer chunk-size for fhe schemes:
if parameters['scheme'] in ['BGV', 'BFV']:
    if parameters['model'] in ['logreg']:
        parameters['chunk_size'] = 10*131072
    elif parameters['model'] == 'squeezenet':
        if parameters['num_clients'] == 3:
            parameters['chunk_size'] = int(np.ceil(727_627/3))
        elif parameters['num_clients'] == 5:
            parameters['chunk_size'] = int(np.ceil(727_627/10))
        elif parameters['num_clients'] == 7:
            parameters['chunk_size'] = int(np.ceil(727_627/21))
    elif parameters['model'] == 'bert':
        if parameters['num_clients'] == 3:
            parameters['chunk_size'] = int(np.ceil(4_386_179/18))
        elif parameters['num_clients'] == 5:
            parameters['chunk_size'] = int(np.ceil(4_386_179/60))
        elif parameters['num_clients'] == 7:
            parameters['chunk_size'] = int(np.ceil(4_386_179/126))
elif parameters['scheme'] == 'CKKS-NF':
    if parameters['model'] in ['logreg']:
        parameters['chunk_size'] = 10*131072
    elif parameters['model'] == 'squeezenet':
        if parameters['num_clients'] == 3:
            parameters['chunk_size'] = int(np.ceil(727_627))
        elif parameters['num_clients'] == 5:
            parameters['chunk_size'] = int(np.ceil(727_627/2))
        elif parameters['num_clients'] == 7:
            parameters['chunk_size'] = int(np.ceil(727_627/3))
    elif parameters['model'] == 'bert':
        if parameters['num_clients'] == 3:
            parameters['chunk_size'] = int(np.ceil(4_386_179/3))
        elif parameters['num_clients'] == 5:
            parameters['chunk_size'] = int(np.ceil(4_386_179/8))
        elif parameters['num_clients'] == 7:
            parameters['chunk_size'] = int(np.ceil(4_386_179/16))

"""Teste samme chunk-size for CKKS-NF på MNIST.
Se om det er noe ekstra overhead."""
SAMME_CHUNK_SIZE = True
if parameters['scheme'] == 'CKKS-NF' and SAMME_CHUNK_SIZE:
    if parameters['model'] in ['logreg']:
        parameters['chunk_size'] = 10*131072
    elif parameters['model'] == 'squeezenet':
        if parameters['num_clients'] == 3:
            parameters['chunk_size'] = int(np.ceil(727_627/3))
        elif parameters['num_clients'] == 5:
            parameters['chunk_size'] = int(np.ceil(727_627/10))
        elif parameters['num_clients'] == 7:
            parameters['chunk_size'] = int(np.ceil(727_627/21))

BGV_UTEN_POLYNOMIAL_ENCODING = False # Har allerede kjørt den!
if BGV_UTEN_POLYNOMIAL_ENCODING:
    if parameters['model'] in ['logreg']:
        parameters['chunk_size'] = 10*131072
    elif parameters['model'] == 'squeezenet':
        if parameters['num_clients'] == 3:
            parameters['chunk_size'] = int(np.ceil(727_627))
        elif parameters['num_clients'] == 5:
            parameters['chunk_size'] = int(np.ceil(727_627))
        elif parameters['num_clients'] == 7:
            parameters['chunk_size'] = int(np.ceil(727_627/2))


# Prometheus queries.
"""
Units:
cpu_query: Percentage of maximum CPU resources? (Er en prosent ihvertfall)
ram_query: Bytes
cache_query: Bytes
recv_net_query: Bytes / second
sent_net_query: Bytes / second
"""
single_value_request = 'http://localhost:9090/api/v1/query?query='
multi_value_request = 'http://localhost:9090/api/v1/query_range?query='
done_query = 'done_counter_total'
acc_query = 'model_accuracy{name!=""}'
loss_query = 'model_loss{name!=""}'
f1_query = 'model_f1{name!=""}'
cpu_query = 'sum(rate(container_cpu_usage_seconds_total{name!="",name!~"(prometheus|cadvisor|grafana|)"}[10s]))by(name)'
ram_query = 'sum(container_memory_rss{name!="",name!~"(prometheus|cadvisor|grafana|)"})by(name)'
cache_query = 'sum(container_memory_cache{name!="",name!~"(prometheus|cadvisor|grafana|)"})by(name)'
recv_net_query = 'sum(rate(container_network_receive_bytes_total{name!="",name!~"(prometheus|cadvisor|grafana|)"}[10s]))by(name)'
sent_net_query = 'sum(rate(container_network_transmit_bytes_total{name!="",name!~"(prometheus|cadvisor|grafana|)"}[10s]))by(name)'

"""Set up environment so that flwr run can actually be run."""
env = os.environ.copy()
env["PATH"] = env["PATH"] + os.pathsep + "/home/krist/thesis_code_v2/.venv/bin"

# Run to check:
result = subprocess.run(
            ["env"],  # 'env' prints all environment variables
            env=env,
            capture_output=True,  # Capture stdout and stderr
            text=True,            # Decode output as text
            check=True            # Raise CalledProcessError if command fails
        )
# print("Output:\n", result.stdout)

print(parameters)

# For å sikre at stasjonær blir ferdig først.
time.sleep(5)

# Følgende punkter skal stå inne i en for-loop som kjører parameters["num_tests"] tester.

for test in range(parameters["num_tests"]):

    print(f"Runde {test+1}/{parameters["num_tests"]}:")

    """Starte containere + flwr run. Må be stasjonær pc om å starte containere også?"""

    print("Skrur på egne containere.")
    args = f'docker compose -f dockerfiles/single-comp-{parameters["num_clients"]}c.yaml up'
    containers = subprocess.Popen(args, shell=True, start_new_session=True)

    # Vent litt for å sikre at server har startet før flwr run kjøres.
    time.sleep(10)

    """Ta tiden herfra, så blir det likt for alle."""
    starting_time = str(int(time.time()))

    time.sleep(3)
    args1 = f"flwr run fhefedavg/ "
    args2 = f"-c 'model=\"{parameters['model']}\""
    if parameters["program"] == "fhefedavg":
        # if parameters['scheme'] == "CKKS-NF":
        #     args1 = f"flwr run {parameters["model"]}/{parameters["model"]}-ckks-nf/ "
        args2 += f" sec-type=\"fhefedavg\" scheme=\"{parameters["scheme"]}\" num-clients={parameters["num_clients"]} "
        args2 += f"security-level=\"{parameters['sec_level']}\" ring-dim={parameters['ring_dim']} "
        if parameters['scheme'] != "CKKS":
            args2 += f"num-coeffs={parameters['num_coeffs']} max-deg-int={parameters['max_deg_int']} "
        if parameters['scheme'] == "BFV":
            args2 += f"mult-tech=\"{parameters['mult_tech']}\" "
        if parameters['scheme'] in ["CKKS", "CKKS-NF"]:
            args2 += f"delta-bits=\"{parameters['delta_bits']}\" "
        if parameters['scheme'] == "CKKS-NF":
            args2 += f"noise-flooding=true "
        if BGV_UTEN_POLYNOMIAL_ENCODING:
            args2 += f"no-polynomial-encoding=true "
        args2 += f"chunk-size={parameters['chunk_size']}' "
    elif parameters['program'] == 'secagg':
        args2 += f" max-weight={parameters['max_weight']}' "
    else:
        args2 += f"' "
    args = args1+args2
    print(f"Command to launch flwr run:\n{args}")
    proc = subprocess.run(args=args, shell=True, env=env)

    """Vente til containere er ferdig, ala benchmark_fhefedavg.py."""

    print("Flower run has been launched.")


    # Logreg går såppass raskt at det ikke er vits å vente like lenge.
    if parameters['model'] == "logreg":
        wait = 1
    else:
        wait = 10

    time.sleep(wait+5)

    # Wait until run is done, which is indicated by the server setting 
    # done_counter to 1.
    DONE = False
    while not DONE:
        response = requests.get(single_value_request+done_query)
        content = literal_eval(response.content.decode())
        # contents['data']['result'][0]['value'][1] contains 
        # the value of the done_counter as a string.
        # If this is 1, then all rounds of training and evaluation are finished.
        # Sometimes, the reply randomly changes from the correct structure
        # to one where 'result' is an empty list,
        # despite the query still being a success.
        # I do not know why this is the case.
        # Hopefully, it fixes itself.
        if len(content['data']['result']) == 0:
            time.sleep(wait)
        elif int(content['data']['result'][0]['value'][1]) == 1:
            DONE = True
        else:
            time.sleep(wait)

    """Stopp tidtakning her."""
    time.sleep(3)
    ending_time = str(int(time.time()))
    times = (starting_time, ending_time)

    """Skrive samlede metrics til csv filer, ala benchmark_fhefedavg."""
    # Hent egne metrics.
    # Create timestamp string for range queries.
    time_query = f'&start={times[0]}&end={times[1]}&step=1s'
    if parameters['model'] != 'logreg' and parameters['num_clients'] == 7:
        # Prometheus støtter ikke mer enn 11_000 datapunkter i en time series.
        # Squeezenet og BERT bruker lenger tid enn det for 7 clients med schemes,
        # så må samle inn annenhvert sekund istedenfor.
        time_query = f'&start={times[0]}&end={times[1]}&step=2s'
    if parameters['model'] == 'bert' and parameters['num_clients'] == 5:
        # Prometheus støtter ikke mer enn 11_000 datapunkter i en time series.
        # BERT 5 clients bruker lenger tid enn det.
        time_query = f'&start={times[0]}&end={times[1]}&step=2s'
    if parameters['model'] == 'bert' and parameters['num_clients'] == 7:
        # Prometheus støtter ikke mer enn 11_000 datapunkter i en time series.
        # BERT 7 clients bruker mer enn 6 timer, så må samle inn hvert tredje sekund.
        time_query = f'&start={times[0]}&end={times[1]}&step=3s'
    # GET all metrics and convert them to python dicts.
    metrics_dict = {}
    metrics_dict['accuracy'] = request_to_dict(single_value_request,
                                               acc_query)
    metrics_dict['loss'] = request_to_dict(single_value_request,
                                           loss_query)
    if parameters['model'] == 'bert':
        metrics_dict['f1'] = request_to_dict(single_value_request,
                                             f1_query)
    metrics_dict['cpu'] = request_to_dict(multi_value_request,
                                          cpu_query,
                                          time_query)
    metrics_dict['memory'] = request_to_dict(multi_value_request,
                                             ram_query,
                                             time_query)
    metrics_dict['cache'] = request_to_dict(multi_value_request,
                                            cache_query,
                                            time_query)
    metrics_dict['network_recv'] = request_to_dict(multi_value_request,
                                                   recv_net_query,
                                                   time_query)
    metrics_dict['network_sent'] = request_to_dict(multi_value_request,
                                                   sent_net_query,
                                                   time_query)

    """Må lese inn kjøretider for training og evaluation fra
        storage/key_store/node{i}/{model}-{program}-training-time/run{j}/time.txt
    og
        storage/key_store/node{i}/{model}-{program}-evaluation-time/run{j}/time.txt
    for alle i og j.
    
    Lag dict med nøkler per client og mean over runs."""
    ml_times_per_round = []
    if parameters["model"] == "logreg" or parameters['num_clients'] in [3,5, 7]:
        if parameters['num_clients'] == 3:
            ids = [0]
        elif parameters['num_clients'] == 5:
            ids = [0,1]
        elif parameters['num_clients'] == 7:
            ids = [0,1,2]
    else:
        if parameters['num_clients'] == 3:
            ids = [0,1]
        elif parameters['num_clients'] == 5:
            ids = [0,1,2]
        elif parameters['num_clients'] == 7:
            ids = [0,1,2,3]
    for i in ids:
        """Each time file has format:
        <time round 1>
        <time round 2>
        <time round 3>"""
        round_times = []
        for round in range(1,4):
            training_path = f"./storage/key_store/node{i}/training-time/run{round}/time.txt"
            evaluation_path = f"./storage/key_store/node{i}/evaluation-time/run{round}/time.txt"
            with open(training_path, "r") as infile:
                line = infile.readline()
                round_times.append(float(line))
            with open(evaluation_path, "r") as infile:
                line = infile.readline()
                round_times[len(round_times)-1] += float(line)
        ml_times_per_round.append(round_times)
    
    laptop_ml_mean_times = np.mean(ml_times_per_round, axis=0)

    """Da gjennstår det egentlig bare å skrive verdier til filer."""
    runtime = int(ending_time) - int(starting_time)
    num_dirs = dict_to_csv(metrics_dict, parameters)

    if parameters['scheme'] == "CKKS-NF" and parameters['program'] == "fhefedavg":
        noise_times = []
        keygen_times = []
        dec_times = []
        block_times = []
        for round in range(1,4):
            noise_path = f"./storage/key_store/server/noise-estimate-time/run{round}/time.txt"
            with open(noise_path, "r") as infile:
                line = infile.readline()
            noise_times.append(float(line))
            keygen_path = f"./storage/key_store/server/keygen-time/run{round}/time.txt"
            with open(keygen_path, "r") as infile:
                line = infile.readline()
            keygen_times.append(float(line))
            decryption_path = f"./storage/key_store/server/decryption-time/run{round}/time.txt"
            with open(decryption_path, "r") as infile:
                lines = infile.readlines()
                lines = [float(line) for line in lines]
            dec_times.append(sum(lines))
            block_path = f"./storage/key_store/server/block-time/run{round}/time.txt"
            with open(block_path, "r") as infile:
                line = infile.readline()
            block_times.append(float(line))
        runtime_to_csv_single_pc(num_dirs, 
                                 parameters, 
                                 (runtime, laptop_ml_mean_times, noise_times, keygen_times, dec_times, block_times))
    elif parameters['program'] == "fhefedavg":
        keygen_times = []
        dec_times = []
        block_times = []
        for round in range(1,4):
            keygen_path = f"./storage/key_store/server/keygen-time/run{round}/time.txt"
            with open(keygen_path, "r") as infile:
                line = infile.readline()
            keygen_times.append(float(line))
            decryption_path = f"./storage/key_store/server/decryption-time/run{round}/time.txt"
            with open(decryption_path, "r") as infile:
                lines = infile.readlines()
                lines = [float(line) for line in lines]
            dec_times.append(sum(lines))
            block_path = f"./storage/key_store/server/block-time/run{round}/time.txt"
            with open(block_path, "r") as infile:
                line = infile.readline()
            block_times.append(float(line))
        runtime_to_csv_single_pc(num_dirs, 
                                 parameters, 
                                 (runtime, laptop_ml_mean_times, keygen_times, dec_times, block_times))
    else:
        runtime_to_csv_single_pc(num_dirs,
                                 parameters, 
                                 (runtime, laptop_ml_mean_times))
    
    """Slett filer som ble brukt til å lagre kjøretider, hvis de eksisterer."""
    if os.path.isdir(f"./storage/key_store/server/keygen-time/"):
        shutil.rmtree(f"./storage/key_store/server/keygen-time/")
    if os.path.isdir(f"./storage/key_store/server/block-time/"):
        shutil.rmtree(f"./storage/key_store/server/block-time/")
    if os.path.isdir(f"./storage/key_store/server/decryption-time/"):
        shutil.rmtree(f"./storage/key_store/server/decryption-time/")
    if os.path.isdir(f"./storage/key_store/server/noise-estimate-time/"):
        shutil.rmtree(f"./storage/key_store/server/noise-estimate-time/")
    if os.path.isdir(f"./storage/key_store/node0/training-time/"):
        shutil.rmtree(f"./storage/key_store/node0/training-time/")
    if os.path.isdir(f"./storage/key_store/node0/evaluation-time/"):
        shutil.rmtree(f"./storage/key_store/node0/evaluation-time/")
    if os.path.isdir(f"./storage/key_store/node1/training-time/"):
        shutil.rmtree(f"./storage/key_store/node1/training-time/")
    if os.path.isdir(f"./storage/key_store/node1/evaluation-time/"):
        shutil.rmtree(f"./storage/key_store/node1/evaluation-time/")
    if os.path.isdir(f"./storage/key_store/node2/training-time/"):
        shutil.rmtree(f"./storage/key_store/node2/training-time/")
    if os.path.isdir(f"./storage/key_store/node2/evaluation-time/"):
        shutil.rmtree(f"./storage/key_store/node2/evaluation-time/")
    if os.path.isdir(f"./storage/key_store/node3/training-time/"):
        shutil.rmtree(f"./storage/key_store/node3/training-time/")
    if os.path.isdir(f"./storage/key_store/node3/evaluation-time/"):
        shutil.rmtree(f"./storage/key_store/node3/evaluation-time/")
        
    move_models(parameters, num_dirs)
   
    print("Skrur av egne containere.")
    args = f'docker compose -f dockerfiles/single-comp-{parameters["num_clients"]}c.yaml down'
    down = subprocess.run(args=args, shell=True, start_new_session=True)

    # For logreg ser det ut til å holde å sleepe 3 min,
    # men for bert og squeezenet har jeg tidligere erfart at jeg må vente 15.
    if parameters['model'] == "logreg":
        time.sleep(300)
    else:
        time.sleep(900)

    gc.collect()

print("All benchmarking ferdig!")

