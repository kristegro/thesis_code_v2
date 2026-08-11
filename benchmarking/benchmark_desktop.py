import multiprocessing.connection as mpc
import subprocess
from pathlib import Path
import os
import shutil
import gc

from prometheus import request_to_csv, request_to_dict
import numpy as np

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

"""Motta parametre fra bærbar."""

# Bruke stasjonær sin ip.
listener = mpc.Listener(("192.168.68.102", 12400))
print("Venter på parametre fra bærbar...")
connection = listener.accept()
parameters = connection.recv()
connection.close()

print("Har mottatt parameters fra bærbar.")
print(parameters)

# Følgende punkter skal stå inne i en for-loop som kjører X tester.

for test in range(parameters["num_tests"]):

    print(f"Runde {test+1}/{parameters["num_tests"]}:")

    """Starte containere. Må få beskjed fra bærbar?"""
    connection = listener.accept()
    connection.close()
    print("Mottatt beskjed om å skru på containere. Gjør det.")
    print("Skrur på egne containere.")
    if parameters["model"] == "logreg" or parameters['num_clients'] in [3,5, 7]:
        args = f'docker compose -f dockerfiles/logreg-desktop-{parameters["num_clients"]}c.yaml up'
    else:
        args = f'docker compose -f dockerfiles/desktop-{parameters["num_clients"]}c.yaml up'
    containers = subprocess.Popen(args, shell=True, start_new_session=True)

    """Få beskjed om å overlevere metrics mellom start- og sluttidspunkter.
    Skru av containere etter at de er hentet ut, men før du svarer."""

    connection = listener.accept()
    times = connection.recv()
    print("Mottatt tider for sample metrics.")

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
        # BERT bruker lenger tid enn det for 5 clients med schemes,
        # så må samle inn annenhvert sekund istedenfor.
        time_query = f'&start={times[0]}&end={times[1]}&step=2s'
    if parameters['model'] == 'bert' and parameters['num_clients'] == 7:
        # Prometheus støtter ikke mer enn 11_000 datapunkter i en time series.
        # BERT bruker lenger tid enn det for 5 clients med schemes,
        # så må samle inn annenhvert sekund istedenfor.
        time_query = f'&start={times[0]}&end={times[1]}&step=3s'

    """Stasjonær pc må hente ut metrics som python objekter
    istedenfor å skrive de rett til fil.
    Enkleste er vel å hente ut som dictionaries og lage
    et overordnet dictionary som så sendes tilbake."""

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
            ids = [0,1]
        elif parameters['num_clients'] == 5:
            ids = [0,1,2]
        elif parameters['num_clients'] == 7:
            ids = [0,1,2,3]
    else:
        if parameters['num_clients'] == 3:
            ids = [0]
        elif parameters['num_clients'] == 5:
            ids = [0,1]
        elif parameters['num_clients'] == 7:
            ids = [0,1,2]
    for i in ids:
        """Each time file has format:
        <time round 1>
        <time round 2>
        <time round 3>"""
        round_times = []
        for round in range(1,4):
            training_path = f"./storage/key_store/node{i}/{parameters['model']}-{parameters['program']}-training-time/run{round}/time.txt"
            evaluation_path = f"./storage/key_store/node{i}/{parameters['model']}-{parameters['program']}-evaluation-time/run{round}/time.txt"
            with open(training_path, "r") as infile:
                line = infile.readline()
                round_times.append(float(line))
            with open(evaluation_path, "r") as infile:
                line = infile.readline()
                round_times[len(round_times)-1] += float(line)
        ml_times_per_round.append(round_times)
    
    """Slett filer som ble brukt til å lagre kjøretider, hvis de eksisterer."""
    if os.path.isdir(f"./storage/key_store/node0/{parameters['model']}-{parameters['program']}-training-time/"):
        shutil.rmtree(f"./storage/key_store/node0/{parameters['model']}-{parameters['program']}-training-time/")
    if os.path.isdir(f"./storage/key_store/node0/{parameters['model']}-{parameters['program']}-evaluation-time/"):
        shutil.rmtree(f"./storage/key_store/node0/{parameters['model']}-{parameters['program']}-evaluation-time/")
    if os.path.isdir(f"./storage/key_store/node1/{parameters['model']}-{parameters['program']}-training-time/"):
        shutil.rmtree(f"./storage/key_store/node1/{parameters['model']}-{parameters['program']}-training-time/")
    if os.path.isdir(f"./storage/key_store/node1/{parameters['model']}-{parameters['program']}-evaluation-time/"):
        shutil.rmtree(f"./storage/key_store/node1/{parameters['model']}-{parameters['program']}-evaluation-time/")
    if os.path.isdir(f"./storage/key_store/node2/{parameters['model']}-{parameters['program']}-training-time/"):
        shutil.rmtree(f"./storage/key_store/node2/{parameters['model']}-{parameters['program']}-training-time/")
    if os.path.isdir(f"./storage/key_store/node2/{parameters['model']}-{parameters['program']}-evaluation-time/"):
        shutil.rmtree(f"./storage/key_store/node2/{parameters['model']}-{parameters['program']}-evaluation-time/")
    if os.path.isdir(f"./storage/key_store/node3/{parameters['model']}-{parameters['program']}-training-time/"):
        shutil.rmtree(f"./storage/key_store/node3/{parameters['model']}-{parameters['program']}-training-time/")
    if os.path.isdir(f"./storage/key_store/node3/{parameters['model']}-{parameters['program']}-evaluation-time/"):
        shutil.rmtree(f"./storage/key_store/node3/{parameters['model']}-{parameters['program']}-evaluation-time/")

    mean_per_round = np.mean(ml_times_per_round, axis=0)
    # Send metrics tilbake.
    connection.send((metrics_dict, mean_per_round))

    connection.close()
    print("Skrur av egne containere.")
    if parameters["model"] == "logreg" or parameters['num_clients'] in [3,5, 7]:
        args = f'docker compose -f dockerfiles/logreg-desktop-{parameters["num_clients"]}c.yaml down'
    else:
        args = f'docker compose -f dockerfiles/desktop-{parameters["num_clients"]}c.yaml down'
    down = subprocess.run(args=args, shell=True, start_new_session=True)

    # Garbage collection.
    gc.collect()

print("All benchmarking ferdig!")

