import time
from ast import literal_eval
import csv
import pprint
import requests
from pathlib import Path
import os
import shutil

import numpy as np

def content_to_csv(content, filename):
    """Take as input a dict resulting from ast.literal_eval(response.content.decode()),
    i.e., the response to a GET query to Prometheus. Parse this dict and
    write the values to a file with name filename.
    
    Args:
        content (dict): A nested dict resulting from ast.literal_eval(response.content.decode())
            for Prometheus' response to a GET request.
        filename (str): The filename to use for the csv file."""
    
    # Storage for parsed values.
    parsed_dict = {}

    resultType = content['data']['resultType']
    result = content['data']['result']

    if resultType == 'matrix':
        for container_dict in result:
            """Each container_dict is a dictionary with keys
            'metric' and 'values'.
            'metric' is a dict which contains information about the metric, 
                such as container name.
            'values' is a list of one or more pairs of [timestamp, "value"],
                where timestamp is a UNIX timestamp and "value"
                is a string containing the numerical value of the metric at 
                that timestamp.
            'values' contains at least two elements since resultType is a matrix."""
            # container_dict['metric']['name'] gives the container name.
            name = container_dict['metric']['name']
            # container_dict['values'][i][0] is the timestamp for the value,
            # which I do not think I need.
            # container_dict['values'][i][1] is the actual value.
            values = [container_dict['values'][i][1] for i in range(len(container_dict['values']))]
            parsed_dict[name] = values
    else:
        # resultType == 'vector'.
        for container_dict in result:
            """Each container_dict is a dictionary with keys
            'metric' and 'value'.
            'metric' is a dict which contains information about the metric, 
                such as container name.
            'value' is a list with form [timestamp, "value"],
                where timestamp is a UNIX timestamp and "value"
                is a string containing the numerical value of the metric at 
                that timestamp."""
            # container_dict['metric']['name'] gives the container name.
            name = container_dict['metric']['name']
            # container_dict['value'][0] is the timestamp for the value,
            # which I do not think I need.
            # container_dict['value'][1] is the actual value.
            value = [container_dict['value'][1]]
            parsed_dict[name] = value


    """parsed_dict now contains key-value pairs of the form:
        (container name, values).
    These can now be written to a csv file with the appropriate format."""
    """Tiltenkt format:
    <tittel>
    <query som ga resultat>
    <container names>
    <verdier 
        ...
    >
    
    <max>
    <container names>
    <verdier 
        ...
    >

    <mean>
    <container names>
    <verdier 
        ...
    >

    <std>
    <container names>
    <verdier 
        ...
    >
    """
    with open(filename, "a", newline='') as csvfile:
        csvwriter = csv.writer(csvfile,
                                quotechar='"', 
                                quoting=csv.QUOTE_MINIMAL)
        # The keys are the container names.
        names = list(parsed_dict.keys())
        # First row of csv should be the container names.
        csvwriter.writerow(names)
        # All the values in parsed_dict will have the same lenght.
        length = len(parsed_dict[names[0]])
        # Subsequent rows should be the values at each timestamp.
        # Each row contains the values for all names at the same timestamp.
        try:
            _ = [[parsed_dict[name][i] for name in names] for i in range(length)]
        except IndexError:
            max_length = np.max([len(parsed_dict[n]) for n in names])
            for name in names:
                parsed_dict[name] += ['-' for _ in range(max_length-len(parsed_dict[name]))]
        rows = [[parsed_dict[name][i] for name in names] for i in range(length)]
        csvwriter.writerows(rows)


def request_to_csv(request, query, timestamps="", title="", filename="test.csv", conversion=str, computation=False):
    """Wrap 'content_to_csv' to go from request to csv file instead.
    
    Args:
        request (str): The part of the GET request which specifies which API to use.
        query (str): The query used to get content. Save this in the file as well.
        timestamps (str): The start and end timestamps when requesting a range vector.
        title (str): The title to write on the first row.
        filename (str): The filename to use for the csv file.
        conversion (type): The type to convert elements to elements to.
        computation (bool): Whether or not to compute max, mean and std of the 
            time series for each container."""
    
    response = requests.get(request+query+timestamps)
    content = literal_eval(response.content.decode())

    content_to_csv(content, title, query, filename, conversion, computation)

def request_to_dict(request, query, timestamps=""):
    """Query prometheus to get the results of requests.get(request+query+timestamps),
    and convert to dict.

    Args:
        request (str): The part of the GET request which specifies which API to use.
        query (str): The query used to get content. Save this in the file as well.
        timestamps (str): The start and end timestamps when requesting a range vector.
    
    Return: The result of the request formatted as a python dict.
    """
    response = requests.get(request+query+timestamps)
    content = literal_eval(response.content.decode())
    return content

def construct_path(parameters):
    """Construct the path based on parameters dict."""
    base_dir = "./benchmarking_results/"
    program_dir = f"{parameters['model']}-{parameters['program']}/"
    if parameters['program'] == "fhefedavg":
        scheme_dir = f"{parameters['scheme']}/"
        """details_dir gir navn utifra parametre, eks antall clients og ring dimensjon,
        men jeg må se på hvilke parametre jeg faktisk skal se på."""
        if parameters["scheme"] == "BGV":
            details_dir = (f"nc{parameters['num_clients']}_"
                        f"sl{parameters['sec_level']}_"
                        f"rd{parameters['ring_dim']}_"
                        f"pe{parameters['num_coeffs']}_"
                        f"cs{parameters['chunk_size']}/")
        elif parameters["scheme"] == "BFV":
            details_dir = (f"nc{parameters['num_clients']}_"
                        f"sl{parameters['sec_level']}_"
                        f"rd{parameters['ring_dim']}_"
                        f"pe{parameters['num_coeffs']}_"
                        f"cs{parameters['chunk_size']}_"
                        f"mt{parameters['mult_tech']}/")
        elif parameters["scheme"] in ["CKKS", "CKKS-NF"]:
            details_dir = (f"nc{parameters['num_clients']}_"
                        f"sl{parameters['sec_level']}_"
                        f"rd{parameters['ring_dim']}_"
                        f"db{parameters['delta_bits']}_"
                        f"cs{parameters['chunk_size']}/")
    else:
        scheme_dir = ""
        details_dir = f"nc{parameters['num_clients']}/"
    return base_dir+program_dir+scheme_dir+details_dir

def dict_to_csv(dict, parameters):
    """Wrap content_to_csv() to take a dict containing multiple contents and writing it to files.
    Format filenames and such using the contents of parameters dict.

    Args:
        dict (dict): A dict contaning the results of multiple Prometheus queries.
        parameters (dict): A dict contaning information on the benchmark run.

    Returns:
        An integer indicating which round was processed.
    """

    path = construct_path(parameters)

    """current_path eksemepel:
    ./benchmarking_results/imdb-fhefedavg/CKKS/nc3/
    for imdb-fhefedavg kjørt med CKKS og 3 clients."""
    # Lag path hvis den ikke eksisterer.
    current_path = Path(path)
    Path.mkdir(current_path, 
               parents=True, 
               exist_ok=True)

    """Må nå finne ut hvor mange runs som er gjort hittil."""
    num_dirs = len(list(current_path.iterdir()))

    """Mappe for run skal ha navn 'run{num_dirs+1}',
    altså første run for navn 'run1' osv."""
    run_dir = f"run{num_dirs+1}/"
    Path.mkdir(current_path/run_dir)

    for key in list(dict.keys()):
        # Call content_to_csv for each.
        content_to_csv(dict[key],
                       path+run_dir+f"{key}.csv")
        
    return num_dirs+1

def runtime_to_csv(num_dirs, parameters, times):
    # Pakk ut tider.
    if parameters['scheme'] == "CKKS-NF" and parameters['program'] == "fhefedavg":
        total_time, desktop_ml_mean_times, laptop_ml_mean_times, noise_times, keygen_times, dec_times, block_times = times
        desktop_ml_mean_times = desktop_ml_mean_times.tolist()
        laptop_ml_mean_times = laptop_ml_mean_times.tolist()
        # Tid som ikke brukes på ML oppgaver.
        other_time = float(total_time - np.max([np.sum(desktop_ml_mean_times), 
                                                np.sum(laptop_ml_mean_times)])
                                      - np.sum(keygen_times)
                                      - np.sum(block_times))
        # Tid som ble brukt på oppdeling/gjennoppbygging, men ikke decryption.
        block_no_deck_times = [block_times[i] - dec_times[i] for i in range(len(block_times))]
    elif parameters['program'] == "fhefedavg":
        total_time, desktop_ml_mean_times, laptop_ml_mean_times, keygen_times, dec_times, block_times = times
        desktop_ml_mean_times = desktop_ml_mean_times.tolist()
        laptop_ml_mean_times = laptop_ml_mean_times.tolist()
        # Tid som ikke brukes på ML oppgaver.
        other_time = float(total_time - np.max([np.sum(desktop_ml_mean_times), 
                                                np.sum(laptop_ml_mean_times)])
                                      - np.sum(keygen_times)
                                      - np.sum(block_times))
        # Tid som ble brukt på oppdeling/gjennoppbygging, men ikke decryption.
        block_no_deck_times = [block_times[i] - dec_times[i] for i in range(len(block_times))]
    else:
        total_time, desktop_ml_mean_times, laptop_ml_mean_times = times
        desktop_ml_mean_times = desktop_ml_mean_times.tolist()
        laptop_ml_mean_times = laptop_ml_mean_times.tolist()

        other_time = float(total_time - np.max([np.sum(desktop_ml_mean_times), 
                                          np.sum(laptop_ml_mean_times)]))

        
    path = construct_path(parameters)
    

    """current_path eksemepel:
    ./benchmarking_results/imdb-fhefedavg/CKKS/nc3/
    for imdb-fhefedavg kjørt med CKKS og 3 clients."""
    # Lag path hvis den ikke eksisterer.
    current_path = Path(path)
    Path.mkdir(current_path, 
               parents=True, 
               exist_ok=True)

    """Mappe for run skal ha navn 'run{num_dirs+1}',
    altså første run for navn 'run1' osv."""
    run_dir = f"run{num_dirs}/"
    Path.mkdir(Path(path+run_dir), exist_ok=True)

    if parameters['scheme'] == "CKKS-NF" and parameters['program'] == "fhefedavg":
        with open(path+run_dir+f"runtime.csv", "a", newline='') as csvfile:
            csvwriter = csv.writer(csvfile,
                                    quotechar='"', 
                                    quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(["runtime", 
                                "ml-desktop-r1",
                                "ml-desktop-r2",
                                "ml-desktop-r3",
                                "ml-laptop-r1",
                                "ml-laptop-r2",
                                "ml-laptop-r3",
                                "noise-r1",
                                "noise-r2",
                                "noise-r3",
                                "keygen-r1",
                                "keygen-r2",
                                "keygen-r3",
                                "dec-r1",
                                "dec-r2",
                                "dec-r3",
                                "block-no-dec-r1",
                                "block-no-dec-r2",
                                "block-no-dec-r3",
                                "block-r1",
                                "block-r2",
                                "block-r3",
                                "other"])
            csvwriter.writerow(([total_time] + desktop_ml_mean_times + laptop_ml_mean_times 
                                + noise_times + keygen_times + dec_times + block_no_deck_times 
                                + block_times + [other_time]))
    elif parameters['program'] == "fhefedavg":
        with open(path+run_dir+f"runtime.csv", "a", newline='') as csvfile:
            csvwriter = csv.writer(csvfile,
                                    quotechar='"', 
                                    quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(["runtime", 
                                "ml-desktop-r1",
                                "ml-desktop-r2",
                                "ml-desktop-r3",
                                "ml-laptop-r1",
                                "ml-laptop-r2",
                                "ml-laptop-r3",
                                "keygen-r1",
                                "keygen-r2",
                                "keygen-r3",
                                "dec-r1",
                                "dec-r2",
                                "dec-r3",
                                "block-no-dec-r1",
                                "block-no-dec-r2",
                                "block-no-dec-r3",
                                "block-r1",
                                "block-r2",
                                "block-r3",
                                "other"])
            # from pprint import pprint
            # pprint(desktop_ml_mean_times)
            # pprint(laptop_ml_mean_times)
            # pprint(keygen_times)
            # pprint(dec_times)
            # pprint(block_times)
            # pprint(other_time)
            # pprint(([total_time] + desktop_ml_mean_times + laptop_ml_mean_times 
            #         + keygen_times + dec_times + block_no_deck_times 
            #         + block_times + [other_time]))
            csvwriter.writerow(([total_time] + desktop_ml_mean_times + laptop_ml_mean_times 
                                + keygen_times + dec_times + block_no_deck_times 
                                + block_times + [other_time]))
    else:
        with open(path+run_dir+f"runtime.csv", "a", newline='') as csvfile:
            csvwriter = csv.writer(csvfile,
                                    quotechar='"', 
                                    quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(["runtime", 
                                "ml-desktop-r1",
                                "ml-desktop-r2",
                                "ml-desktop-r3",
                                "ml-laptop-r1",
                                "ml-laptop-r2",
                                "ml-laptop-r3",
                                "other"])
            csvwriter.writerow([total_time] + desktop_ml_mean_times + laptop_ml_mean_times + [other_time])

def move_models(parameters, test_run, num_rounds = 3):
    """Move stored models from storage/key_store/server
    to the correct run folder in benchmarking_results.
    Delete files in storage/key_store/server afterwards."""
    path = construct_path(parameters)
    run_dir = f"run{test_run}/"
    path = Path(path+run_dir)

    if parameters['model'] == "imdb":
        # BERT is saved by transformers as a folder.
        for round in range(1, num_rounds+1):
            if os.path.isdir(f"./storage/key_store/server/{parameters['model']}-tmp-r{round}"):
                shutil.move(f"./storage/key_store/server/{parameters['model']}-tmp-r{round}",
                            path/f'{parameters['model']}-tmp-r{round}')
    else:
        # Both squeezenet and logreg are saved by pytorch as files.
        for round in range(1, num_rounds+1):
            if os.path.exists(f"./storage/key_store/server/{parameters['model']}-tmp-r{round}"):
                shutil.move(f"./storage/key_store/server/{parameters['model']}-tmp-r{round}",
                            path/f'{parameters['model']}-tmp-r{round}')
        

if __name__ == "__main__":

    """Hent ut verdi på et tidspunkt."""
    single_value_request = 'http://localhost:9090/api/v1/query?query='
    ram_per_container = 'sum(container_memory_rss{name!="",name!~"(prometheus|cadvisor|grafana|)"})by(name)'
    response = requests.get(single_value_request+ram_per_container)
    # print(response.content)
    content = literal_eval(response.content.decode())
    # content_to_csv(content, "testing_single_ram_value.csv")


    """Måten man henter ut informasjon fra prometheus på er følgende:
    Send GET request til 'localhost:9090/api/v1/query?query=<QUERY>'
    hvor <QUERY> må byttes ut med et gyldig prometheus query.

    Se her: https://prometheus.io/docs/prometheus/latest/querying/api/

    For å hente ut mer enn et tall kan man bruke Range query,
    fra '/api/v1/query_range' (med localhost foran som over.)

    Alle queries kan sendes gjennom python ved å bruke
    response = requests.get("http://localhost:9090/api/v1/...")
    og innholdet i svaret hentes ut med response.content.

    requests.get('http://localhost:9090/api/v1/query?query=sum(container_memory_rss{instance=~".*",name=~".*",name=~".+",name!~"(prometheus|cadvisor|grafana)"})by(name)')
    gir en tom response, selv om det er denne Grafana bruker for å produsere grafer.
    Jeg regner med at man må spesifisere et timestamp også, ikke bare spørre etter en verdi.
    Jeg er egentlig ikke sikker? Ifølge dokumentasjon er time en optional parameter.


    Some eksempel, studer følgende query:
        sum(container_memory_rss{instance=~".*",name=~".*",name=~".+",name!~"(prometheus|cadvisor|grafana)"})by(name)

    - sum: Summer over alle verdier
        by(name): Summer verdier for hver 'name' label. 'name' er dermed dimensjonen/aksen man summerer over. 
            En sum per 'name' label.
    - container_memory_rss: Navnet på metricen som letes etter 
        {} brukes for å kun velge spesifikke instanser av container_memory_rss
    - instance=~".*": Ta med alle som har 'instance' label som matcher Regex uttrykk ".*"
    - name=~".*": Ta med alle som har 'name' label som matcher Regex uttrykk ".*"
        ".*": Matcher alle mulige strenger, INKLUDERT tomme strenger.
    - name=~".+": Ta med alle som har 'name' label som matcher Regex uttrykk ".+"
        ".+": Matcher alle mulige strenger, UNNTATT tomme strenger.
    - name!~"(prometheus|cadvisor|grafana)": Fjern alle som har 'name' label som 
        matcher Regex uttrykk "(prometheus|cadvisor|grafana)"

    Er ikke 'name=~".*",name=~".+"' unødvendig? Man kan altså ha med alle mulige navn, inkludert tomme navn,
    samtidig som man kan ha med alle mulige navn eksludert tomme navn?
    For øvrig, er ikke 'instance=~".*"' også unødvendig? Siden den sier at man skal ta med alle instances, inkludert tomme,
    men det er vel det som gjøres i utgangspunktet hvis man ikke filterer vek noe?

    Så da er vel følgende egentlig ekvivalent?
    sum(container_memory_rss{name!~"(prometheus|cadvisor|grafana)"})by(name)

    Får da
    {"status":"success",
    "data":
        {"resultType":"vector",
        "result":
            [{"metric":{},
            "value":[1765553241.136,"11799130112"]},
            {"metric":{"name":"dockerfiles-serverapp-1"},
            "value":[1765553241.136,"46493696"]},
            {"metric":{"name":"dockerfiles-supernode-0-1"},
            "value":[1765553241.136,"46493696"]},
            {"metric":{"name":"dockerfiles-supernode-2-1"},
            "value":[1765553241.136,"46493696"]},
            {"metric":{"name":"dockerfiles-clientapp-0-1"},
            "value":[1765553241.136,"48717824"]},
            {"metric":{"name":"dockerfiles-supernode-1-1"},
            "value":[1765553241.136,"46489600"]},
            {"metric":{"name":"dockerfiles-superlink-1"},
            "value":[1765553241.136,"49414144"]},
            {"metric":{"name":"dockerfiles-clientapp-1-1"},
            "value":[1765553241.136,"48709632"]},
            {"metric":{"name":"dockerfiles-clientapp-2-1"},
            "value":[1765553241.136,"48713728"]}
            ]
        }
    }
    som ser ut til å være riktig.
    Får da verdien til 'container_memory_rss' ved UNIX timestamp '1765553241.136'
    for alle containere untatt prometheus, cadvisor og grafana,
    og også det jeg antar at er enten verdien for alle containere, eller verdien for hele datamaskinen generelt sett.
    Verdien for tomt navn er kanskje ikke så nyttig egentlig, så holder kanskje med
        sum(container_memory_rss{name!="",name!~"(prometheus|cadvisor|grafana|)"})by(name)
    for å eksludere den.
    """

    """Hente ut verdier over tid."""
    start_time = str(int(time.time()))
    time.sleep(10)
    end_time = str(int(time.time()))
    multi_value_request = 'http://localhost:9090/api/v1/query_range?query='
    ram_per_container = 'sum(container_memory_rss{name!="",name!~"(prometheus|cadvisor|grafana|)"})by(name)'
    start_end = f'&start={start_time}&end={end_time}&step=1s'
    # print(multi_value_request+ram_per_container_5min+start_end)
    response = requests.get(multi_value_request+ram_per_container+start_end)
    content = literal_eval(response.content.decode())
    # print(content)
    # print(content["data"]["result"][0]["metric"]["name"])
    # print(content["data"]["result"][0]["values"])

    # content_to_csv(content, "testing_ram_values.csv")


    """Får å hente ut verdier over tid kan man sende GET request til container, men trenger ekstra argumenter.
    'localhost:9090/api/v1/query_range?query=<QUERY>&start=<TIME>&end=<TIME>&step=<STEP>'

    <TIME> må være et unix timestamp, eks output fra time.time(). 
    <STEP> må være en steglengde, eks '1s' for ett sekund eller '5m' for 5 minutter.

    Requesten over gir output:
    {"status":"success",
    "data":
        {"resultType":"matrix",
        "result":
            [{"metric": {"name":"dockerfiles-clientapp-0-1"},
            "values":[[1765623119,"50819072"],[1765623120,"50819072"],[1765623121,"50819072"],[1765623122,"50819072"],[1765623123,"50819072"],[1765623124,"50819072"],[1765623125,"50819072"],[1765623126,"50819072"],[1765623127,"50819072"],[1765623128,"50819072"],[1765623129,"50819072"]]},
            {"metric": {"name":"dockerfiles-clientapp-1-1"},
            "values":[[1765623119,"48742400"],[1765623120,"48742400"],[1765623121,"48742400"],[1765623122,"48742400"],[1765623123,"48742400"],[1765623124,"48742400"],[1765623125,"48742400"],[1765623126,"48742400"],[1765623127,"48742400"],[1765623128,"48742400"],[1765623129,"48742400"]]},
            {"metric": {"name":"dockerfiles-clientapp-2-1"},
            "values":[[1765623119,"48738304"],[1765623120,"48738304"],[1765623121,"48738304"],[1765623122,"48738304"],[1765623123,"48738304"],[1765623124,"48738304"],[1765623125,"48738304"],[1765623126,"48738304"],[1765623127,"48738304"],[1765623128,"48738304"],[1765623129,"48738304"]]},
            {"metric": {"name":"dockerfiles-serverapp-1"},
            "values":[[1765623119,"102612992"],[1765623120,"102612992"],[1765623121,"102612992"],[1765623122,"102612992"],[1765623123,"102686720"],[1765623124,"102686720"],[1765623125,"102686720"],[1765623126,"102686720"],[1765623127,"98717696"],[1765623128,"98717696"],[1765623129,"102875136"]]},
            {"metric": {"name":"dockerfiles-superlink-1"},
            "values":[[1765623119,"94810112"],[1765623120,"94810112"],[1765623121,"94810112"],[1765623122,"94810112"],[1765623123,"94810112"],[1765623124,"94810112"],[1765623125,"94810112"],[1765623126,"94810112"],[1765623127,"94810112"],[1765623128,"94810112"],[1765623129,"94814208"]]},
            {"metric": {"name":"dockerfiles-supernode-0-1"},
            "values":[[1765623119,"93147136"],[1765623120,"93147136"],[1765623121,"93163520"],[1765623122,"93163520"],[1765623123,"93163520"],[1765623124,"93163520"],[1765623125,"93163520"],[1765623126,"93163520"],[1765623127,"93163520"],[1765623128,"93163520"],[1765623129,"93163520"]]},
            {"metric": {"name":"dockerfiles-supernode-1-1"},
            "values":[[1765623119,"93126656"],[1765623120,"93126656"],[1765623121,"93143040"],[1765623122,"93147136"],[1765623123,"93147136"],[1765623124,"93147136"],[1765623125,"93147136"],[1765623126,"93147136"],[1765623127,"93147136"],[1765623128,"93147136"],[1765623129,"93147136"]]},
            {"metric": {"name":"dockerfiles-supernode-2-1"},
            "values":[[1765623119,"93126656"],[1765623120,"93126656"],[1765623121,"93126656"],[1765623122,"93143040"],[1765623123,"93143040"],[1765623124,"93143040"],[1765623125,"93143040"],[1765623126,"93147136"],[1765623127,"93147136"],[1765623128,"93147136"],[1765623129,"93147136"]]}
            ]
        }
    }

    Så omtrent det samme som for ett tidspunkt, men man får ei liste med [tid, verdi] par med et par for hvert timestep.
    """

    """
    Navn på metrics jeg trenger:
    - Accuracy: 
        model_accuracy
    - Loss:
        model_loss
    - F1-Score:
        model_f1
    - CPU usage:
        container_cpu_usage_seconds_total
    - Memory usage:
        container_memory_rss
    - Memory Cached:
        container_memory_cache
    - Received Network Traffic:
        container_network_receive_bytes_total
    - Sent Network Traffic:
        container_network_transmit_bytes_total

    Grafana beregner også mean og max for alle container metrics.
    """

    accuracy_request = 'http://localhost:9090/api/v1/query?query='
    # name er det som identifiserer runde, så kan bruke alle metrics som har name.
    # alle andre verdier i 'metrics' nøkkel er irrelevante.
    acc_query = 'model_accuracy{name!=""}'
    print(accuracy_request+acc_query)
    response = requests.get(accuracy_request+acc_query)
    content = literal_eval(response.content.decode())
    # pprint.pprint(content)
    # content_to_csv(content, "testing_accuracy.csv")

    done_request = 'done_counter_total'
    response = requests.get(single_value_request+done_request)
    content = literal_eval(response.content.decode())
    # pprint.pprint(content)
    # print(content['data']['result'][0]['value'][1])

    """ 
    Jeg må finne ut av hvilken struktur hver av disse har, da cpu_query ihvertfall krasjer når jeg kaller content_to_csv på den
    pga at den ikke har en 'name' key?
    
    cpu_query = 'sum(rate(container_cpu_usage_seconds_total{name!="",name!~"(prometheus|cadvisor|grafana|)"}[10s]))by(name)'
    ram_query = 'sum(container_memory_rss{name!="",name!~"(prometheus|cadvisor|grafana|)"})by(name)'
    cache_query = 'sum(container_memory_cache{name!="",name!~"(prometheus|cadvisor|grafana|)"})by(name)'
    recv_net_query = 'sum(rate(container_network_receive_bytes_total{name!="",name!~"(prometheus|cadvisor|grafana|)"}[10s]))by(name)'
    sent_net_query = 'sum(rate(container_network_transmit_bytes_total{name!="",name!~"(prometheus|cadvisor|grafana|)"}[10s]))by(name)'
    """
    cpu_query = 'sum(rate(container_cpu_usage_seconds_total{name!="",name!~"(prometheus|cadvisor|grafana|)"}[10s]))by(name)'
    response = requests.get(multi_value_request+cpu_query+start_end)
    content = literal_eval(response.content.decode())
    pprint.pprint(content)

    request_to_csv(multi_value_request, cpu_query, start_end, "Testing for CPU bruk", "testing_cpu.csv")

    cpu_avg_query = 'avg(sum(rate(container_cpu_usage_seconds_total{name!="",name!~"(prometheus|cadvisor|grafana|)"}[10s]))by(name))by(name)'
    request_to_csv(multi_value_request, cpu_avg_query, start_end, "Average CPU bruk per container", "testing_cpu.csv")

    print("ALL TESTING DONE!")