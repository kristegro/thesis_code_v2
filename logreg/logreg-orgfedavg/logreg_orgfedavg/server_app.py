import os
from pathlib import Path
from typing import Optional, Union
from logging import WARNING

import torch
from flwr.common.logger import log
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import (
    Context,
    EvaluateRes,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.strategy.aggregate import aggregate, aggregate_inplace, weighted_loss_avg
from prometheus_client import Gauge, Counter, start_http_server

from logreg_orgfedavg.task import get_weights, set_weights, LogisticRegression

"""The only real difference between Log_FedAvg_Scraping and the Log_FedAvg 
in org-fedavg-docker is that Scraping takes Prometheus gauges as arguments
for initialization and updates these whenever model evaluation is done.
No calculations are actually changed."""

class Log_FedAvg_Scraping(FedAvg):
    def __init__(
            self, 
            accuracy_gauge: Gauge = None, 
            loss_gauge: Gauge = None,
            done_counter: Counter = None,
            num_rounds: int = None,
            scale: str = None,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.accuracy_gauge = accuracy_gauge
        self.loss_gauge = loss_gauge
        self.done_counter = done_counter
        self.num_rounds = num_rounds
        self.scale = scale

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        if self.inplace:
            # Does in-place weighted average of results
            aggregated_ndarrays = aggregate_inplace(results)
        else:
            # Convert results
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]
            aggregated_ndarrays = aggregate(weights_results)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        # Save to be able to create plots later.
        # 28*28 = 784, so 784 features.
        if self.scale == "classic":
            model = LogisticRegression(28*28, 10)
        elif self.scale == "larger":
            model = LogisticRegression(224*224, 10)
        elif self.scale == "rgb":
            model = LogisticRegression(3*224*224, 10)
        set_weights(model, parameters_to_ndarrays(parameters_aggregated))
        storage = os.path.abspath('storage/') + "/"
        torch.save(model.state_dict(), storage+f"logreg-tmp-r{server_round}")
        path = Path(storage+f"logreg-tmp-r{server_round}")
        Path.chmod(path, mode=0o777)

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        # with open("result_files/benchmark-org-fedavg-docker.txt", "a") as outfile:
        #     outfile.write(f"Round {server_round}: Aggregated loss = {loss_aggregated}\n")
        #     for metric in metrics_aggregated.keys():
        #         outfile.write(f"Round {server_round}: Aggregated {metric} = {metrics_aggregated[metric]}\n")
        #     outfile.write("\n")

        # Update the Prometheus gauges with the latest aggregated values
        self.loss_gauge.labels(f"round-{server_round}").set(loss_aggregated)
        self.accuracy_gauge.labels(f"round-{server_round}").set(metrics_aggregated['accuracy'])

        # Set counter to declare that last round is done.
        if server_round == self.num_rounds:
            self.done_counter.inc()

        return loss_aggregated, metrics_aggregated

def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize global model
    scale = context.run_config["scale"]
    # Load the model and initialize it with the received weights
    if scale == "classic":
        global_model = LogisticRegression(28*28, 10)
    elif scale == "larger":
        global_model = LogisticRegression(224*224, 10)
    elif scale == "rgb":
        global_model = LogisticRegression(3*224*224, 10)

    weights = get_weights(global_model)
    initial_parameters = ndarrays_to_parameters(weights)

    """Selv om initial_parameters er et Parameters objekt,
    så er parameters som clients mottar ei liste av numpy arrays,
    så parameters_to_ndarrays() blir nok kalt implisitt på forhånd."""

    # Create labels for each round.
    labels = ["name"]

    # Define a gauge to track the global model accuracy
    accuracy_gauge = Gauge("model_accuracy", 
                           "Current accuracy of the global model",
                           labelnames=labels)
    
    # Define a gauge to track the global model loss
    loss_gauge = Gauge("model_loss", 
                       "Current loss of the global model",
                       labelnames=labels)
    
    # Define counter to track when everything is done.
    done_counter = Counter("done_counter",
                           "Is everything done?")
    # Reset in case the value lingers in Prometheus.
    done_counter.reset()
    
    # Initialize labels for all gauges
    for i in range(1, num_rounds+1):
        accuracy_gauge.labels(name=f"round-{i}")
        loss_gauge.labels(name=f"round-{i}")

    # Define strategy
    strategy = Log_FedAvg_Scraping(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn=evaluation,
        accuracy_gauge=accuracy_gauge,
        loss_gauge=loss_gauge,
        done_counter=done_counter,
        num_rounds=num_rounds,
        scale=scale
    )
    config = ServerConfig(num_rounds=num_rounds)

    # Start Prometheus Metric server on the specified port.
    start_http_server(8000)

    return ServerAppComponents(strategy=strategy, config=config)


"""Prøv å skrive ut res.num_examples, res.metrics
for alle clients først.
Hvis num_examples er like for alle, beregn vanlig snitt
av accuracy.
Hvis ikke, beregn weighted average.
Beregn average for hver runde uansett, men det gjør vel
server allerede.

evaluate_metrics_aggregation_fn brukes slik:
eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
prøv å bare printe alle inputs først for å se hva som skjer.

Ser ut til at res.num_examples er størrelsen
på evaluation dataset, mens
res.metrics er dict på formen {'accuracy': <value>}.
Alle klientene ser ut til å ha samme størrelse på evaluation dataset
ihvertfall, for 1 round har alle res.num_examples = 32,
så kan beregne vanlig gjennomsnitt, men kanskje
like greit å beregne weighted."""

def evaluation(lst):
    # Sum of sizes to divide by.
    div = 0
    # Sum of metric*size.
    metrics_acc_sum = 0
    for (num_examples, metrics) in lst:
        div += num_examples
        metrics_acc_sum += metrics['accuracy']*num_examples
    # Return as dict since if not the program crashes.
    aggregation = {'accuracy': metrics_acc_sum/div}
    return aggregation


# Create ServerApp
app = ServerApp(server_fn=server_fn)
