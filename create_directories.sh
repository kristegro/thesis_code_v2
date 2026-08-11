mkdir storage && \

# Create storage for FHE keys.
mkdir storage/key_store && \
mkdir storage/key_store/node0 && \
mkdir storage/key_store/node1 && \
mkdir storage/key_store/node2 && \
mkdir storage/key_store/node3 && \
mkdir storage/key_store/server && \
sudo chown 49999:49999 -R storage/key_store && \
sudo chmod 777 -R storage/key_store && \

# Create storage for misc files.
mkdir storage/results_docker && \
sudo chown 49999:49999 -R storage/results_docker && \
sudo chmod 777 -R storage/results_docker && \

# Create storage for prometheus.
mkdir storage/prometheus-storage && \
sudo chown 65534:65534 -R storage/prometheus-storage && \
sudo chmod 777 -R storage/prometheus-storage && \

# Create storage for Grafana.
mkdir storage/grafana-storage && \
sudo chown 472:472 -R storage/grafana-storage && \
sudo chmod 777 -R storage/grafana-storage && \

# Create storage for experiment results.
mkdir benchmarking_results