"#!/bin/bash"

# Bytte ut argumenter med riktige verdier for hver kjøring:
# benchmark_laptop.py [-h] program num_clients model num_tests scheme sec_level ring_dim chunk_size num_coeff mult_tech delta_bits
/home/krist/thesis_code_v2/.venv/bin/python \
/home/krist/thesis_code_v2/benchmarking/benchmark_laptop.py fhefedavg 7 logreg 3 CKKS-NF 128c 2**13 10*131072 16 HPS high && \
wait && \
/home/krist/thesis_code_v2/.venv/bin/python \
/home/krist/thesis_code_v2/benchmarking/benchmark_laptop.py fhefedavg 7 logreg 3 CKKS-NF 256c 2**14 10*131072 16 HPS high && \ 
wait && \
/home/krist/thesis_code_v2/.venv/bin/python \
/home/krist/thesis_code_v2/benchmarking/benchmark_laptop.py fhefedavg 7 logreg 3 CKKS-NF 128c 2**17 10*131072 16 HPS high && \
wait && \
/home/krist/thesis_code_v2/.venv/bin/python \
/home/krist/thesis_code_v2/benchmarking/benchmark_laptop.py fhefedavg 7 logreg 3 CKKS-NF 256c 2**17 10*131072 16 HPS high && \ 
wait && \
/home/krist/thesis_code_v2/.venv/bin/python \
/home/krist/thesis_code_v2/benchmarking/benchmark_laptop.py fhefedavg 7 logreg 3 CKKS-NF 128c 2**13 10*131072 16 HPS low && \
wait && \
/home/krist/thesis_code_v2/.venv/bin/python \
/home/krist/thesis_code_v2/benchmarking/benchmark_laptop.py fhefedavg 7 logreg 3 CKKS-NF 256c 2**14 10*131072 16 HPS low && \ 
wait && \
/home/krist/thesis_code_v2/.venv/bin/python \
/home/krist/thesis_code_v2/benchmarking/benchmark_laptop.py fhefedavg 7 logreg 1 CKKS-NF 128c 2**17 10*131072 16 HPS low && \
wait && \
/home/krist/thesis_code_v2/.venv/bin/python \
/home/krist/thesis_code_v2/benchmarking/benchmark_laptop.py fhefedavg 7 logreg 3 CKKS-NF 256c 2**17 10*131072 16 HPS low && \ 
wait