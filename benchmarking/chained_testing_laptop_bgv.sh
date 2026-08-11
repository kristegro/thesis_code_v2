"#!/bin/bash"

# Bytte ut argumenter med riktige verdier for hver kjøring:
# benchmark_laptop.py [-h] program num_clients model num_tests scheme sec_level ring_dim chunk_size num_coeff mult_tech delta_bits
# /home/krist/master_thesis_code_kritg/.venv/bin/python \
# /home/krist/master_thesis_code_kritg/benchmarking/benchmark_laptop.py fhefedavg 7 logreg 3 BGV 128c 2**13 10*131072 16 HPS low && \
# wait && \
# /home/krist/master_thesis_code_kritg/.venv/bin/python \
# /home/krist/master_thesis_code_kritg/benchmarking/benchmark_laptop.py fhefedavg 3 squeezenet 3 BGV 256c 2**14 10*131072 16 HPS low && \ 
# wait && \
# sleep 15m && \
# /home/krist/master_thesis_code_kritg/.venv/bin/python \
# /home/krist/master_thesis_code_kritg/benchmarking/benchmark_laptop.py fhefedavg 3 squeezenet 2 BGV 256c 2**14 10*131072 16 HPS low && \ 
# wait && \
# /home/krist/master_thesis_code_kritg/.venv/bin/python \
# /home/krist/master_thesis_code_kritg/benchmarking/benchmark_laptop.py fhefedavg 3 squeezenet 2 BGV 256c 2**17 10*131072 16 HPS low && \
# wait && \
# /home/krist/master_thesis_code_kritg/.venv/bin/python \
# /home/krist/master_thesis_code_kritg/benchmarking/benchmark_laptop.py fhefedavg 3 squeezenet 3 BFV 256c 2**14 10*131072 16 HPS low && \ 
# wait && \
# /home/krist/master_thesis_code_kritg/.venv/bin/python \
# /home/krist/master_thesis_code_kritg/benchmarking/benchmark_laptop.py fhefedavg 3 squeezenet 3 BFV 256c 2**17 10*131072 16 HPS low && \
# wait && \
# /home/krist/master_thesis_code_kritg/.venv/bin/python \
# /home/krist/master_thesis_code_kritg/benchmarking/benchmark_laptop.py fhefedavg 5 squeezenet 3 BGV 256c 2**14 10*131072 16 HPS low && \ 
# wait && \
# /home/krist/master_thesis_code_kritg/.venv/bin/python \
# /home/krist/master_thesis_code_kritg/benchmarking/benchmark_laptop.py fhefedavg 5 squeezenet 2 BGV 256c 2**17 10*131072 16 HPS low && \
# wait && \
# /home/krist/master_thesis_code_kritg/.venv/bin/python \
# /home/krist/master_thesis_code_kritg/benchmarking/benchmark_laptop.py fhefedavg 5 squeezenet 3 BFV 256c 2**14 10*131072 16 HPS low && \ 
# wait && \
# /home/krist/master_thesis_code_kritg/.venv/bin/python \
# /home/krist/master_thesis_code_kritg/benchmarking/benchmark_laptop.py fhefedavg 5 squeezenet 3 BFV 256c 2**17 10*131072 16 HPS low && \
# wait && \
# sleep 15m && \

# /home/krist/master_thesis_code_kritg/.venv/bin/python \
# /home/krist/master_thesis_code_kritg/benchmarking/benchmark_laptop.py fhefedavg 3 squeezenet 3 CKKS-NF 256c 2**14 10*131072 16 HPS high && \ 
# wait && \
# /home/krist/master_thesis_code_kritg/.venv/bin/python \
# /home/krist/master_thesis_code_kritg/benchmarking/benchmark_laptop.py fhefedavg 3 squeezenet 3 CKKS-NF 256c 2**17 10*131072 16 HPS high && \ 
# wait && \
# /home/krist/master_thesis_code_kritg/.venv/bin/python \
# /home/krist/master_thesis_code_kritg/benchmarking/benchmark_laptop.py fhefedavg 5 squeezenet 3 CKKS-NF 256c 2**14 10*131072 16 HPS high && \ 
# wait && \
# /home/krist/master_thesis_code_kritg/.venv/bin/python \
# /home/krist/master_thesis_code_kritg/benchmarking/benchmark_laptop.py fhefedavg 7 squeezenet 1 BGV 256c 2**14 10*131072 16 HPS high && \ 
# wait && \
# sleep 15m && \
# /home/krist/master_thesis_code_kritg/.venv/bin/python \
# /home/krist/master_thesis_code_kritg/benchmarking/benchmark_laptop.py fhefedavg 3 bert 3 BGV 256c 2**17 10*131072 16 HPS high && \ 
# wait && \
# /home/krist/master_thesis_code_kritg/.venv/bin/python \
# /home/krist/master_thesis_code_kritg/benchmarking/benchmark_laptop.py fhefedavg 3 bert 2 BFV 256c 2**17 10*131072 16 HPS high && \ 
# wait &&
# /home/krist/master_thesis_code_kritg/.venv/bin/python \
# /home/krist/master_thesis_code_kritg/benchmarking/benchmark_laptop.py fhefedavg 5 bert 1 BGV 256c 2**17 10*131072 16 HPS high && \ 
# wait && \
# /home/krist/master_thesis_code_kritg/.venv/bin/python \
# /home/krist/master_thesis_code_kritg/benchmarking/benchmark_laptop.py fhefedavg 7 squeezenet 1 CKKS-NF 256c 2**14 10*131072 16 HPS high && \ 
# wait && \
# /home/krist/master_thesis_code_kritg/.venv/bin/python \
# /home/krist/master_thesis_code_kritg/benchmarking/benchmark_laptop.py fhefedavg 5 squeezenet 1 CKKS-NF 256c 2**14 10*131072 16 HPS high && \ 
# wait && \
# /home/krist/master_thesis_code_kritg/.venv/bin/python \
# /home/krist/master_thesis_code_kritg/benchmarking/benchmark_laptop.py fhefedavg 5 squeezenet 1 BGV 256c 2**17 10*131072 16 HPS high && \ 
# wait && \
# /home/krist/master_thesis_code_kritg/.venv/bin/python \
# /home/krist/master_thesis_code_kritg/benchmarking/benchmark_laptop.py fhefedavg 7 squeezenet 1 BGV 256c 2**17 10*131072 16 HPS high && \ 
# wait && \
# /home/krist/master_thesis_code_kritg/.venv/bin/python \
# /home/krist/master_thesis_code_kritg/benchmarking/benchmark_laptop.py fhefedavg 3 squeezenet 1 BGV 256c 2**14 10*131072 16 HPS high && \ 
# wait && \
/home/krist/master_thesis_code_kritg/.venv/bin/python \
/home/krist/master_thesis_code_kritg/benchmarking/benchmark_laptop.py fhefedavg 7 bert 1 BFV 256c 2**17 10*131072 16 HPS high && \ 
# wait && \
# /home/krist/master_thesis_code_kritg/.venv/bin/python \
# /home/krist/master_thesis_code_kritg/benchmarking/benchmark_laptop.py secagg 7 bert 2 BGV 256c 2**14 10*131072 16 HPS high && \ 
# wait && \
# /home/krist/master_thesis_code_kritg/.venv/bin/python \
# /home/krist/master_thesis_code_kritg/benchmarking/benchmark_laptop.py fhefedavg 5 squeezenet 1 CKKS-NF 256c 2**17 10*131072 16 HPS high && \ 
# wait && \
# /home/krist/master_thesis_code_kritg/.venv/bin/python \
# /home/krist/master_thesis_code_kritg/benchmarking/benchmark_laptop.py secagg 7 bert 3 CKKS-NF 256c 2**17 10*131072 16 HPS high && \ 
# /home/krist/master_thesis_code_kritg/.venv/bin/python \
# /home/krist/master_thesis_code_kritg/benchmarking/benchmark_laptop.py fhefedavg 3 bert 3 BGV 256c 2**17 10*131072 16 HPS low && \ 
# wait && \
# /home/krist/master_thesis_code_kritg/.venv/bin/python \
# /home/krist/master_thesis_code_kritg/benchmarking/benchmark_laptop.py fhefedavg 7 logreg 3 BGV 128c 2**13 10*131072 32 HPS low && \
# wait && \
# /home/krist/master_thesis_code_kritg/.venv/bin/python \
# /home/krist/master_thesis_code_kritg/benchmarking/benchmark_laptop.py fhefedavg 7 logreg 3 BGV 256c 2**14 10*131072 32 HPS low && \ 
# wait && \
# /home/krist/master_thesis_code_kritg/.venv/bin/python \
# /home/krist/master_thesis_code_kritg/benchmarking/benchmark_laptop.py fhefedavg 7 logreg 3 BGV 128c 2**17 10*131072 32 HPS low && \
# wait && \
# /home/krist/master_thesis_code_kritg/.venv/bin/python \
# /home/krist/master_thesis_code_kritg/benchmarking/benchmark_laptop.py fhefedavg 7 logreg 3 BGV 256c 2**17 10*131072 32 HPS low && \ 
wait