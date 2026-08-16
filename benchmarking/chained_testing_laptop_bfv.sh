# Bytte ut argumenter med riktige verdier for hver kjøring:
# benchmark_laptop.py [-h] program num_clients model num_tests scheme sec_level ring_dim chunk_size num_coeff delta_bits

# /home/krist/thesis_code_v2/.venv/bin/python \
# /home/krist/thesis_code_v2/benchmarking/benchmark_laptop.py fhefedavg 7 logreg 3 BFV 128c 2**13 10*131072 16 HPS low && \
# wait && \
# /home/krist/thesis_code_v2/.venv/bin/python \
# /home/krist/thesis_code_v2/benchmarking/benchmark_laptop.py fhefedavg 7 logreg 3 BFV 128c 2**13 10*131072 32 HPS low && \ 
# wait && \
# /home/krist/thesis_code_v2/.venv/bin/python \
# /home/krist/thesis_code_v2/benchmarking/benchmark_laptop.py fhefedavg 3 squeezenet 3 BFV 256c 2**14 10*131072 16 HPS low && \ 
# wait && \
# /home/krist/thesis_code_v2/.venv/bin/python \
# /home/krist/thesis_code_v2/benchmarking/benchmark_laptop.py fhefedavg 7 logreg 3 BFV 256c 2**14 10*131072 32 HPS low && \ 
# wait && \
/home/krist/thesis_code_v2/.venv/bin/python \
/home/krist/thesis_code_v2/benchmarking/benchmark_laptop.py fhefedavg 3 squeezenet 1 BFV 256c 2**17 10*131072 16 HPS low && \
# wait && \
# /home/krist/thesis_code_v2/.venv/bin/python \
# /home/krist/thesis_code_v2/benchmarking/benchmark_laptop.py fhefedavg 7 logreg 3 BFV 128c 2**17 10*131072 32 HPS low && \ 
# wait && \
# /home/krist/thesis_code_v2/.venv/bin/python \
# /home/krist/thesis_code_v2/benchmarking/benchmark_laptop.py fhefedavg 7 logreg 3 BFV 256c 2**17 10*131072 16 HPS low && \ 
# wait && \
# /home/krist/thesis_code_v2/.venv/bin/python \
# /home/krist/thesis_code_v2/benchmarking/benchmark_laptop.py fhefedavg 7 logreg 3 BFV 256c 2**17 10*131072 32 HPS low && \ 
# wait && \
# /home/krist/thesis_code_v2/.venv/bin/python \
# /home/krist/thesis_code_v2/benchmarking/benchmark_laptop.py fhefedavg 7 logreg 3 BFV 128c 2**13 10*131072 16 BEHZ low && \
# wait && \
# /home/krist/thesis_code_v2/.venv/bin/python \
# /home/krist/thesis_code_v2/benchmarking/benchmark_laptop.py fhefedavg 7 logreg 3 BFV 128c 2**13 10*131072 32 BEHZ low && \ 
# wait && \
# /home/krist/thesis_code_v2/.venv/bin/python \
# /home/krist/thesis_code_v2/benchmarking/benchmark_laptop.py fhefedavg 7 logreg 3 BFV 256c 2**14 10*131072 16 BEHZ low && \ 
# wait && \
# /home/krist/thesis_code_v2/.venv/bin/python \
# /home/krist/thesis_code_v2/benchmarking/benchmark_laptop.py fhefedavg 7 logreg 3 BFV 256c 2**14 10*131072 32 BEHZ low && \ 
# wait && \
# /home/krist/thesis_code_v2/.venv/bin/python \
# /home/krist/thesis_code_v2/benchmarking/benchmark_laptop.py fhefedavg 7 logreg 3 BFV 128c 2**17 10*131072 16 BEHZ low && \
# wait && \
# /home/krist/thesis_code_v2/.venv/bin/python \
# /home/krist/thesis_code_v2/benchmarking/benchmark_laptop.py fhefedavg 7 logreg 3 BFV 128c 2**17 10*131072 32 BEHZ low && \ 
# wait && \
# /home/krist/thesis_code_v2/.venv/bin/python \
# /home/krist/thesis_code_v2/benchmarking/benchmark_laptop.py fhefedavg 7 logreg 3 BFV 256c 2**17 10*131072 16 BEHZ low && \ 
# wait && \
# /home/krist/thesis_code_v2/.venv/bin/python \
# /home/krist/thesis_code_v2/benchmarking/benchmark_laptop.py fhefedavg 7 logreg 3 BFV 256c 2**17 10*131072 32 BEHZ low && \ 
wait