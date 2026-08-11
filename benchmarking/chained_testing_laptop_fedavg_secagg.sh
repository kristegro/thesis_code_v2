# Bytte ut argumenter med riktige verdier for hver kjøring:
# benchmark_laptop.py [-h] program num_clients model num_tests scheme sec_level ring_dim chunk_size num_coeff delta_bits

# /home/krist/master_thesis_code_kritg/.venv/bin/python \
# /home/krist/master_thesis_code_kritg/benchmarking/benchmark_laptop.py secagg 3 bert 3 BFV 128c 2**13 10*131072 16 HPS low && \
# wait && \
# /home/krist/master_thesis_code_kritg/.venv/bin/python \
# /home/krist/master_thesis_code_kritg/benchmarking/benchmark_laptop.py secagg 5 bert 3 BFV 128c 2**13 10*131072 16 HPS low && \ 
# wait && \
/home/krist/master_thesis_code_kritg/.venv/bin/python \
/home/krist/master_thesis_code_kritg/benchmarking/benchmark_laptop.py secagg 7 squeezenet 1 BFV 128c 2**13 10*131072 16 HPS low && \ 
# wait && \
# /home/krist/master_thesis_code_kritg/.venv/bin/python \
# /home/krist/master_thesis_code_kritg/benchmarking/benchmark_laptop.py secagg 3 squeezenet 3 BFV 128c 2**13 10*131072 16 HPS low && \
# wait && \
# /home/krist/master_thesis_code_kritg/.venv/bin/python \
# /home/krist/master_thesis_code_kritg/benchmarking/benchmark_laptop.py secagg 5 squeezenet 3 BFV 128c 2**13 10*131072 16 HPS low && \ 
# wait && \
# /home/krist/master_thesis_code_kritg/.venv/bin/python \
# /home/krist/master_thesis_code_kritg/benchmarking/benchmark_laptop.py secagg 7 squeezenet 3 BFV 128c 2**13 10*131072 16 HPS low && \ 
wait