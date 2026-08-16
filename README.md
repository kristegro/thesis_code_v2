# thesis_code_v2
Improved version of thesis code while allowing the original to remain undisturbed

## Install instructions

### Virtual environment

To create a virtual environment '.venv' inside the dir 'project-folder-name',
which should be substituted for an actual name, do

> python3 -m venv project-folder-name/.venv

where 'project-folder-name' should be substituted for an actual name.

To activate this env do:

> source project-folder-name/.venv/bin/activate

To deactivate the environment, run in a terminal:

> deactivate

### Install dependencies

Flower and the Prometheus client must be installed directly to be able to run experiments and process results.

These can be installed from the pyproject.toml or requirements.txt (lockfile) file in the outermost directory.

> pip install -e .

for pyproject or

> pip install -r requirements.txt

### Create necessary directories

All neccesary directories are created by ``create_directories.sh``.

- ``benchmarking_results`` is used for storing experiment results created by files in the `benchmarking` directory.

- ``storage/key_store`` and its subdirectories are used by clients and the server for storage of keys and other temporary files.

- ``storage/results_docker`` is used by clients and server to store misc files.

- ``storage/prometheus-storage`` is used by Promtheus.

- ``storage/grafana-storage`` is used by Grafana.
