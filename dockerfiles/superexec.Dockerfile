FROM flwr/superexec:1.35.0-py3.12-ubuntu24.04

WORKDIR /app

# Copy pyproject.toml to install dependencies.
COPY --chown=app:app fhefedavg/pyproject.toml .

# Copy lockfile to install dependencies.
# COPY --chown=app:app dockerfiles/requirements.txt .

# Run commands as root to avoid permission denied?
USER root

# Install dependencies declared in pyproject.toml
RUN sed -i 's/.*flwr\[simulation\].*//' pyproject.toml \
    && python -m pip install -U --no-cache-dir .

# Install dependencies from lockfile requirements.txt
# RUN python -m pip install -r ./requirements.txt

# Set environment variables to non-interactive (this prevents some prompts)
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary dependencies for OpenFHE and JupyterLab
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    # python3 \
    python3-dev \
    # python3-pip \
    python3-venv \
    sudo \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# # Install PyBind11
# RUN pip3 install "pybind11[global]"


# Clone and build OpenFHE-development
RUN git clone https://github.com/openfheorg/openfhe-development.git && \
    cd openfhe-development && \
    git checkout tags/v1.5.1
# RUN git tag
# RUN git checkout tags/v1.5.0

RUN cd openfhe-development \
    && mkdir build \
    && cd build \
    && cmake -DBUILD_UNITTESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF .. \
    && make -j$(nproc) \
    && make install

# Assume that OpenFHE installs libraries into /usr/local/lib
# Update LD_LIBRARY_PATH to include this directory
ENV LD_LIBRARY_PATH=/usr/local/lib 
#:${LD_LIBRARY_PATH}


RUN pip3 install openfhe==1.5.1.0.24.4

# Switch back to correct user before entrypoint.
WORKDIR /app
USER app

# Copy help code into containers so they can access it.
COPY --chown=app:app decryption_docker/ ./

# Copy flwr config.toml into container so that superlinks launched from inside container uses correct ports????
COPY --chown=app:app dockerfiles/flwr_config.toml /app/.flwr/config.toml

ENTRYPOINT ["flower-superexec"]