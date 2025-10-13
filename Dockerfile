ARG BUILDER_IMAGE="ghcr.io/reenvision-ai/base-image:1.1.0"
#FROM nvcr.io/nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04
#FROM nvcr.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS builder
FROM ${BUILDER_IMAGE} AS builder
LABEL maintainer="ReEnvision-AI"
LABEL repository="agent-grid"
LABEL org.opencontainers.image.source="https://github.com/ReEnvision-AI/agent-grid"

WORKDIR /home
# Set en_US.UTF-8 locale by default
RUN echo "LC_ALL=en_US.UTF-8" >> /etc/environment

VOLUME /cache
ENV AGENT_GRID_CACHE=/cache

# Create project directory and copy only dependency-related files first to leverage Docker cache
RUN mkdir -p /home/agent-grid/src/agentgrid/
#COPY setup.cfg /home/agent-grid/
COPY pyproject.toml /home/agent-grid/
COPY src/agentgrid/VERSION /home/agent-grid/src/agentgrid/VERSION
COPY src/agentgrid/__init__.py /home/agent-grid/src/agentgrid/__init__.py

# This directory is required by pyproject.toml to install the local .whl file.
COPY deps/ /home/agent-grid/deps/

# Install core project dependencies
WORKDIR /home/agent-grid/
RUN pip install --no-cache-dir -e .[full]

# Copy the rest of the application code
COPY . /home/agent-grid/

# --- Runtime Stage ---
FROM nvcr.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

LABEL maintainer="ReEnvision-AI"
LABEL repository="agent-grid"
LABEL org.opencontainers.image.source="https://github.com/ReEnvision-AI/agent-grid"

WORKDIR /home
# Set en_US.UTF-8 locale by default
RUN echo "LC_ALL=en_US.UTF-8" >> /etc/environment

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  git \
  tzdata \
  && apt-get install software-properties-common -y --no-install-recommends \
  && add-apt-repository -y ppa:deadsnakes/ppa \
  && apt-get update && apt-get install -y --no-install-recommends \
  python3.11 \
  python3.11-dev \
  python3.11-distutils \
  curl \
  && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
  && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
  && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
  && apt-get clean autoclean && rm -rf /var/lib/apt/lists/{apt,dpkg,cache,log} /tmp/* /var/tmp/*

VOLUME /cache
ENV AGENT_GRID_CACHE=/cache

# Copy installed Python packages from the builder stage
COPY --from=builder /usr/local/lib/python3.11/dist-packages/ /usr/local/lib/python3.11/dist-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/ 

# Copy the application code
COPY --from=builder /home/agent-grid/ /home/agent-grid/

ENV HF_HUB_DISABLE_XET=1

WORKDIR /home/agent-grid/
CMD bash
