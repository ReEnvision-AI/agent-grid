#FROM nvcr.io/nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04
FROM nvcr.io/nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04
LABEL maintainer="ReEnvision-AI"
LABEL repository="agent-grid"
LABEL org.opencontainers.image.source="https://github.com/ReEnvision-AI/agent-grid"

WORKDIR /home
# Set en_US.UTF-8 locale by default
RUN echo "LC_ALL=en_US.UTF-8" >> /etc/environment

# Install packages
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  python3 \
  python3-pip \
  python-is-python3 \
  python3-dev \
  git \
  && apt-get clean autoclean && rm -rf /var/lib/apt/lists/{apt,dpkg,cache,log} /tmp/* /var/tmp/*

#RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O install_miniconda.sh && \
#  bash install_miniconda.sh -b -p /opt/conda && rm install_miniconda.sh
#ENV PATH="/opt/conda/bin:${PATH}"

#RUN conda install python~=3.11 pip && \
RUN pip install --no-cache-dir "torch>=1.12" 
#    conda clean --all && rm -rf ~/.cache/pip

VOLUME /cache
ENV AGENT_GRID_CACHE=/cache

COPY . agent-grid/
RUN pip install --no-cache-dir -e agent-grid

WORKDIR /home/agent-grid/
CMD bash
