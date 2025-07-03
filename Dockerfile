#FROM nvcr.io/nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04
FROM nvcr.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
LABEL maintainer="ReEnvision-AI"
LABEL repository="agent-grid"
LABEL org.opencontainers.image.source="https://github.com/ReEnvision-AI/agent-grid"

WORKDIR /home
# Set en_US.UTF-8 locale by default
RUN echo "LC_ALL=en_US.UTF-8" >> /etc/environment

# Install packages
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  software-properties-common \
  git \
  tzdata \
  && add-apt-repository -y ppa:deadsnakes/ppa \
  && apt-get update && apt-get install -y --no-install-recommends \
  python3.11 \
  python3.11-dev \
  python3.11-distutils \
  curl \
  && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
  && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
  && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
  && apt-get -y purge software-properties-common \
  && apt-get clean autoclean && rm -rf /var/lib/apt/lists/{apt,dpkg,cache,log} /tmp/* /var/tmp/*

#RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O install_miniconda.sh && \
#  bash install_miniconda.sh -b -p /opt/conda && rm install_miniconda.sh
#ENV PATH="/opt/conda/bin:${PATH}"

#RUN conda install python~=3.11 pip && \
RUN pip install --no-cache-dir "torch==2.6.0" --index-url "https://download.pytorch.org/whl/cu124"
#    conda clean --all && rm -rf ~/.cache/pip

VOLUME /cache
ENV AGENT_GRID_CACHE=/cache

COPY . agent-grid/
RUN pip install --no-cache-dir -e agent-grid

# Install flash attention for CUDA 12.8, PyTorch 2.6.0, and Python 3.11
RUN MAX_JOBS=4 pip install --no-cache-dir "flash-attn==2.6.2" --use-pep517 --no-build-isolation

WORKDIR /home/agent-grid/
CMD bash
