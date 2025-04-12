FROM nvidia/cuda:12.6.0-base-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    sudo \
    git \
    wget \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install ipykernel && \
    python3 -m ipykernel install --user --name python3 --display-name "Python 3"

WORKDIR /mount
CMD ["tail", "-f", "/dev/null"]