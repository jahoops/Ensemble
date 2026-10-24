# Use the base image with Ubuntu 22.04 and CUDA
FROM ubuntu:22.04

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install TensorFlow with GPU support
RUN pip3 install --upgrade pip
RUN pip3 install tensorflow-gpu

# Set the working directory
WORKDIR /workspace

# Copy the requirements file
COPY requirements.txt .

# Install the Python dependencies
RUN pip3 install -r requirements.txt

# Copy the rest of the application
COPY . .

# Set the entry point
ENTRYPOINT ["python3"]

