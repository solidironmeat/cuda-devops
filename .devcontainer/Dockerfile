FROM nvidia/cuda:12.8.1-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install basic dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    nvidia-cuda-toolkit \
    build-essential \
    git \
    cmake \
    gdb \
    libgtest-dev \
    clang-tidy \
    clang-format

# Set working directory
WORKDIR /app

# Default command
CMD ["bash"]