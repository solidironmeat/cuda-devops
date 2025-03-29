# DevOps with CUDA
Automates CUDA app deployment with Docker and Jenkins for seamless GPU workflows.

## Setup
1. Install Docker and NVIDIA Docker.
2. Build: `docker build -t nvidia/cuda:12.8.1-base-ubuntu24.04 .`
3. Run: `docker run --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 /app/test_cuda`

## CI/CD
- Github Actions pipeline builds, tests, and deploys the app.
- Jenkins pipeline builds, tests, and deploys the app.
- Requires Jenkins with NVIDIA Docker support.