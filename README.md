GPU Inference Microservice with FastAPI + PyTorch

This project is a minimal, production-style microservice for GPU-accelerated AI/ML image classification. 
It uses pre-trained models (ResNet18 and ResNet50) from PyTorch, wrapped in a FastAPI server, and containerized with Docker.
Additionally, it includes Kubernetes deployment configurations for easy scaling and orchestration.

What It Does

- Accepts image uploads via a `/predict` endpoint
- Runs inference using a GPU-enabled ResNet model (user can select either ResNet18 or ResNet50)
- Returns the predicted class label (e.g., “golden retriever”)
- Deployable via Docker and Kubernetes

Tech Stack 
- PyTorch (ResNet18, pretrained on ImageNet)
- FastAPI for the web server
- Docker for containerization
- Kubernetes for orchestration and scaling
- Uvicorn for serving the API


Run It Yourself:

1. Clone the repo
   ```bash
   git clone https://github.com/your-username/gpu-inference-microservice.git
   cd gpu-inference-microservice
   ```

2. Build the Docker image - [Docker](https://www.docker.com/)
   ```
   docker build -t gpu-inference-app .
   ```
3. Run the container
   ```
   docker run -p 8000:8000 gpu-inference-app
   ```
4. Use the API
   ```
   Visit http://localhost:8000/docs
   Upload an image → get the predicted label
   ```

Running on GPU (Optional) 
   ```
   To enable GPU acceleration with NVIDIA GPUs, you'll need:
   * NVIDIA GPU drivers
   * NVIDIA Container Toolkit
   * Docker with GPU support

   Once set up, use this command to run the container using your GPU:

   docker run --gpus all -p 8000:8000 gpu-inference-app
   ```
Deploying with Kubernetes (Optional) 
If you want to deploy the application using Kubernetes: 

1. Ensure Minikube is installed:
   [Install Minikube](https://minikube.sigs.k8s.io/docs/start/?arch=%2Fwindows%2Fx86-64%2Fstable%2F.exe+download)
2. Start Minikube:
   ```
   minikube start --driver=docker
   ```
3. Build the docker image inside Minikube:
   ```
   eval $(minikube docker-env)
   docker build -t gpu-inference-app:latest .
   ```
4. Apply the Kubernetes Manifests:
   ```
   kubectl apply -f k8s/
   ```
5. Verify the Deployments:
   ```
   kubectl get pods
   kubectl get svc
   ```
6. Access the Application:
   ```
   minikube service gpu-inference-service
   ```
   
   
