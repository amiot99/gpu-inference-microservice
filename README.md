GPU Inference Microservice with FastAPI + PyTorch

This project is a minimal, production-style microservice for GPU-accelerated image classification. 
It uses a pre-trained ResNet18 model from PyTorch, wrapped in a FastAPI server, and containerized with Docker.

What It Does

- Accepts image uploads via a `/predict` endpoint
- Runs inference using a GPU-enabled ResNet18 model
- Returns the predicted class label (e.g., “golden retriever”)

Tech Stack 
- PyTorch (ResNet18, pretrained on ImageNet)
- FastAPI for the web server
- Docker for containerization
- Uvicorn for serving the API


Run It Yourself:

1. Clone the repo
   ```bash
   git clone https://github.com/your-username/gpu-inference-microservice.git
   cd gpu-inference-microservice
   ```

2. Build the Docker image
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

   One set up, use this command to run the container using your GPU:

   docker run --gpus all -p 8000:8000 gpu-inference-app
   ```

   
