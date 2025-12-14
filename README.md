# RealEyes

**RealEyes** is a production‑grade AI microservice for detecting manipulated (deepfake) images, built with an end‑to‑end MLOps + DevOps mindset. It combines deep‑learning–based image forensics with cloud‑native engineering practices: containerization, Kubernetes orchestration, CI/CD automation, and observability.

---

## Executive Summary

* **Problem**: Synthetic and manipulated images undermine trust in digital media.
* **Solution**: A scalable AI microservice that detects and localizes manipulated regions in images.
* **Approach**: Deep learning (U‑Net–style segmentation) deployed as a FastAPI service, containerized with Docker, orchestrated via Kubernetes, and monitored with Prometheus/Grafana.
* **Outcome**: A reproducible, observable, and production‑ready system suitable for academic evaluation and industry demonstration.

---

## High‑Level Architecture

**Flow**:

1. User uploads an image (REST API / UI)
2. API validates and preprocesses input
3. Deep learning model performs inference
4. Model returns:

   * Manipulation probability (real vs fake)
   * Pixel‑level mask highlighting edited regions
5. Metrics and logs are exported for monitoring

**Core Components**:

* **Model**: PyTorch U‑Net–based image segmentation
* **API**: FastAPI + Uvicorn
* **Containerization**: Docker (non‑root, slim images)
* **Orchestration**: Kubernetes (Minikube locally, EKS‑ready)
* **Monitoring**: Prometheus + Grafana
* **CI/CD**: GitHub Actions

---

## Technology Stack

### Machine Learning

* Python 3.11
* PyTorch
* Transfer‑learning–friendly U‑Net architecture
* Deterministic inference with reproducible checkpoints

### Backend / API

* FastAPI
* Uvicorn
* Pydantic v2
* Pillow / NumPy

### DevOps / Platform

* Docker (multi‑stage, non‑root)
* Kubernetes (Namespaces, Deployments, Services, PVCs)
* Prometheus + Grafana
* GitHub Actions (CI + CD)

---

## Repository Structure

```text
RealEyes/
├── app/
│   ├── main.py              # FastAPI entrypoint
│   ├── config.py            # Environment‑based configuration
│   ├── models/
│   │   └── unet_infer.py     # Model loading & inference
│   └── utils/               # Pre/post‑processing helpers
├── checkpoints/             # Model weights (mounted in prod)
├── k8s/
│   ├── api-deployment.yaml
│   ├── api-service.yaml
│   ├── servicemonitor.yaml
│   └── namespaces.yaml
├── tests/
│   └── test_api.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── Makefile
└── README.md
```

---

## Model Details

* **Architecture**: U‑Net‑style convolutional neural network
* **Task**: Pixel‑level manipulation segmentation
* **Output**:

  * Probability score for manipulation
  * Binary or heatmap mask of edited regions

> The system is designed to be dataset‑agnostic. Training can be performed on FaceForensics++, Celeb‑DF, or synthetic datasets. Dataset access is intentionally decoupled from the repo.

---

## API Overview

### Health Check

```
GET /health
```

Returns service liveness and readiness.

### Prediction

```
POST /predict
```

**Input**: Image file (JPEG/PNG)

**Output**:

```json
{
  "label": "fake",
  "score": 0.87,
  "mask": "<base64-encoded segmentation mask>"
}
```

### Metrics

```
GET /metrics
```

Prometheus‑compatible metrics endpoint.

---

## Local Development

### 1. Create Virtual Environment

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run API Locally

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Run with Docker

```bash
docker build -t realeyes-api .
docker run -p 8000:8000 realeyes-api
```

---

## Kubernetes Deployment

* Uses isolated namespaces (`deepfake`, `monitoring`)
* Model weights mounted via PersistentVolume (read‑only)
* Horizontal scaling supported via replicas
* Metrics scraped using ServiceMonitor

```bash
kubectl apply -f k8s/
```

---

## Observability

**Metrics**:

* Request count & latency
* Error rates
* Model inference duration
* Pod restarts and health

**Dashboards**:

* API performance
* Resource utilization (CPU / memory)
* System stability over time

---

## CI/CD Pipeline

**Continuous Integration**:

* Linting & unit tests
* Docker image build
* Image vulnerability‑aware design (non‑root, slim base)

**Continuous Deployment**:

* Image pushed to registry
* Smoke tests executed post‑deploy

---

## Security Considerations

* Non‑root Docker containers
* Explicit upload size limits
* Strict image type validation
* No secrets committed (env‑based config)
* Namespace isolation in Kubernetes

---
