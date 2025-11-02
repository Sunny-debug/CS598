# Use Bash for eval; required for minikube docker-env
SHELL := /bin/bash
.ONESHELL:

.PHONY: install run test docker-build docker-run docker-stop docker-shell docker-logs docker-test \
        k8s-build k8s-load k8s-restart k8s-url k8s-port-forward k8s-logs streamlit venv clean deploy
.PHONY: hardening-apply

# -------------------- Virtualenv / tool paths --------------------
VENV ?= venv
PY    ?= $(VENV)/Scripts/python.exe
PIP   ?= $(VENV)/Scripts/pip.exe
UVICORN ?= $(VENV)/Scripts/uvicorn.exe

# POSIX defaults when not on Windows
ifeq (,$(findstring Windows_NT,$(OS)))
PY     := $(VENV)/bin/python
PIP    := $(VENV)/bin/pip
UVICORN:= $(VENV)/bin/uvicorn
NULLDEV := /dev/null
else
NULLDEV := NUL
endif

# -------------------- App / image params --------------------
APP ?= deepfake-microservice
TAG ?= 0.1.0
IMAGE ?= $(APP):$(TAG)
PORT ?= 8000

# Docker CLI path (your original)
DOCKER ?= docker
ifeq ($(OS),Windows_NT)
DOCKER ?= "C:/Program Files/Docker/Docker/resources/bin/docker.exe"
endif

# -------------------- Python tasks --------------------
venv:
	python -m venv $(VENV)

install: venv
	$(PIP) install --upgrade pip wheel
	$(PIP) install -r requirements.txt

run:
	$(UVICORN) app.main:app --host 0.0.0.0 --port $(PORT) --reload

test:
	$(PY) -m pytest -q

clean:
	-@rm -rf $(VENV) 2> $(NULLDEV) || true
	-@find . -type d -name "__pycache__" -exec rm -rf {} + 2> $(NULLDEV) || true

# -------------------- Docker --------------------
docker-build:
	$(DOCKER) build -t $(IMAGE) .

docker-run: docker-stop
	$(DOCKER) run --rm -d --name $(APP) -p $(PORT):8000 $(IMAGE)

docker-stop:
	-@$(DOCKER) rm -f $(APP) 2> $(NULLDEV) || true

docker-shell:
	$(DOCKER) run --rm -it --entrypoint /bin/bash $(IMAGE)

docker-logs:
	$(DOCKER) logs -f $(APP)

docker-test:
	# run tests inside the image
	$(DOCKER) run --rm --entrypoint python $(IMAGE) -m pytest -q

# -------------------- Minikube / K8s helpers --------------------
# Build directly into Minikube's Docker daemon
k8s-build:
	eval $$(minikube docker-env); $(DOCKER) build -t $(IMAGE) .

# If you built locally, copy the image into Minikube's daemon
k8s-load:
	minikube image load $(IMAGE)

# Restart deployment to pick up the new image
k8s-restart:
	kubectl rollout restart deploy/deepfake-detector
	kubectl rollout status  deploy/deepfake-detector

# Print reachable URL (NodePort)
k8s-url:
	@echo "$$(minikube ip):$$(kubectl get svc deepfake-detector-svc -o jsonpath='{.spec.ports[0].nodePort}')"

# Port-forward service -> localhost:8080
k8s-port-forward:
	kubectl port-forward svc/deepfake-detector-svc 8080:80

# Tail app logs from the Deployment
k8s-logs:
	kubectl logs deploy/deepfake-detector -f
	
hardening-apply:
	@set -euo pipefail
	kubectl apply -f k8s/deployment-api.yaml
	kubectl apply -f k8s/deployment-streamlit-realeyes.yaml
	# Optional: only if you created it
	-kubectl apply -f k8s/secret-api.yaml
	# Optional: CPU autoscaling for API
	-kubectl apply -f k8s/hpa-api.yaml
	# Optional: restrict traffic to API
	-kubectl apply -f k8s/netpol-api.yaml
	# Optional: Prometheus scraping (if you add one)
	-kubectl apply -f k8s/servicemonitor-api.yaml
	kubectl -n deepfake rollout status deploy/deepfake-api
	kubectl -n deepfake rollout status deploy/deepfake-ui-v2
# -------------------- Streamlit demo client --------------------
streamlit:
	$(PY) -m pip install -U streamlit requests
	$(PY) -m streamlit run streamlit_app/app.py

train:
	@export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True && \
	venv-train/bin/python -m training.train_unet \
		--src_dir data/clean/edited \
		--out_dir checkpoints \
		--epochs 2 \
		--bs 2 \
		--size 256

# -------------------- Deploy --------------------
deploy:
	@set -euo pipefail
	SHA7="$$(git rev-parse --short=7 HEAD)"
	echo "Rolling out tag $$SHA7..."
	kubectl -n deepfake set image deploy deepfake-api \
	  api=ghcr.io/sunny-debug/cs598-api:$$SHA7
	kubectl -n deepfake set image deploy deepfake-ui-v2 \
	  ui=ghcr.io/sunny-debug/cs598-ui:$$SHA7
	kubectl -n deepfake rollout status deploy/deepfake-api
	kubectl -n deepfake rollout status deploy/deepfake-ui-v2

rollback:
	@set -euo pipefail
	kubectl -n deepfake rollout undo deploy/deepfake-api
	kubectl -n deepfake rollout undo deploy/deepfake-ui-v2
	kubectl -n deepfake rollout status deploy/deepfake-api
	kubectl -n deepfake rollout status deploy/deepfake-ui-v2	