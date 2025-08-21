.PHONY: install run test docker-build docker-run docker-stop docker-shell docker-logs docker-test

VENV?=venv
PY?=$(VENV)/Scripts/python.exe
PIP?=$(VENV)/Scripts/pip.exe
UVICORN?=$(VENV)/Scripts/uvicorn.exe

# Use POSIX defaults when running on mac/linux
ifeq (,$(findstring Windows_NT,$(OS)))
PY?=$(VENV)/bin/python
PIP?=$(VENV)/bin/pip
UVICORN?=$(VENV)/bin/uvicorn
endif

# App params
APP?=deepfake-microservice
TAG?=local
PORT?=8000

# ---- Docker CLI path handling (Windows-safe) ----
DOCKER?=docker
ifeq ($(OS),Windows_NT)
DOCKER?="C:/Program Files/Docker/Docker/resources/bin/docker.exe"
endif

install:
	python -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run:
	$(UVICORN) app.main:app --host 0.0.0.0 --port $(PORT) --reload

test:
	$(PY) -m pytest -q

# ---------- Docker ----------
docker-build:
	$(DOCKER) build -t $(APP):$(TAG) .

docker-run: docker-stop
	$(DOCKER) run --rm -d --name $(APP) -p $(PORT):8000 $(APP):$(TAG)

docker-stop:
	-@$(DOCKER) rm -f $(APP) 2> NUL || true

docker-shell:
	$(DOCKER) run --rm -it --entrypoint /bin/bash $(APP):$(TAG)

docker-logs:
	$(DOCKER) logs -f $(APP)

docker-test:
	# run tests inside the image
	$(DOCKER) run --rm --entrypoint python $(APP):$(TAG) -m pytest -q
