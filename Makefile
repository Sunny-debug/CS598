.PHONY: install run test fmt lint

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

install:
	python -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run:
	$(UVICORN) app.main:app --host 0.0.0.0 --port 8000 --reload

test:
	$(PY) -m pytest -q