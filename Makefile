SHELL := /bin/sh

IMAGE ?= sepsis-env:latest
CONTAINER_NAME ?= sepsis-env
PORT ?= 7860
DOCKERFILE ?= Dockerfile
REGISTRY_IMAGE ?=

.PHONY: help build run run-local stop logs clean deploy

help:
	@echo "Targets:"
	@echo "  build   Build the Docker image"
	@echo "  run     Run the Docker container"
	@echo "  run-local  Run locally on port 8001"
	@echo "  stop    Stop the running container"
	@echo "  logs    Follow container logs"
	@echo "  clean   Remove the image"
	@echo "  deploy  Tag + push to registry (set REGISTRY_IMAGE)"
	@echo ""
	@echo "Examples:"
	@echo "  make build IMAGE=sepsis-env:latest"
	@echo "  make run PORT=7860"
	@echo "  make deploy REGISTRY_IMAGE=registry.hf.space/your-space:latest"

build:
	docker build -f $(DOCKERFILE) -t $(IMAGE) .

run:
	docker run --rm -p $(PORT):7860 --name $(CONTAINER_NAME) $(IMAGE)

run-local:
	docker run --rm -p 8001:7860 --name $(CONTAINER_NAME)-local $(IMAGE)
stop:
	-@docker stop $(CONTAINER_NAME) >/dev/null 2>&1 || true

logs:
	@docker logs -f $(CONTAINER_NAME)

clean:
	-@docker rmi $(IMAGE) >/dev/null 2>&1 || true

deploy:
	@if [ -z "$(REGISTRY_IMAGE)" ]; then \
		echo "Set REGISTRY_IMAGE=registry.hf.space/<space>:latest"; \
		exit 1; \
	fi
	docker tag $(IMAGE) $(REGISTRY_IMAGE)
	docker push $(REGISTRY_IMAGE)
