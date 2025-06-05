# Define variables
APP_NAME = local-rag-app
APP_PORT = 8000


OLLAMA_HOST = http://host.docker.internal:11434

.PHONY: all build run stop clean ollama-pull

all: build run

build:
	@echo "Building Docker image for $(APP_NAME)..."
	docker build -t $(APP_NAME) .
	@echo "Docker image built: $(APP_NAME)"

run:
	@echo "Running Docker container for $(APP_NAME) on port $(APP_PORT)..."
	@echo "Ensure Ollama is running on your host machine and accessible at $(OLLAMA_HOST)"
	docker run -d \
		--name $(APP_NAME) \
		-p $(APP_PORT):$(APP_PORT) \
		-v $(PWD)/documents:/app/documents \
		-v $(PWD)/faiss_index:/app/faiss_index \
		-e OLLAMA_HOST=$(OLLAMA_HOST) \
		$(APP_NAME)
	@echo "Container $(APP_NAME) started. Access FastAPI at http://localhost:$(APP_PORT)/docs"
	@echo "Check logs with: docker logs $(APP_NAME)"

stop:
	@echo "Stopping Docker container $(APP_NAME)..."
	docker stop $(APP_NAME) || true
	docker rm $(APP_NAME) || true
	@echo "Container $(APP_NAME) stopped and removed."

clean:
	@echo "Cleaning up generated files..."
	rm -rf faiss_index/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "Cleanup complete."

ollama-pull:
	@echo "Pulling Mistral model using Ollama. (Requires Ollama CLI installed on host)"
	ollama pull mistral
	@echo "Mistral model pulled."

logs:
	@echo "Showing logs for $(APP_NAME)..."
	docker logs -f $(APP_NAME)