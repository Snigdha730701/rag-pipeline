

## Setup Instructions

To set up a virtual environment, run the following commands:
```bash
cd rag-pipeline
```
To switch to pyhton virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
Install required packages using pip:
```bash
pip install -r requirements.txt
```

Install Ollama: Follow the instructions on the official Ollama website: https://ollama.com/
```bash
ollama pull mistral
```
Run the FastAPI Application
Start the FastAPI application locally
```bash
uvicorn main:app --reload
```

## Run the application using Makefile
Pull the Mistral model using Ollama
```bash
make ollama-pull
```
Build the Docker image
```bash
make build
```
Run the Docker container
```bash
make run
```

## Screenshot

![Project Screenshot showing the main UI](Image/Output-image.jpg "Main UI Screenshot")

Output
