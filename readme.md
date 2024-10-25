# Compliance GPT

Compliance GPT is a Python app for dealing with large compliance document to extract updated information.

## Installing Ollama LLM instances as docker container

https://hub.docker.com/r/ollama/ollama site has been referenced to install local LLMs.

Ollama docker container will be started using docker-compose.yaml file.

After ollama docker container started execute this command to load your preferred LLM.


```bash
docker exec -it ollama ollama pull llama3.1:8b
```

## Vector Database

Qdrant has been used as vector database.

Qdrant docker container will be started using docker-compose.yaml file.

## Streamlit Application

Self - RAG has been used in this application as described 
[here](https://python.langchain.com/docs/modules/agents/tools/custom_tools)

## Execute whole application
Run 
```bash 
docker compose build
docker compose up
```
commands respectively. 