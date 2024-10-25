#Run application using local LLM
#Installing Ollama LLM instances as docker container
https://hub.docker.com/r/ollama/ollama site has been referenced to install local LLMs.
Ollama docker container will be started using docker-compose.yaml file.
After ollama docker container started execute this command to load your preferred LLM.
docker exec -it ollama ollama pull qwen2:1.5b


#Vector Database
Qdrant has been used as vector database.
Qdrant docker container will be started using docker-compose.yaml file.


#Streamlit Application
Self - RAG has been used in this application as described here.
https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/rag/langgraph_self_rag_local.ipynb 


#Execute whole application
Run "docker compose build" and "docker compose up" commands respectively. 





