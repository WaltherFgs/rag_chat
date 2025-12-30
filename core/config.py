import os
from dotenv import load_dotenv

load_dotenv()

# Configuraciones de Modelo
CHAT_MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
TEMPERATURE = 0.5

# Configuraciones de Fragmentación
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Configuraciones de Búsqueda
RETRIEVER_K = 4
