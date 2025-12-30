from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from .config import EMBEDDING_MODEL_NAME

def get_vector_store(chunks):
    """Crea una base vectorial FAISS a partir de fragmentos de texto."""
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL_NAME
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore
