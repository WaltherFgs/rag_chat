import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import CHUNK_SIZE, CHUNK_OVERLAP

def process_documents(uploaded_files):
    """Carga los PDFs subidos, los guarda temporalmente y los divide en fragmentos."""
    documents = []
    
    for file in uploaded_files:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        
        try:
            loader = PyPDFLoader(tmp_path)
            documents.extend(loader.load())
        finally:

            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    
    return documents, chunks
