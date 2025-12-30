from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from .config import CHAT_MODEL_NAME, TEMPERATURE, RETRIEVER_K

def format_docs(docs):
    """Formatea los documentos recuperados para el contexto."""
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain(retriever):
    """Configura y devuelve la cadena de RAG."""
    chat_model = ChatOpenAI(
        model=CHAT_MODEL_NAME,
        temperature=TEMPERATURE
    )
    
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Responde SOLO con la información proporcionada en el contexto. "
            "Si no sabes, di que no está en el documento."
        ),
        (
            "human",
            "Contexto:\n{context}\n\nPregunta:\n{question}"
        )
    ])
    
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | chat_model
    )
    
    return chain
