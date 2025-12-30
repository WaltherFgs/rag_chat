import streamlit as st
import tempfile
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

load_dotenv()


st.set_page_config(page_title="Chat con documentos (RAG)", page_icon="📄")
st.title("📄 Chat con documentos (RAG)")

# --------- MODELO ----------
chat_model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5
)

# --------- INICIALIZAR SESSION STATE ----------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

# --------- SUBIR PDFs ----------
uploaded_files = st.file_uploader(
    "Sube tus documentos (PDF)",
    type=["pdf"],
    accept_multiple_files=True
)

# --------- PROCESAR DOCUMENTOS ----------
# Solo procesar si hay archivos nuevos o diferentes
if uploaded_files:
    current_files = [file.name for file in uploaded_files]
    
    # Verificar si son archivos diferentes a los ya procesados
    if current_files != st.session_state.processed_files:
        documents = []
        
        with st.spinner("📄 Cargando documentos..."):
            for file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp.write(file.read())
                    loader = PyPDFLoader(tmp.name)
                    documents.extend(loader.load())
        
        with st.spinner("✂️ Dividiendo documentos en fragmentos..."):
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100
            )
            chunks = splitter.split_documents(documents)
        
        with st.spinner("🧠 Generando embeddings y creando base vectorial..."):
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small"
            )
            st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
        
        st.session_state.retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
        st.session_state.processed_files = current_files
        
        # Notificación de éxito
        st.success(f"✅ ¡Listo! Procesados {len(documents)} páginas en {len(chunks)} fragmentos. Ya puedes hacer preguntas.")
else:
    # Si no hay archivos subidos, limpiar el estado
    st.session_state.vectorstore = None
    st.session_state.retriever = None
    st.session_state.processed_files = []

# --------- PROMPT ----------
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

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --------- CADENA RAG ----------
if st.session_state.retriever:
    chain = (
        {
            "context": st.session_state.retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | chat_model
    )

# --------- HISTORIAL DE MENSAJES ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --------- INPUT ----------
pregunta = st.chat_input("Haz una pregunta sobre los documentos")

# --------- RESPUESTA ----------
if pregunta and st.session_state.retriever:
    # Mostrar pregunta del usuario
    with st.chat_message("user"):
        st.markdown(pregunta)
    st.session_state.messages.append({"role": "user", "content": pregunta})
    
    # Generar y mostrar respuesta
    with st.chat_message("assistant"):
        placeholder = st.empty()
        respuesta = ""

        for chunk in chain.stream(pregunta):
            respuesta += chunk.content
            placeholder.markdown(respuesta)
    
    st.session_state.messages.append({"role": "assistant", "content": respuesta})

elif pregunta and not st.session_state.retriever:
    st.warning("Primero debes subir al menos un documento PDF.")
