import streamlit as st
from core.processor import process_documents
from core.vector_store import get_vector_store
from core.engine import get_rag_chain
from core.config import RETRIEVER_K

# Configuración de la página
st.set_page_config(page_title="Chat con documentos (RAG)", page_icon="📄", layout="wide")
st.title("📄 Chat con documentos (RAG)")
st.markdown("---")

# --------- INICIALIZAR SESSION STATE ----------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "messages" not in st.session_state:
    st.session_state.messages = []

# --------- BARRA LATERAL (Subida de archivos) ----------
with st.sidebar:
    st.header("Configuración")
    uploaded_files = st.file_uploader(
        "Sube tus documentos (PDF)",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    if st.button("Limpiar historial"):
        st.session_state.messages = []
        st.rerun()

# --------- PROCESAR DOCUMENTOS ----------
if uploaded_files:
    current_files = [file.name for file in uploaded_files]
    
    # Solo procesar si hay cambios en los archivos
    if current_files != st.session_state.processed_files:
        with st.status("Procesando documentos...", expanded=True) as status:
            st.write("📄 Cargando y dividiendo PDFs...")
            documents, chunks = process_documents(uploaded_files)
            
            st.write("🧠 Generando base vectorial...")
            st.session_state.vectorstore = get_vector_store(chunks)
            st.session_state.retriever = st.session_state.vectorstore.as_retriever(
                search_kwargs={"k": RETRIEVER_K}
            )
            
            st.session_state.processed_files = current_files
            status.update(label="✅ Documentos procesados con éxito!", state="complete", expanded=False)
            st.success(f"Se procesaron {len(documents)} páginas y se crearon {len(chunks)} fragmentos.")
else:
    # Limpiar estado si no hay archivos
    st.session_state.vectorstore = None
    st.session_state.retriever = None
    st.session_state.processed_files = []

# --------- INTERFAZ DE CHAT ----------

# Mostrar historial de mensajes
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input de usuario
if pregunta := st.chat_input("Haz una pregunta sobre los documentos"):
    if not st.session_state.retriever:
        st.warning("⚠️ Primero debes subir al menos un documento PDF.")
    else:
        # Añadir mensaje de usuario al historial
        st.session_state.messages.append({"role": "user", "content": pregunta})
        with st.chat_message("user"):
            st.markdown(pregunta)

        # Generar respuesta
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            
            # Obtener cadena de RAG
            chain = get_rag_chain(st.session_state.retriever)
            
            with st.spinner("Pensando..."):
                for chunk in chain.stream(pregunta):
                    full_response += chunk.content
                    placeholder.markdown(full_response + "▌")
            
            placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
