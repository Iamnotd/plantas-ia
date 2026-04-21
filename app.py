import streamlit as st
import json
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO

st.set_page_config(page_title="🌿 Plantas PRO", layout="wide")
st.title("🌿 Plantas Medicinales IA")
st.info("💎 Versión Cloud • PDF • 100+ plantas")

@st.cache_data
def cargar_plantas():
    with open("plantas.json") as f:
        return json.load(f)

# Sidebar síntomas
sintomas = st.sidebar.selectbox("🔍 Buscar por:", 
    ["", "Estómago", "Ansiedad", "Resfriado", "Dolor", "Insomnio"])

# Chat
if "messages" not in st.session_state: st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

prompt = st.chat_input("Ej: manzanilla, dolor cabeza...")
if prompt or sintomas:
    user_input = sintomas or prompt
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"): st.markdown(user_input)
    
    with st.chat_message("assistant"):
        with st.spinner("🌿 Buscando..."):
            plantas = cargar_plantas()
            textos = [f"🌿{p['nombre']}: {p['uso']}. {p['preparacion']}. ⚠️{p['precaucion']}. {p['dosis']}" 
                     for p in plantas]
            
            splitter = CharacterTextSplitter(chunk_size=300)
            docs = splitter.create_documents(textos)
            embeddings = OllamaEmbeddings(model="llama3.2")
            vectorstore = Chroma.from_documents(docs, embeddings)
            
            llm = Ollama(model="llama3.2", temperature=0.1)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
            
            chain = (
                {"context": retriever, "pregunta": RunnablePassthrough()}
                | lambda x: f"PLANTAS:\n{x['context']}\n\nP: {x['pregunta']}\n\n**Respuesta clara:**\n1. Planta\n2. Uso\n3. Preparar\n4. ⚠️CUIDADO\n5. Dosis",
                | llm
                | StrOutputParser()
            )
            
            respuesta = chain.invoke(user_input)
            st.markdown(respuesta)
            st.session_state.messages.append({"role": "assistant", "content": respuesta})

# PDF + Clear
col1, col2 = st.columns(2)
with col1:
    if st.button("📄 PDF Receta") and st.session_state.messages:
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        c.drawString(100, 750, "🌿 RECETA PLANTAS")
        y, i = 700, 0
        for msg in st.session_state.messages[-3:]:
            c.drawString(50, y, f"{msg['role']}: {msg['content'][:100]}")
            y -= 25
            i += 1
        c.save()
        buffer.seek(0)
        st.download_button("⬇️ Descargar", buffer.getvalue(), "receta.pdf", "application/pdf")
with col2:
    if st.button("🗑️ Nuevo"): st.session_state.messages = []; st.rerun()