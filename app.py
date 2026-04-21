import streamlit as st
import json
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import speech_recognition as sr
import pyttsx3

# Cargar plantas
@st.cache_data
def cargar_plantas():
    with open("plantas.json", "r", encoding="utf-8") as f:
        return json.load(f)

st.set_page_config(page_title="🌿 Plantas PRO", layout="wide", initial_sidebar_state="expanded")

# Header ÉPICO
st.markdown("""
# 🌿 **Plantas Medicinales IA PRO**
**¡Tu herbolario inteligente!** 
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1: st.success("✅ 100+ plantas")
with col2: st.info("🎤 Voz ON")
with col3: st.warning("⚠️ Consulta médico")

# Sidebar PRO
with st.sidebar:
    st.header("🔍 Búsqueda rápida")
    sintoma = st.selectbox("Elige síntoma:", 
        ["", "Estómago", "Ansiedad", "Resfriado", "Dolor cabeza", "Insomnio"])
    if sintoma:
        st.session_state.quick_search = sintoma

# Chat ÉPICO
if "messages" not in st.session_state:
    st.session_state.messages = []

# Voice input
colv1, colv2 = st.columns(2)
with colv1:
    if st.button("🎤 Hablar", type="primary"):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("Habla ahora...")
            audio = r.listen(source, timeout=5)
            try:
                prompt = r.recognize_google(audio, language="es-ES")
                st.session_state.messages.append({"role": "user", "content": prompt})
            except:
                st.error("No entendí :(")

# Mostrar chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input + búsqueda rápida
prompt = st.chat_input("💬 Escribe o habla...")
if prompt or "quick_search" in st.session_state:
    if "quick_search" in st.session_state:
        user_input = st.session_state.quick_search
        del st.session_state.quick_search
    else:
        user_input = prompt
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"): st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🌿 Analizando..."):
            plantas = cargar_plantas()
            textos = [f"🌿{p['nombre']}: {p['uso']}. Preparar: {p['preparacion']}. ⚠️{p['precaucion']}. {p['dosis']}" 
                     for p in plantas]
            
            splitter = CharacterTextSplitter(chunk_size=400)
            docs = splitter.create_documents(textos)
            embeddings = OllamaEmbeddings(model="llama3.2")
            vectorstore = Chroma.from_documents(docs, embeddings)
            
            llm = Ollama(model="llama3.2", temperature=0.1)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            
            chain = (
                {"context": retriever, "pregunta": RunnablePassthrough()}
                | lambda x: f"""PLANTAS MEDICINALES:
                {x['context']}

                PREGUNTA: {x['pregunta']}
                
                **RESPUESTA PERFECTA:**
                1. **🌿 Planta recomendada**
                2. **🎯 Para qué sirve**
                3. **🥄 Cómo preparar** 
                4. **⚠️ PRECAUCIONES OBLIGATORIAS**
                5. **📏 Dosis exacta**
                6. **🔬 Evidencia** (si aplica)
                """
                | llm
                | StrOutputParser()
            )
            
            respuesta = chain.invoke(user_input)
            st.markdown(respuesta)
            st.session_state.messages.append({"role": "assistant", "content": respuesta})

# Footer
st.markdown("---")
st.caption("🤖 Creada con Ollama + Streamlit | Actualiza `plantas.json` para más")