import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Importamos las funciones de tu motor backend
from app import consultar_asistente, procesar_documentos, crear_base_vectorial

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# Directorio raíz para organizar las materias
BASE_DIR = "./Mis_Documentos"
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)

st.set_page_config(page_title="Overknow - Asistente RAG", page_icon="📚", layout="wide")

# ==========================================
# CACHÉ: CARGAR BASE DE DATOS DINÁMICA
# ==========================================
@st.cache_resource
def cargar_base_datos(materia):
    DIRECTORIO_DB = "./chroma_db"
    if os.path.exists(DIRECTORIO_DB):
        modelo_embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=API_KEY
        )
        db = Chroma(
            persist_directory=DIRECTORIO_DB, 
            embedding_function=modelo_embeddings,
            collection_name=materia # Carga la colección específica de la materia
        )
        return db
    return None

# ==========================================
# PANEL LATERAL: GESTOR DE ARCHIVOS
# ==========================================
with st.sidebar:
    st.header("📁 Organización")
    st.markdown("Crea carpetas y sube tus PDFs para mantener el contexto separado.")
    
    # 1. Crear nueva carpeta
    st.subheader("1. Nueva Materia")
    nueva_carpeta = st.text_input("Nombre de la carpeta (ej. Matematicas):")
    if st.button("Crear Carpeta") and nueva_carpeta:
        # Formatear nombre para evitar errores (sin espacios)
        nombre_limpio = nueva_carpeta.strip().replace(" ", "_").lower()
        ruta_nueva = os.path.join(BASE_DIR, nombre_limpio)
        os.makedirs(ruta_nueva, exist_ok=True)
        st.success(f"Carpeta '{nombre_limpio}' lista.")
        st.rerun() # Recarga la app para actualizar la lista

    st.divider()

    # 2. Seleccionar carpeta existente
    st.subheader("2. Seleccionar Contexto")
    carpetas = [f for f in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, f))]
    
    if not carpetas:
        st.warning("Crea una carpeta primero para empezar.")
        carpeta_actual = None
    else:
        carpeta_actual = st.selectbox("Carpeta Activa:", carpetas)

    st.divider()

    # 3. Subir y procesar PDFs
    if carpeta_actual:
        st.subheader("3. Subir Documentos")
        archivos_subidos = st.file_uploader(f"Añadir PDFs a '{carpeta_actual}'", type=["pdf"], accept_multiple_files=True)
        
        if st.button("Procesar y Aprender") and archivos_subidos:
            ruta_carpeta = os.path.join(BASE_DIR, carpeta_actual)
            
            with st.spinner('Guardando y vectorizando...'):
                # Guardar los PDFs físicamente
                for archivo in archivos_subidos:
                    ruta_guardado = os.path.join(ruta_carpeta, archivo.name)
                    with open(ruta_guardado, "wb") as f:
                        f.write(archivo.getbuffer())
                
                # Procesar y crear/actualizar vectores
                fragmentos = procesar_documentos(ruta_carpeta)
                if fragmentos:
                    # Pasamos la carpeta actual como nombre de la colección
                    crear_base_vectorial(fragmentos, nombre_coleccion=carpeta_actual)
                    st.success("¡Documentos aprendidos con éxito!")
                    st.cache_resource.clear() # Limpiar caché para recargar la BD fresca
                else:
                    st.error("Hubo un error leyendo los PDFs.")

# ==========================================
# INTERFAZ PRINCIPAL: CHAT MINIMALISTA
# ==========================================
st.title("🧠 Overknow")

if carpeta_actual:
    st.markdown(f"**Contexto activo:** Buscando en la biblioteca de `{carpeta_actual}`")
    
    # Cargar la base de datos de la carpeta seleccionada
    db = cargar_base_datos(carpeta_actual)
    
    # Inicializar el historial de chat específico para esta materia
    historial_key = f"mensajes_{carpeta_actual}"
    if historial_key not in st.session_state:
        st.session_state[historial_key] = []

    # Mostrar historial
    for mensaje in st.session_state[historial_key]:
        with st.chat_message(mensaje["rol"]):
            st.markdown(mensaje["contenido"])

    # Caja de chat
    pregunta_usuario = st.chat_input(f"Pregunta sobre {carpeta_actual}...")

    if pregunta_usuario:
        with st.chat_message("user"):
            st.markdown(pregunta_usuario)
        st.session_state[historial_key].append({"rol": "user", "contenido": pregunta_usuario})

        with st.chat_message("assistant"):
            if db:
                with st.spinner("Analizando documentos..."):
                    try:
                        respuesta_texto, fuentes = consultar_asistente(pregunta_usuario, db)
                        st.markdown(respuesta_texto)
                        
                        if fuentes:
                            with st.expander(f"📄 Ver {len(fuentes)} fuentes extraídas"):
                                for i, doc in enumerate(fuentes):
                                    st.info(f"**Fragmento {i+1}:**\n{doc.page_content}")
                    except Exception as e:
                        respuesta_texto = f"❌ Error al consultar: {e}"
                        st.error(respuesta_texto)
            else:
                respuesta_texto = "Aún no hay documentos procesados en esta carpeta. Sube PDFs en el panel lateral."
                st.warning(respuesta_texto)

        st.session_state[historial_key].append({"rol": "assistant", "contenido": respuesta_texto})
else:
    st.info("👈 Por favor, crea y selecciona una carpeta en el menú lateral para comenzar a interactuar.")