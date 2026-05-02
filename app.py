import os
import re
import sys

# Forzar UTF-8 en la salida estándar para soportar emojis en Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# Cargar variables de entorno (API Key)
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("[ERROR] GOOGLE_API_KEY no encontrada. Verifica el archivo .env")

# ==========================================
# FASE 1.5: SANITIZACIÓN (Regex Local)
# ==========================================
def sanitizar_texto(texto):
    if not texto: return ""
    texto = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[CORREO_OCULTO]', texto)
    texto = re.sub(r'\b\d{7,10}\b', '[ID_OCULTO]', texto)
    texto = re.sub(r'\+?\d{10,12}', '[TELEFONO_OCULTO]', texto)
    return texto

# ==========================================
# FASE 1 Y 2: CARGA Y FRAGMENTACIÓN
# ==========================================
def procesar_documentos(directorio_pdfs):
    documentos_cargados = []
    
    if not os.path.exists(directorio_pdfs):
        os.makedirs(directorio_pdfs)
        return []

    archivos_pdf = [f for f in os.listdir(directorio_pdfs) if f.endswith(".pdf")]
    
    if not archivos_pdf: return []

    print(f"Iniciando carga de {len(archivos_pdf)} documento(s)...")
    
    for archivo in archivos_pdf:
        ruta_completa = os.path.join(directorio_pdfs, archivo)
        loader = PyPDFLoader(ruta_completa)
        paginas = loader.load()
        
        for pagina in paginas:
            pagina.page_content = sanitizar_texto(pagina.page_content)
            
        documentos_cargados.extend(paginas)
        print(f"✅ Cargado y sanitizado: {archivo}")

    print("\nIniciando fragmentación de texto...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       
        chunk_overlap=50,     
        separators=["\n\n", "\n", " ", ""] 
    )
    
    fragmentos = text_splitter.split_documents(documentos_cargados)
    return fragmentos

# ==========================================
# FASE 3 Y 4: VECTORIZACIÓN Y CHROMADB
# ==========================================
# Añadimos 'nombre_coleccion' como parámetro dinámico
def crear_base_vectorial(fragmentos, directorio_db="./chroma_db", nombre_coleccion="general"):
    """
    Convierte fragmentos en embeddings y los guarda en ChromaDB.
    """
    if not fragmentos:
        print("No hay fragmentos para procesar.")
        return None

    print(f"\nIniciando vectorización de {len(fragmentos)} fragmentos...")
    
    modelo_embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=API_KEY
    )

    print(f"Almacenando vectores en ChromaDB (Directorio: {directorio_db})...")
    
    vector_store = Chroma.from_documents(
        documents=fragmentos,
        embedding=modelo_embeddings,
        persist_directory=directorio_db,
        collection_name=nombre_coleccion, # <--- AHORA ES DINÁMICO
        collection_metadata={"hnsw:space": "cosine"} 
    )
    
    print("✅ Base de datos vectorial creada con éxito.")
    return vector_store

# ==========================================
# FASE 5 Y 6: RECUPERACIÓN Y GENERACIÓN LLM
# ==========================================
# Este decorador reintentará la función si falla por cualquier error de conexión
# Esperará 25 segundos entre cada intento y lo hará máximo 3 veces.
@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=20, min=20, max=90), 
    stop=stop_after_attempt(4),
    before_sleep=lambda retry_state: print(f"⏳ Error: {retry_state.outcome.exception()}. Reintentando (Intento {retry_state.attempt_number}/4)...")
)
def consultar_asistente(pregunta, vector_store):
    """
    Busca fragmentos relevantes y genera una respuesta usando Gemini.
    """
    print(f"\nBuscando información para: '{pregunta}'...")
    
    # 1. Configurar el retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    # 2. Definir el Prompt Estricto
    PROMPT_TEMPLATE = """Eres un asistente académico experto.
    Responde la pregunta usando ÚNICAMENTE la información del contexto proporcionado.
    Si la respuesta no está en el contexto, indica exactamente: "No encontré información sobre esto en el documento."

    Contexto recuperado:
    {context}

    Pregunta del usuario: {question}

    Respuesta:"""
    
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    # 3. Configurar el LLM usando el modelo disponible
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest", 
        temperature=0.0,
        google_api_key=API_KEY 
    )

    # 4. Recuperar los fragmentos relevantes
    documentos_recuperados = retriever.invoke(pregunta)
    
    # 5. Ensamblar el texto
    contexto = "\n\n".join([doc.page_content for doc in documentos_recuperados])
    
    # 6. Crear el prompt final
    prompt_aumentado = prompt_template.invoke({
        "context": contexto,
        "question": pregunta
    })

    # 7. Generar la respuesta
    print("\nGenerando respuesta...")
    respuesta = llm.invoke(prompt_aumentado)
    
    # EXTRAER TEXTO LIMPIO
    texto_final = respuesta.content
    if isinstance(texto_final, list): 
        texto_final = texto_final[0].get('text', str(texto_final))
        
    return texto_final, documentos_recuperados

# ==========================================
# BLOQUE PRINCIPAL
# ==========================================
if __name__ == "__main__":
    CARPETA_PDFS = "./Documentos_Prueba"
    DIRECTORIO_DB = "./chroma_db"
    
    # Comprobar si la base de datos ya existe
    if os.path.exists(DIRECTORIO_DB) and os.listdir(DIRECTORIO_DB):
        print(f"\n📂 Cargando base de datos existente desde '{DIRECTORIO_DB}'...")
        # Cargar la base de datos existente sin re-vectorizar
        modelo_embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=API_KEY
        )
        # CARGAR LA COLECCIÓN CORRECTA
        db = Chroma(
            persist_directory=DIRECTORIO_DB, 
            embedding_function=modelo_embeddings,
            collection_name="mi_reglamento"
        )
    else:
        # Si no existe, ejecutar el proceso completo
        print("\n⚙️ No se encontró base de datos. Iniciando proceso completo...")
        mis_fragmentos = procesar_documentos(CARPETA_PDFS)
        if mis_fragmentos:
            db = crear_base_vectorial(mis_fragmentos, DIRECTORIO_DB)
        else:
            db = None

    # Si tenemos la base de datos (nueva o cargada), hacemos la pregunta
    if db:
        pregunta_prueba = "¿Qué significa Term Frequency (TF) y cómo se calcula?"
        
        # Como solo enviamos la pregunta, el consumo de cuota es mínimo
        respuesta_llm, fuentes = consultar_asistente(pregunta_prueba, db)
        
        print("\n" + "="*50)
        print("🤖 RESPUESTA DEL ASISTENTE:")
        print("="*50)
        print(respuesta_llm)
        print("-" * 50)
        print(f"📄 Se utilizaron {len(fuentes)} fragmentos como contexto.")