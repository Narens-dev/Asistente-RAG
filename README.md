# Asistente-RAG
# 🧠 Overknow: Asistente Académico RAG Multicontexto

Overknow es un motor de búsqueda y asistente conversacional impulsado por Inteligencia Artificial, diseñado para consultar documentos académicos y técnicos de forma aislada y precisa. Utiliza una arquitectura **RAG (Retrieval-Augmented Generation)** respaldada por los modelos de Google Gemini y una base de datos vectorial local.

## 🚀 Características Principales

* **Arquitectura RAG Completa:** Conecta tus propios documentos PDF con el LLM de Google para obtener respuestas precisas basadas *únicamente* en tu contexto, reduciendo alucinaciones.
* **Gestión Multicontexto (Materias):** Interfaz que permite organizar documentos en carpetas separadas (ej. "Redes", "Matemáticas"). El asistente aísla el conocimiento y el historial de chat según la materia seleccionada.
* **Sanitización de Datos:** Limpieza automática de los PDFs entrantes mediante expresiones regulares para ocultar correos electrónicos, identificaciones y números de teléfono antes de vectorizarlos.
* **Tolerancia a Fallos (Resiliencia):** Implementación de reintentos con retraso exponencial (*Exponential Backoff*) utilizando la librería `tenacity` para evadir los límites de cuota de la API (Errores 429).
* **Trazabilidad:** Posibilidad de expandir y auditar los fragmentos exactos del documento original de donde la IA extrajo la respuesta.

## 🛠️ Tecnologías Utilizadas

* **Python 3**
* **LangChain:** Framework central para la orquestación del flujo RAG.
* **Google Gemini API:** 
  * `gemini-1.5-flash` para generación de texto (LLM).
  * `gemini-embedding-001` para vectorización.
* **ChromaDB:** Base de datos vectorial persistente en local (`./chroma_db`).
* **Streamlit:** Framework para la interfaz gráfica web.
* **Tenacity & PyPDF:** Manejo de reintentos y lectura de documentos.

---

## 📸 Capturas de Pantalla


> *Vista de la interfaz principal de chat interactuando con un documento.*
<img width="1919" height="1037" alt="image" src="https://github.com/user-attachments/assets/37937857-fe99-4bd5-8bcc-60b393a13066" />


> *Panel lateral para la carga de PDFs y selección dinámica del contexto.*
<img width="372" height="1005" alt="image" src="https://github.com/user-attachments/assets/ade53a91-108a-4653-9a18-1f2beac8d1ce" />


> *Visualización de los fragmentos recuperados por ChromaDB para generar la respuesta.*
<img width="1348" height="707" alt="image" src="https://github.com/user-attachments/assets/a70738f2-718b-4e03-a16e-57b599aa312d" />

---

## ⚙️ Instalación y Configuración

Sigue estos pasos para ejecutar Overknow en tu entorno local:

### 1. Clonar el repositorio y preparar el entorno
```bash
git clone [https://github.com/TU_USUARIO/overknow.git](https://github.com/TU_USUARIO/overknow.git)
cd overknow
python -m venv venv
Activar el entorno virtual:

En Windows: .\venv\Scripts\activate

En Mac/Linux: source venv/bin/activate

2. Instalar dependencias
Asegúrate de instalar los requerimientos listados en el proyecto:

Bash
pip install -r requirements.txt
3. Configurar variables de entorno
Crea un archivo llamado .env en la raíz del proyecto y añade tu API Key de Google AI Studio:

Fragmento de código
GOOGLE_API_KEY="tu_api_key_aqui"

🚀 Uso y Ejecución
Overknow puede ejecutarse de dos maneras diferentes dependiendo de lo que desees probar:

Opción A: Prueba de Motor Backend (Consola)
Si solo deseas probar el flujo de vectorización y generación en texto plano, puedes ejecutar el script principal app.py.

Nota sobre carpetas: Para que este script funcione, debes crear manualmente una carpeta llamada Documentos_Prueba en la raíz del proyecto y colocar al menos un archivo PDF dentro de ella. El sistema leerá este directorio, creará la base de datos y ejecutará una pregunta de prueba predefinida.

Bash
python app.py
Opción B: Interfaz Gráfica Completa (Recomendada)
Para levantar la aplicación web interactiva y gestionar múltiples documentos y materias de forma visual, utiliza Streamlit.

Nota: Esta interfaz creará automáticamente una carpeta llamada Mis_Documentos donde organizará todo tu material por subcarpetas.

Bash
streamlit run interfaz.py
🏗️ Flujo de Procesamiento Interno (Cómo funciona)
Carga y Sanitización: PyPDFLoader extrae el texto y aplica filtros Regex de privacidad.

Fragmentación: RecursiveCharacterTextSplitter divide el texto en bloques de 500 caracteres con un solapamiento de 50.

Vectorización y Almacenamiento: Los bloques se convierten en embeddings y se guardan en una colección de ChromaDB separada por "Materia".

Recuperación: Al hacer una pregunta, el sistema busca los 5 fragmentos (k=5) más similares semánticamente usando similitud del coseno.

Generación LLM: Un prompt estricto obliga al modelo Gemini a responder usando únicamente la información recuperada, evitando alucinaciones.

Desarrollado por Naren Rojas, David Prieto - Fundación Universitaria Konrad Lorenz.
