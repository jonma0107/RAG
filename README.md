# Sistema RAG con OpenAI

Este es un sistema de Retrieval-Augmented Generation (RAG) que permite hacer consultas sobre documentos PDF usando embeddings de OpenAI y el modelo GPT-3.5-turbo.

## 🚀 Características

- **Embeddings de OpenAI**: Usa `text-embedding-3-small` para crear representaciones vectoriales de alta calidad
- **Modelo de lenguaje**: Utiliza `gpt-3.5-turbo` para generar respuestas contextuales
- **Base de datos vectorial**: ChromaDB para almacenar y buscar embeddings
- **Fallback automático**: Si no hay API key de OpenAI, usa Ollama local
- **Detección de duplicados**: Identifica y elimina chunks duplicados automáticamente
- **Gestión inteligente**: Solo actualiza documentos modificados

## 📋 Requisitos

- Python 3.8+
- API key de OpenAI (opcional, pero recomendado)
- Ollama instalado (para fallback local)

## 🔧 Instalación

1. **Clonar el repositorio:**
```bash
git clone <tu-repositorio>
cd RAG
```

2. **Crear entorno virtual:**
```bash
python -m venv venv
source venv/bin/activate  # En Linux/Mac
# o
venv\Scripts\activate  # En Windows
```

3. **Instalar dependencias:**
```bash
pip install -r requirements.txt
```

4. **Configurar variables de entorno:**
```bash
cp env_example.txt .env
# Editar .env y agregar tu OPENAI_API_KEY
```

## 🔑 Configuración de OpenAI

1. Ve a [OpenAI Platform](https://platform.openai.com/api-keys)
2. Crea una nueva API key
3. Copia la key en tu archivo `.env`:
```env
OPENAI_API_KEY=sk-...tu-api-key-aqui...
```

## 📁 Estructura de archivos

```
RAG/
├── data/                    # Documentos PDF a procesar
├── chroma_db/              # Base de datos vectorial
├── create_database.py      # Script para crear/actualizar la BD
├── query_data.py           # Script para consultar la BD
├── get_embedding_function.py # Configuración de embeddings
├── test_openai_integration.py # Script de prueba
├── requirements.txt         # Dependencias de Python
└── .env                    # Variables de entorno (crear)
```

## 🚀 Uso

### 1. Preparar documentos

Coloca tus archivos PDF en la carpeta `data/sena/`

### 2. Crear/actualizar la base de datos

```bash
# Crear base de datos por primera vez
python create_database.py

# Actualizar base de datos existente
python create_database.py

# Resetear base de datos completamente
python create_database.py --reset
```

### 3. Hacer consultas

```bash
python query_data.py "¿Cuáles son los deberes del aprendiz?"
```

### 4. Probar la integración

```bash
python test_openai_integration.py
```

## 🔍 Cómo funciona

1. **Carga de documentos**: Los PDFs se cargan y dividen en chunks de 800 caracteres
2. **Embeddings**: Cada chunk se convierte en un vector usando OpenAI
3. **Almacenamiento**: Los vectores se guardan en ChromaDB con metadatos
4. **Búsqueda**: Las consultas se convierten en vectores y se buscan los más similares
5. **Generación**: GPT-3.5-turbo genera respuestas basadas en el contexto encontrado

## ⚙️ Configuración avanzada

### Cambiar modelo de embeddings
Edita `get_embedding_function.py`:
```python
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")  # Modelo más potente
```

### Cambiar modelo de lenguaje
Edita `query_data.py`:
```python
model = ChatOpenAI(model="gpt-4")  # Modelo más avanzado
```

### Ajustar parámetros de chunks
Edita `create_database.py`:
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Chunks más grandes
    chunk_overlap=100,     # Más overlap
)
```

## 🐛 Solución de problemas

### Error: "No module named 'langchain_openai'"
```bash
pip install langchain-openai
```

### Error: "OPENAI_API_KEY not found"
- Verifica que el archivo `.env` existe
- Asegúrate de que `OPENAI_API_KEY` esté configurada
- Ejecuta `python test_openai_integration.py` para verificar

### Fallback a Ollama
Si no hay API key de OpenAI, el sistema automáticamente usa Ollama local. Asegúrate de tener Ollama instalado y ejecutándose.

## 📊 Monitoreo

El sistema muestra información detallada durante la ejecución:
- 🔑 Usando OpenAI embeddings
- 🖥️ Usando Ollama local
- 📝 Chunks procesados
- 🚫 Duplicados detectados
- ✅ Documentos agregados/actualizados

## 🤝 Contribuir

1. Fork el proyecto
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.