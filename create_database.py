from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import os
import shutil
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()


# Para crear la base de datos de Chroma
CHROMA_DB_PATH = "chroma_db"

DATA_PATH = "data"

def load_documents():
    loader = DirectoryLoader("data", glob="*.pdf")
    documents = loader.load()
    return documents

# SEPARADOR DE TEXTO RECURSIVO POR CARACTERES   
def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Chunks más pequeños = menos tokens
        chunk_overlap=100,  # Menos overlap = menos redundancia
        length_function=len,
        add_start_index=True,
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks!!!")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)
    # print(document.metadata["source"])
    return chunks

# CONVERTIR LOS FRAGMENTOS DE LOS DOCUMENTOS CARGADOS EN UNA BASE DE DATOS CHROMA

def create_database(chunks: list[Document]):
    # Si la base de datos ya existe, la eliminamos
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)
    # Creamos la base de datos
    db = Chroma.from_documents(
        chunks,
        OpenAIEmbeddings(model="text-embedding-3-small"), # Usando el nuevo modelo
        persist_directory=CHROMA_DB_PATH,
        collection_name="documents",
    )
    # Verificar que se guardaron los documentos
    print("Base de datos creada correctamente")
    print(f"Saved {len(chunks)} chunks in the database {CHROMA_DB_PATH}")
    print(f"Database collection count: {db._collection.count()}")
    
    # Forzar persistencia explícita si el método existe
    try:
        db.persist()
        print("Database persisted explicitly")
    except AttributeError:
        print("persist() method not available - using automatic persistence")


# Cargar la base de datos de Chroma
def load_database():
    db = Chroma(
        persist_directory=CHROMA_DB_PATH, 
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
        collection_name="documents"  # Especificar el mismo nombre de colección
    )
    return db


if __name__ == "__main__":
    print("Loading documents...")
    documents = load_documents()
    print(f"Loaded {len(documents)} documents")
    
    print("Splitting documents...")
    chunks = split_text(documents)
    print("Document processing completed!")
    
    print("Creating database...")
    create_database(chunks)
    print("Database created successfully")

    # Consultar la base de datos de Chroma
    print("Loading database...")
    print(f"Database directory exists: {os.path.exists(CHROMA_DB_PATH)}")
    if os.path.exists(CHROMA_DB_PATH):
        print(f"Database directory contents: {os.listdir(CHROMA_DB_PATH)}")
    
    db = load_database()
    print(f"Database loaded. Collection count: {db._collection.count()}")
    
    query_text = "CUALES SON LOS DEBERES DEL APRENDIZ?"
    print(f"Searching for: '{query_text}'")
    results = db.similarity_search_with_score(query_text, k=5)
    
    # Si no hay resultados, probar con una consulta más simple
    if len(results) == 0:
        print("No results found. Trying simpler query...")
        simple_query = "aprendiz"
        results = db.similarity_search_with_score(simple_query, k=5)
        print(f"Simple query results: {len(results)}")
    
    # Verificaciones antes de procesar los resultados
    print(f"Raw results count: {len(results)}")
    if len(results) > 0:
        print(f"Best score: {results[0][1]:.3f}")
        print(f"All scores: {[score for _, score in results]}")
    
    if len(results) == 0 or results[0][1] < 0.5:  # Umbral más realista
        print(f"Unable to find matching results.")
        print("Debug: Showing top results anyway:")
        for i, (document, score) in enumerate(results[:3]):  # Mostrar top 3
            print(f"Result {i+1} - Score: {score:.3f}")
            print(f"Content: {document.page_content[:500]}...")  # Mostrar 500 caracteres
            print(f"Source: {document.metadata.get('source', 'Unknown')}")
            print("-" * 50)
    else:
        print(f"Found {len(results)} results with scores:")
        for i, (document, score) in enumerate(results):
            print(f"Result {i+1} - Score: {score:.3f}")
            print(f"Content: {document.page_content[:200]}...")
            print(f"Source: {document.metadata.get('source', 'Unknown')}")
            print("-" * 50)