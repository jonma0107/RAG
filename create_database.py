from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import os
import shutil
import argparse
import hashlib
from dotenv import load_dotenv

from get_embedding_function import get_embedding_function

# Cargar variables de entorno desde .env
load_dotenv()


# Para crear la base de datos de Chroma
CHROMA_DB_PATH = "chroma_db"

DATA_PATH = "data/sena/"

def main():

    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)


def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def calculate_content_hash(text: str) -> str:
    """Calcula un hash SHA-256 del contenido del texto."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def find_duplicate_chunks(chunks: list[Document]) -> tuple[list[Document], list[Document]]:
    """
    Encuentra y separa chunks duplicados basándose en contenido similar.
    Retorna: (chunks_unicos, chunks_duplicados)
    """
    unique_chunks = []
    duplicate_chunks = []
    seen_hashes = set()
    
    for chunk in chunks:
        content_hash = chunk.metadata["content_hash"]
        
        if content_hash in seen_hashes:
            duplicate_chunks.append(chunk)
            print(f"⚠️  Duplicado detectado: {chunk.metadata.get('source', 'unknown')} - {chunk.metadata.get('page', 'unknown')}")
        else:
            seen_hashes.add(content_hash)
            unique_chunks.append(chunk)
    
    return unique_chunks, duplicate_chunks


def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_DB_PATH, embedding_function=get_embedding_function()
    )

    # Calculate Page IDs and content hashes.
    chunks_with_ids = calculate_chunk_ids(chunks)
    
    # Add content hash to metadata
    for chunk in chunks_with_ids:
        chunk.metadata["content_hash"] = calculate_content_hash(chunk.page_content)

    # Detect and remove duplicates from new chunks
    unique_chunks, duplicate_chunks = find_duplicate_chunks(chunks_with_ids)
    
    if duplicate_chunks:
        print(f"🚫 Se encontraron {len(duplicate_chunks)} chunks duplicados que serán ignorados")
        print(f"📝 Se procesarán {len(unique_chunks)} chunks únicos")

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"]) if existing_items["ids"] else set()
    existing_metadatas = existing_items["metadatas"] if existing_items["metadatas"] else []
    
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Create a mapping of existing IDs to their content hashes
    existing_hash_map = {}
    if existing_metadatas:  # Solo procesar si hay metadatos existentes
        for i, metadata in enumerate(existing_metadatas):
            if metadata and "id" in metadata and "content_hash" in metadata:
                existing_hash_map[metadata["id"]] = metadata["content_hash"]

    # Separate chunks into new, updated, and unchanged
    new_chunks = []
    updated_chunks = []
    
    for chunk in unique_chunks:  # Solo procesar chunks únicos
        chunk_id = chunk.metadata["id"]
        chunk_hash = chunk.metadata["content_hash"]
        
        if chunk_id not in existing_ids:
            # New chunk
            new_chunks.append(chunk)
        elif chunk_id in existing_hash_map and existing_hash_map[chunk_id] != chunk_hash:
            # Updated chunk (same ID, different content)
            updated_chunks.append(chunk)
        # If same ID and same hash, chunk is unchanged

    # Process new chunks
    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        print("✅ New documents added successfully")

    # Process updated chunks (delete old, add new)
    if len(updated_chunks):
        print(f"🔄 Updating modified documents: {len(updated_chunks)}")
        updated_chunk_ids = [chunk.metadata["id"] for chunk in updated_chunks]
        
        # Delete old versions
        db.delete(ids=updated_chunk_ids)
        
        # Add updated versions
        db.add_documents(updated_chunks, ids=updated_chunk_ids)
        print("✅ Modified documents updated successfully")

    if not new_chunks and not updated_chunks:
        print("✅ No changes detected - database is up to date")
    else:
        print(f"📊 Summary: {len(new_chunks)} new, {len(updated_chunks)} updated")


def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0
     

    # Podemos recorrer todos los fragmentos y mirar sus metadatos
    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)


if __name__ == "__main__":
    # Ejecutar la función principal
    main()