from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_openai import OpenAIEmbeddings
import os


def get_embedding_function():   
    # Verificar si hay API key de OpenAI disponible
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if openai_api_key:
        # Usar OpenAI si hay API key disponible
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        print("🔑 Using OpenAI embeddings")
    else:
        # Usar Ollama por defecto para desarrollo local
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        print("🖥️  Using local Ollama embeddings (no OpenAI API key found)")
    
    # Alternativa: Amazon Bedrock (requiere permisos de AWS)
    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name="default", region_name="us-east-1"
    # )
    
    return embeddings

