from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings


def get_embedding_function():   

    # Usar Ollama por defecto para desarrollo local
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # Alternativa: Amazon Bedrock (requiere permisos de AWS)
    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name="default", region_name="us-east-1"
    # )
    
    return embeddings

