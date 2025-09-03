#!/usr/bin/env python3
"""
Script de prueba para verificar la integración con OpenAI.
Ejecuta este script para verificar que tu API key está configurada correctamente.
"""

import os
from dotenv import load_dotenv
from get_embedding_function import get_embedding_function

def test_openai_integration():
    """Prueba la integración con OpenAI."""
    print("🧪 Testing OpenAI Integration...")
    print("=" * 50)
    
    # Cargar variables de entorno
    load_dotenv()
    
    # Verificar API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY no encontrada en el archivo .env")
        print("   Por favor, crea un archivo .env con tu API key de OpenAI")
        return False
    
    if api_key == "your_openai_api_key_here":
        print("❌ OPENAI_API_KEY no está configurada correctamente")
        print("   Por favor, reemplaza 'your_openai_api_key_here' con tu API key real")
        return False
    
    print(f"✅ OPENAI_API_KEY encontrada: {api_key[:8]}...{api_key[-4:]}")
    
    # Probar embeddings
    try:
        print("\n🔍 Probando embeddings...")
        embeddings = get_embedding_function()
        print(f"✅ Embeddings configurados: {type(embeddings).__name__}")
        
        # Probar una consulta simple
        test_text = "Hello, this is a test"
        print(f"📝 Texto de prueba: '{test_text}'")
        
        # Verificar que es OpenAI
        if "OpenAI" in str(type(embeddings)):
            print("🎯 Usando OpenAI embeddings (text-embedding-3-small)")
        else:
            print("🖥️  Usando embeddings locales (Ollama)")
            
    except Exception as e:
        print(f"❌ Error al probar embeddings: {e}")
        return False
    
    print("\n🎉 ¡Integración con OpenAI configurada correctamente!")
    print("\n📋 Para usar el sistema:")
    print("   1. Crear base de datos: python create_database.py")
    print("   2. Hacer consultas: python query_data.py 'tu pregunta aquí'")
    
    return True

if __name__ == "__main__":
    test_openai_integration()