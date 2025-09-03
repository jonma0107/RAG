import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

from get_embedding_function import get_embedding_function

# Cargar variables de entorno desde .env
load_dotenv()

CHROMA_PATH = "chroma_db"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=embedding_function,
        collection_name="documents"
    )

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)
    
    # Verificar si hay resultados y si la similitud es suficiente
    if len(results) == 0 or results[0][1] < 0.5:  # Umbral más realista
        print(f"⚠️  Unable to find matching results or similarity too low.")
        print(f"Best score: {results[0][1] if results else 'N/A'}")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    # Verificar si hay API key de OpenAI disponible
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if openai_api_key:
        # Usar OpenAI si hay API key disponible
        model = ChatOpenAI(model="gpt-3.5-turbo")
        print("🔑 Using OpenAI GPT-3.5-turbo")
    else:
        # Fallback a Ollama si no hay API key de OpenAI
        from langchain_community.llms.ollama import Ollama
        model = Ollama(model="mistral")
        print("🖥️  Using local Ollama (no OpenAI API key found)")

    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()