import os
import requests

from dotenv import load_dotenv
load_dotenv()

from langfuse.openai import openai
from langfuse import get_client

from config import MODEL_WEB      # modelo ligero, mismo nivel que el RAG

# ==========================================================
# OPENAI + LANGFUSE
# ==========================================================

client = openai.OpenAI()
lf     = get_client()

# ==========================================================
# CONSTANTE: número máximo de resultados que se le pasan al LLM
# ==========================================================

MAX_RESULTS = 5

# ==========================================================
# BÚSQUEDA WEB  (Tavily Search API)
# ----------------------------------------------------------
# Requiere en el .env:
#   TAVILY_API_KEY=tvly-xxxxxxxxxxxx
#
# La función devuelve una lista de dicts con:
#   - title   : título del resultado
#   - url     : URL de la fuente
#   - content : fragmento de texto relevante
# ==========================================================

TAVILY_URL = "https://api.tavily.com/search"


def _raw_search(query: str, max_results: int = MAX_RESULTS) -> list[dict]:
    """
    Llama a la API de Tavily y devuelve los resultados crudos.
    Si la llamada falla devuelve una lista vacía (el agente lo manejará).
    """

    api_key = os.getenv("TAVILY_API_KEY", "")

    if not api_key:
        return []

    payload = {
        "api_key":      api_key,
        "query":        query,
        "max_results":  max_results,
        "search_depth": "basic",
        "include_answer": False,
    }

    try:
        resp = requests.post(TAVILY_URL, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", [])

    except Exception:
        return []


# ==========================================================
# ASK WEB SEARCH
# ----------------------------------------------------------
# Función principal.  Recibe una pregunta, busca en la web,
# construye un contexto y usa el LLM para generar la respuesta.
# ==========================================================

def ask_web_search(question: str) -> dict:
    """
    Realiza una búsqueda web para la pregunta dada y genera
    una respuesta fundamentada en los resultados obtenidos.

    Retorna:
        {
            "answer":   str,
            "sources":  list[dict],   # [{title, url, content}, ...]
            "metadatas": list[dict],  # alias de sources (compatible con run_agent)
            "retrieved_documents": list[str],  # solo los textos
        }
    """

    with lf.start_as_current_observation(
        as_type="span",
        name="websearch_pipeline"
    ) as ws_span:

        ws_span.update(input={"question": question})

        # ── Span 1: búsqueda web ─────────────────────────────
        with lf.start_as_current_observation(
            as_type="span",
            name="web_search_retrieve"
        ) as ret_span:

            ret_span.update(input={"query": question})

            results = _raw_search(question)

            # Truncar a MAX_RESULTS por si acaso
            results = results[:MAX_RESULTS]

            retrieved_chunks = [
                {
                    "title":   r.get("title", ""),
                    "url":     r.get("url", ""),
                    "content": r.get("content", "")[:400],
                }
                for r in results
            ]

            ret_span.update(
                output={
                    "num_results": len(retrieved_chunks),
                    "urls": [c["url"] for c in retrieved_chunks],
                }
            )

        # Sin resultados: respuesta de fallo informativa
        if not retrieved_chunks:
            no_result_answer = (
                "No se encontraron resultados web relevantes para esta consulta. "
                "Por favor reformula la pregunta o intenta con los apuntes del curso."
            )
            ws_span.update(output={"answer": no_result_answer})
            return {
                "answer":              no_result_answer,
                "sources":             [],
                "metadatas":           [],
                "retrieved_documents": [],
            }

        # ── Span 2: generación con LLM ───────────────────────
        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            block = (
                f"[Fuente {i}] {chunk['title']}\n"
                f"URL: {chunk['url']}\n"
                f"{chunk['content']}"
            )
            context_blocks.append(block)

        context = "\n\n".join(context_blocks)

        prompt = f"""Eres un asistente académico del curso de Inteligencia Artificial.
Se realizó una búsqueda web para responder la siguiente pregunta.
Utiliza ÚNICAMENTE la información de los resultados de búsqueda proporcionados.
Si la información no es suficiente, indícalo claramente.
Cita las fuentes relevantes al final de tu respuesta.

RESULTADOS DE BÚSQUEDA:
{context}

PREGUNTA:
{question}
"""

        with lf.start_as_current_observation(
            as_type="span",
            name="websearch_generate_answer"
        ) as gen_span:

            gen_span.update(input={"num_sources": len(retrieved_chunks)})

            response = client.chat.completions.create(
                model=MODEL_WEB,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )

            answer = response.choices[0].message.content

            gen_span.update(output={"answer": answer[:300]})

        ws_span.update(output={"answer": answer[:300]})

        return {
            "answer":              answer,
            "sources":             retrieved_chunks,
            "metadatas":           retrieved_chunks,
            "retrieved_documents": [c["content"] for c in retrieved_chunks],
        }
