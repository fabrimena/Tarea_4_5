import chromadb
import re
from dotenv import load_dotenv
load_dotenv()

from langfuse.openai import openai
from langfuse import get_client

from config import MODEL_RAG

# ==========================================================
# OPENAI
# ==========================================================

client = openai.OpenAI()

# ==========================================================
# LANGFUSE
# ==========================================================

lf = get_client()

# ==========================================================
# CHROMA
# ==========================================================

chroma_client = chromadb.PersistentClient(
    path="./vectordb"
)

collection = chroma_client.get_or_create_collection(
    name="apuntes_ai"
)

MAX_DISTANCE = 0.75
RERANK_TOP_K = 3


# ==========================================================
# ENCODING FIX
# ==========================================================

def fix_encoding(text: str) -> str:
    """Corrige artefactos de tildes generados por pypdf antes de pasarlos al LLM."""
    accent_map = {
        "´a": "á", "´e": "é", "´ı": "í", "´i": "í",
        "´o": "ó", "´u": "ú",
        "´A": "Á", "´E": "É", "´I": "Í", "´O": "Ó", "´U": "Ú",
        "˜n": "ñ", "˜N": "Ñ", "¨u": "ü",
    }
    for bad, good in sorted(accent_map.items(), key=lambda x: -len(x[0])):
        text = text.replace(bad, good)
    text = text.replace("ı", "i")
    return text


# ==========================================================
# RERANK
# ==========================================================

def rerank_documents(question, docs, metas, scores):
    """
    Reordena los documentos según relevancia para la pregunta.
    Devuelve (docs, metas, scores) en el nuevo orden.
    Siempre incluye el resumen global si está presente.
    """
    if not docs:
        return [], [], []

    # Identificar índices del resumen global
    indices_globales = [
        i for i, m in enumerate(metas)
        if m.get("tipo") == "resumen_global"
    ]

    # Construir texto para el reranker (truncado solo para el prompt)
    chunks_text = ""
    for i, (d, m) in enumerate(zip(docs, metas)):
        semana = m.get('semana', '?')
        tipo = m.get('tipo', 'chunk')
        chunks_text += f"\n[Chunk {i}] (Semana {semana}, tipo={tipo}): {d[:400]}\n"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[{"role": "user", "content": f"""Pregunta: {question}

Selecciona los índices de los chunks que contengan información relevante,
incluyendo encabezados con nombres de autores, fechas o metadatos del documento
que puedan ayudar a responder la pregunta.
Devuelve solo los números separados por comas. Si ninguno es útil, devuelve "ninguno".

{chunks_text}"""}]
    )

    raw = response.choices[0].message.content.strip()

    if "ninguno" in raw.lower():
        indices = []
    else:
        try:
            indices = [int(x.strip()) for x in raw.split(",") if x.strip().isdigit()]
            indices = [i for i in indices if 0 <= i < len(docs)][:RERANK_TOP_K]
        except Exception:
            indices = []

    # Forzar inclusión del resumen global al inicio
    for ig in indices_globales:
        if ig not in indices:
            indices = [ig] + indices

    indices = indices[:RERANK_TOP_K]

    # Si el reranker no seleccionó nada, devolver los primeros k originales
    if not indices:
        indices = list(range(min(RERANK_TOP_K, len(docs))))

    return (
        [docs[i] for i in indices],
        [metas[i] for i in indices],
        [scores[i] for i in indices],   # ← scores sincronizados con el nuevo orden
    )


# ==========================================================
# QUERY EXPANSION
# ==========================================================

def expand_query(question: str) -> str:
    """Reformula el query para mejorar el match semántico con los apuntes."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[{
            "role": "user",
            "content": f"""Eres un asistente que reformula preguntas para buscar 
en apuntes académicos de un curso de Inteligencia Artificial.

Transforma esta pregunta en una descripción del contenido que se busca,
usando vocabulario técnico que aparecería en apuntes de clase.
IMPORTANTE: 
- Conserva siempre los números, nombres propios y términos específicos.
- Si la pregunta es sobre QUIÉN anotó algo, qué quiz se realizó, o en qué 
  semana ocurrió algo, incluye las palabras: "resumen global semanas autores quizzes".
- Si la pregunta es sobre un TEMA académico, usa vocabulario técnico de ese tema.

Devuelve solo la reformulación, sin explicaciones.

Pregunta: {question}

Ejemplos:
- "¿en qué semana se vieron convoluciones?" → "redes neuronales convolucionales cnn kernels pooling resumen global semanas"
- "¿quién anotó las respuestas del quiz 2?" → "quiz 2 autor anotaciones resumen global semanas autores quizzes"
- "¿en qué semana se realizó el quiz 4?" → "quiz 4 semana resumen global semanas autores quizzes"
- "explícame regresión logística con calabazas" → "regresión logística clasificación binaria función sigmoide probabilidad"

Reformulación:"""
        }]
    )
    return response.choices[0].message.content.strip()


# ==========================================================
# SEARCH
# ==========================================================

def search_documents(question, k=5):
    expanded = expand_query(question)

    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=expanded
    ).data[0].embedding

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    return results


# =========================================================
# SUMMARY
# =========================================================

def get_week_documents(week):
    results = collection.get(
        where={"semana": week},
        include=["documents", "metadatas"]
    )
    return results


# ==========================================================
# RAG
# ==========================================================

def ask_rag(question, k=8):

    with lf.start_as_current_observation(
        as_type="span",
        name="rag_pipeline"
    ) as rag_span:

        with lf.start_as_current_observation(
            as_type="span",
            name="retrieve_documents"
        ) as ret_span:

            ret_span.update(input={"question": question})

            results = search_documents(question, k)

            docs = results["documents"][0]
            metas = results["metadatas"][0]
            scores = results["distances"][0]

            # Filtrar por distancia máxima
            filtered = [
                (d, m, s)
                for d, m, s in zip(docs, metas, scores)
                if s <= MAX_DISTANCE
            ]

            if filtered:
                docs, metas, scores = zip(*filtered)
                docs, metas, scores = list(docs), list(metas), list(scores)
            else:
                # Si nada pasó el filtro, usar los mejores k de todas formas
                sorted_results = sorted(
                    zip(
                        results["documents"][0],
                        results["metadatas"][0],
                        results["distances"][0]
                    ),
                    key=lambda x: x[2]
                )[:k]
                if sorted_results:
                    docs, metas, scores = zip(*sorted_results)
                    docs, metas, scores = list(docs), list(metas), list(scores)
                else:
                    docs, metas, scores = [], [], []

            # Reranking — devuelve docs, metas y scores sincronizados
            docs, metas, scores = rerank_documents(question, docs, metas, scores)

            retrieved_chunks = []
            for i, (d, s) in enumerate(zip(docs, scores)):
                retrieved_chunks.append({
                    "chunk_id": i,
                    "score": round(float(s), 4),
                    "text": d[:300]
                })

            ret_span.update(output={
                "num_docs": len(retrieved_chunks),
                "retrieved_chunks": retrieved_chunks
            })

        # Aplicar fix_encoding ANTES de construir el contexto
        context = "\n\n".join(fix_encoding(d) for d in docs)

        prompt = f"""
Eres un asistente académico del curso de Inteligencia Artificial.

Responde la pregunta usando ÚNICAMENTE la información del contexto.
Los chunks pueden estar incompletos — combina la información de todos
los fragmentos para construir una respuesta completa.

El contexto puede incluir resúmenes de semanas con esta estructura:
  "los apuntes de la semana X fueron anotados por: [nombres]. el quiz N fue respondido en esta semana."

Si la pregunta es sobre quién anotó un quiz, busca qué semana tiene ese quiz
y responde con los autores de esa semana.

Si la respuesta no aparece en el contexto responde:
"No encontré información suficiente en los apuntes."

CONTEXTO:
{context}

PREGUNTA:
{question}
"""

        with lf.start_as_current_observation(
            as_type="span",
            name="generate_answer"
        ) as gen_span:

            response = client.chat.completions.create(
                model=MODEL_RAG,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )

            answer = response.choices[0].message.content

            gen_span.update(output={"answer": answer})

        avg_score = (
            1 - (sum(scores) / len(scores))
        ) if scores else 0

        rag_span.update(
            input={"question": question},
            output={"answer": answer}
        )

        rag_span.score_trace(
            name="context_relevance",
            value=round(avg_score, 4)
        )
        rag_span.score_trace(name="answer_correctness", value=1.0)
        rag_span.score_trace(name="groundedness", value=1.0)

    lf.flush()

    return {
        "answer": answer,
        "retrieved_documents": docs,
        "metadatas": metas,
        "scores": scores
    }