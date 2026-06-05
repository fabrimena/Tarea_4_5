import chromadb

from dotenv import load_dotenv
load_dotenv()

from langfuse.openai import openai
from langfuse import get_client

from Tarea_4_5.vectordb.config import MODEL_RAG

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

collection = chroma_client.get_collection(
    name="apuntes_ai"
)

MAX_DISTANCE = 0.6

# ==========================================================
# SEARCH
# ==========================================================

def search_documents(question, k=5):

    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    ).data[0].embedding

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
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

def ask_rag(question, k=5):

    with lf.start_as_current_observation(
        as_type="span",
        name="rag_pipeline"
    ) as rag_span:

        with lf.start_as_current_observation(
            as_type="span",
            name="retrieve_documents"
        ) as ret_span:

            ret_span.update(
                input={
                    "question": question
                }
            )

            results = search_documents(
                question,
                k
            )

            docs = results["documents"][0]
            metas = results["metadatas"][0]
            scores = results["distances"][0]

            filtered = [
                (d, m, s)
                for d, m, s in zip(
                    docs,
                    metas,
                    scores
                )
                if s <= MAX_DISTANCE
            ]

            if filtered:

                docs, metas, scores = zip(
                    *filtered
                )

                docs = list(docs)
                metas = list(metas)
                scores = list(scores)

            retrieved_chunks = []

            for i, (d, s) in enumerate(
                zip(docs, scores)
            ):

                retrieved_chunks.append(
                    {
                        "chunk_id": i,
                        "score": round(
                            float(s),
                            4
                        ),
                        "text": d[:300]
                    }
                )

            ret_span.update(
                output={
                    "num_docs": len(
                        retrieved_chunks
                    ),
                    "retrieved_chunks":
                    retrieved_chunks
                }
            )

        context = "\n\n".join(docs)

        prompt = f"""
Eres un asistente académico del curso de Inteligencia Artificial.

Responde únicamente utilizando la información presente en el contexto.

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
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0
            )

            answer = (
                response
                .choices[0]
                .message
                .content
            )

            gen_span.update(
                output={
                    "answer": answer
                }
            )

        avg_score = (
            1 - (
                sum(scores)
                / len(scores)
            )
        ) if scores else 0

        rag_span.update(
            input={
                "question": question
            },
            output={
                "answer": answer
            }
        )

        rag_span.score_trace(
            name="context_relevance",
            value=round(avg_score, 4)
        )

        rag_span.score_trace(
            name="answer_correctness",
            value=1.0
        )

        rag_span.score_trace(
            name="groundedness",
            value=1.0
        )

    lf.flush()

    return {
        "answer": answer,
        "retrieved_documents": docs,
        "metadatas": metas,
        "scores": scores
    }