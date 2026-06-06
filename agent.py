import json
from langfuse import get_client
from langfuse.openai import openai
from config import MODEL_ORCHESTRATOR, MODEL_SUMMARIZER
from rag import ask_rag, search_documents, collection

client = openai.OpenAI()
lf = get_client()

SYSTEM_PROMPT = """
Eres un agente académico del curso de Inteligencia Artificial.

Dispones de una herramienta llamada RAG.

Utilízala cuando el usuario pregunte sobre contenido de los apuntes.

Responde siempre en español.
"""

# ==========================================================
# SUMMARIZE WEEK  →  usa búsqueda semántica (RAG)
# ==========================================================
def summarize_week(week):
    query = f"principales temas vistos en la semana {week}"

    # Búsqueda semántica CON filtro de metadato semana.
    # Esto garantiza que solo entren chunks de esa semana.
    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=8,
        where={"semana": week},          # ← filtro por metadato
        include=["documents", "metadatas", "distances"]
    )

    docs      = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # Si no hay resultados para esa semana, informar claramente.
    if not docs:
        return {
            "answer":              f"No encontré apuntes de la semana {week} en la base de datos.",
            "retrieved_documents": [],
            "metadatas":           [],
            "scores":              [],
        }

    context = "\n\n".join(docs)
    prompt = f"""
Resume los principales temas vistos en la semana {week}.
Contexto:
{context}
"""
    with lf.start_as_current_observation(as_type="span", name="summarize_week") as span:
        span.update(input={"week": week, "query": query, "num_docs": len(docs)})
        response = client.chat.completions.create(
            model=MODEL_SUMMARIZER,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        summary = response.choices[0].message.content
        span.update(output={"summary": summary[:300]})

    return {
        "answer":              summary,
        "retrieved_documents": docs,
        "metadatas":           metadatas,
        "scores":              distances,
    }

# ==========================================================
# RUN AGENT  →  todo el flujo dentro de UN solo trace.
# El anidamiento es automático por contexto (with statements).
# ==========================================================
def run_agent(question):

    with lf.start_as_current_observation(
        as_type="span",
        name="agent_run",
    ) as agent_span:

        agent_span.update(input={"question": question})

        # ── Span 1: decisión del orquestador ────────────────
        with lf.start_as_current_observation(
            as_type="span",
            name="agent_decision",
        ) as decision_span:

            decision = client.chat.completions.create(
                model=MODEL_ORCHESTRATOR,
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": question},
                ],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "consultar_apuntes",
                            "description": "Busca información dentro de los apuntes del curso",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "pregunta": {"type": "string"}
                                },
                                "required": ["pregunta"],
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "resumir_semana",
                            "description": "Genera un resumen de una semana específica del curso",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "semana": {"type": "integer"}
                                },
                                "required": ["semana"],
                            },
                        },
                    },
                ],
            )

            message = decision.choices[0].message
            decision_span.update(
                output={
                    "tool_called": (
                        message.tool_calls[0].function.name
                        if message.tool_calls
                        else "none"
                    )
                }
            )

        # ── Span 2: ejecución de herramienta ────────────────
        if message.tool_calls:

            tool_messages = []  # collect responses for ALL tool calls

            with lf.start_as_current_observation(
                as_type="span",
                name="tool_calls",
            ) as tool_span:

                for tool_call in message.tool_calls:  # ← loop, not just [0]
                    tool_name = tool_call.function.name
                    args      = json.loads(tool_call.function.arguments)

                    if tool_name == "consultar_apuntes":
                        rag_result   = ask_rag(args["pregunta"])
                        tool_content = rag_result["answer"]
                    elif tool_name == "resumir_semana":
                        rag_result   = summarize_week(args["semana"])
                        tool_content = rag_result["answer"]
                    else:
                        tool_content = "Herramienta no reconocida"
                        rag_result   = {
                            "answer": tool_content,
                            "retrieved_documents": [],
                            "metadatas": [],
                            "scores": [],
                        }

                    tool_messages.append({        # ← one reply per tool call
                        "role":         "tool",
                        "tool_call_id": tool_call.id,
                        "content":      tool_content,
                    })

                tool_span.update(output={"num_tool_calls": len(tool_messages)})

            # ── Span 3: respuesta final del orquestador ─────
            with lf.start_as_current_observation(
                as_type="span",
                name="agent_final_response",
            ):
                final = client.chat.completions.create(
                    model=MODEL_ORCHESTRATOR,
                    temperature=0,
                    messages=[
                        {"role": "system",    "content": SYSTEM_PROMPT},
                        {"role": "user",      "content": question},
                        {
                            "role":       "assistant",
                            "content":    None,
                            "tool_calls": message.tool_calls,  # all tool calls
                        },
                        *tool_messages,   # ← unpack ALL tool responses
                    ],
                )

            answer = final.choices[0].message.content
            agent_span.update(output={"answer": answer[:300]})
            lf.flush()

            return {
                "answer":              answer,
                "sources":             rag_result["metadatas"],
                "retrieved_documents": rag_result["retrieved_documents"],
                "metadatas":           rag_result["metadatas"],
            }

        # ── Sin tool call: respuesta directa ────────────────
        agent_span.update(output={"answer": message.content[:300] if message.content else ""})
        lf.flush()

        return {
            "answer":              message.content,
            "sources":             [],
            "retrieved_documents": [],
            "metadatas":           [],
        }