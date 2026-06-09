import json

from langfuse import get_client
from langfuse.openai import openai

from config import MODEL_ORCHESTRATOR, MODEL_SUMMARIZER
from rag import ask_rag, search_documents, collection
from websearch import ask_web_search

client = openai.OpenAI()
lf = get_client()

SYSTEM_PROMPT = """
Eres un agente académico del curso de Inteligencia Artificial.

Dispones de tres herramientas:

1. consultar_apuntes   → úsala cuando el usuario pregunte sobre contenido
                         cubierto en los apuntes del curso.

2. resumir_semana      → úsala cuando el usuario pida un resumen de una
                         semana específica del curso.

3. buscar_en_web       → úsala cuando:
                         a) el usuario lo solicite de forma explícita
                            (ej. "busca en internet", "qué dice la web sobre..."), o
                         b) la información no esté en los apuntes del curso.
                         En caso de duda, intenta primero consultar_apuntes y si
                         la respuesta es insuficiente, usa buscar_en_web
                         automáticamente SIN pedir permiso al usuario.

REGLA IMPORTANTE: Nunca le preguntes al usuario si desea que busques en la web.
Si determinas que la búsqueda web es necesaria, ejecútala directamente y explica tu razonamiento.

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
                    {
                        "type": "function",
                        "function": {
                            "name": "buscar_en_web",
                            "description": (
                                "Realiza una búsqueda en internet. "
                                "Úsala solo cuando el usuario lo solicite explícitamente o cuando "
                                "la información no pueda encontrarse en los apuntes del curso."
                            ),
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "consulta": {
                                        "type":        "string",
                                        "description": "La consulta de búsqueda web a realizar"
                                    }
                                },
                                "required": ["consulta"],
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

            tool_messages = []
            last_tool_result = {
                "answer":              "",
                "retrieved_documents": [],
                "metadatas":           [],
            }

            with lf.start_as_current_observation(as_type="span", name="tool_calls") as tool_span:

                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    args      = json.loads(tool_call.function.arguments)

                    if tool_name == "consultar_apuntes":
                        tool_result  = ask_rag(args["pregunta"])
                        tool_content = tool_result["answer"]
                        last_tool_result = tool_result

                        # ── Fallback: si RAG no encontró nada, forzar búsqueda web ──
                        if "No encontré información suficiente" in tool_content:
                            with lf.start_as_current_observation(as_type="span", name="websearch_fallback"):
                                fallback_result  = ask_web_search(args["pregunta"])
                                tool_content     = fallback_result["answer"]
                                last_tool_result = fallback_result

                    elif tool_name == "resumir_semana":
                        tool_result  = summarize_week(args["semana"])
                        tool_content = tool_result["answer"]
                        last_tool_result = tool_result

                    elif tool_name == "buscar_en_web":
                        tool_result  = ask_web_search(args["consulta"])
                        tool_content = tool_result["answer"]
                        last_tool_result = tool_result

                    else:
                        tool_content = "Herramienta no reconocida"
                        tool_result  = {
                            "answer": tool_content, "retrieved_documents": [], "metadatas": [], "scores": [],
                        }

                    tool_messages.append({
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
                "sources":             last_tool_result.get("metadatas", []),
                "retrieved_documents": last_tool_result.get("retrieved_documents", []),
                "metadatas":           last_tool_result.get("metadatas", []),
            }

        # ── Sin tool call: respuesta directa ────────────────
        agent_span.update(
            output={"answer": message.content[:300] if message.content else ""}
            )
        lf.flush()

        return {
            "answer":              message.content,
            "sources":             [],
            "retrieved_documents": [],
            "metadatas":           [],
        }