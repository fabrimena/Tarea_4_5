import json

from langfuse import get_client
from langfuse.openai import openai
from config import MODEL_ORCHESTRATOR, MODEL_SUMMARIZER, MODEL_TRANSACTIONAL
from rag import ask_rag, search_documents, collection
from mcp_client import call_mcp_tool
from websearch import ask_web_search
from memory import (
    search_history,
    get_recent_sessions,
    get_all_messages_today,
    get_session_messages,
)

client = openai.OpenAI()
lf = get_client()

SYSTEM_PROMPT = """
Eres un agente académico del curso de Inteligencia Artificial.

Dispones de cinco herramientas:

1. consultar_apuntes        → úsala cuando el usuario pregunte sobre contenido
                            cubierto en los apuntes del curso.

2. resumir_semana           → úsala cuando el usuario pida un resumen de una
                            semana específica del curso.

3. consultar_transacciones  → úsala solo para preguntas sobre transacciones,
                            fraude o clientes ficticios.

4. buscar_en_web            → SOLO úsala en estos dos casos:
                            a) El usuario lo solicita EXPLÍCITAMENTE con frases como
                            "busca en internet", "qué dice la web", "busca en línea".
                            b) Consultaste primero consultar_apuntes, no encontraste
                            información suficiente, Y el usuario confirmó que desea
                            la búsqueda web cuando se le preguntó.                    
                            NUNCA uses buscar_en_web directamente sin haber intentado
                            consultar_apuntes antes, aunque el tema parezca externo al curso.

5. consultar_memoria        → úsala cuando el usuario pregunte sobre:
                            - preguntas anteriores realizadas en esta u otras sesiones
                            - respuestas previas del agente
                            - resúmenes de sesiones pasadas
                            - si ya se consultó algo anteriormente
                            Ejemplos: "¿qué pregunté antes?", "¿ya consultamos X?",
                            "resume esta sesión", "¿qué dijiste sobre Y antes?"
                            - mensajes_usuario: para "¿sobre qué he consultado?", 
                            "busca en todas mis sesiones", "¿qué pregunté antes?"
                            - buscar: para buscar un tema específico en el historial,
                            ej: "¿ya pregunté sobre convoluciones?"
                            - sesiones_recientes: para "¿qué hice en sesiones anteriores?"
                            - hoy: para "¿qué consulté hoy?"
                            - sesion_actual: para "resume esta conversación"

Tienes acceso al historial de la conversación actual en el contexto.
Úsalo para mantener coherencia entre preguntas consecutivas — por ejemplo,
si el usuario responde "sí" a una pregunta que hiciste, entiende a qué se refiere.

Responde siempre en español y explica qué datos usaste.
"""

# ==========================================================
# SUMMARIZE WEEK
# ==========================================================

def summarize_week(week):
    query = f"principales temas vistos en la semana {week}"

    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=8,
        where={"semana": week},
        include=["documents", "metadatas", "distances"]
    )

    docs      = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

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
# TRANSACTIONAL AGENT
# ==========================================================

def run_transactional_agent(args: dict) -> dict:
    accion = args["accion"]
    justificacion = args["justificacion"]

    mcp_args = {"justification": justificacion}
    optional_fields = [
        "transaction_id", "customer_id", "days", "reason", "severity",
        "status", "is_flagged", "min_amount", "max_amount", "start_date", "end_date",
    ]
    for field in optional_fields:
        if field in args and args[field] is not None:
            mcp_args[field] = args[field]

    with lf.start_as_current_observation(as_type="span", name="transactional_agent") as span:
        span.update(input={"accion": accion, "justificacion": justificacion})

        with lf.start_as_current_observation(as_type="span", name=f"mcp_call:{accion}") as mcp_span:
            mcp_span.update(input=mcp_args)
            try:
                mcp_result = call_mcp_tool(accion, mcp_args)
            except Exception as exc:
                mcp_result = {"error": str(exc)}
            mcp_span.update(output=mcp_result)

        prompt = f"""
Eres el Agente Transaccional. Interpreta el resultado del MCP y responde al usuario.
Indica qué datos consultaste y si hubo restricciones de seguridad.

Pregunta original:
{args.get('pregunta_original', '')}

Resultado MCP ({accion}):
{json.dumps(mcp_result, ensure_ascii=False, indent=2)}
"""
        response = client.chat.completions.create(
            model=MODEL_TRANSACTIONAL,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response.choices[0].message.content
        span.update(output={"answer": answer[:300]})

    return {
        "answer": answer,
        "mcp_result": mcp_result,
        "retrieved_documents": [],
        "metadatas": [{"source": "mcp", "tool": accion, "data_used": mcp_result.get("data_used", [])}],
        "scores": [],
    }


# ==========================================================
# MEMORY TOOL
# ==========================================================

def run_memory_query(args: dict, session_id: int) -> dict:
    """
    Consulta la memoria histórica y/o de sesión según el tipo de consulta.
    """
    tipo = args.get("tipo", "buscar")
    query = args.get("query", "")

    with lf.start_as_current_observation(as_type="span", name="memory_query") as span:
        span.update(input={"tipo": tipo, "query": query})

        if tipo == "sesion_actual":
            # Devolver historial de la sesión actual
            messages = get_session_messages(session_id)
            if not messages:
                content = "No hay mensajes previos en esta sesión."
            else:
                lines = [f"[{m['role']}]: {m['content']}" for m in messages]
                content = "Historial de esta sesión:\n" + "\n".join(lines)

        elif tipo == "hoy":
            # Mensajes de hoy en todas las sesiones
            messages = get_all_messages_today()
            if not messages:
                content = "No hay conversaciones registradas hoy."
            else:
                lines = [f"[Sesión {m['session_id']} - {m['role']}]: {m['content']}" for m in messages]
                content = "Conversaciones de hoy:\n" + "\n".join(lines)

        elif tipo == "sesiones_recientes":
            # Resumen de sesiones anteriores
            sessions = get_recent_sessions(limit=5)
            if not sessions:
                content = "No hay sesiones anteriores registradas."
            else:
                parts = []
                for s in sessions:
                    resumen = s["summary"] or "sin resumen"
                    n_msgs = len(s["messages"])
                    parts.append(f"Sesión {s['session_id']} ({s['started_at'][:10]}): {n_msgs} mensajes. {resumen}")
                content = "Sesiones recientes:\n" + "\n".join(parts)

        elif tipo == "mensajes_usuario":
            from memory import get_user_messages
            messages = get_user_messages(exclude_session_id=session_id, limit=30)
            if not messages:
                content = "No hay preguntas anteriores registradas en otras sesiones."
            else:
                lines = [
                    f"[Sesión {m['session_id']} - {m['timestamp'][:10]}]: {m['content']}"
                    for m in messages
                ]
                content = "Preguntas realizadas en sesiones anteriores:\n" + "\n".join(lines)
        else:
            results = search_history(query, limit=5)
            if not results:
                content = f"No encontré mensajes anteriores relacionados con '{query}'."
            else:
                lines = [
                    f"[{r['timestamp'][:10]} - {r['role']}]: {r['content'][:200]}"
                    for r in results
                ]
                content = f"Mensajes anteriores sobre '{query}':\n" + "\n".join(lines)

        span.update(output={"content": content[:300]})

    return {
        "answer": content,
        "retrieved_documents": [],
        "metadatas": [],
        "scores": [],
    }


# ==========================================================
# RUN AGENT
# ==========================================================

def run_agent(question: str, chat_history: list[dict] = None, session_id: int = None) -> dict:
    """
    Ejecuta el agente con memoria de sesión e histórica.

    Args:
        question:     Pregunta del usuario.
        chat_history: Lista de mensajes previos de esta sesión
                      [{"role": "user"|"assistant", "content": "..."}]
        session_id:   ID de la sesión actual en la BD de memoria.
    """
    if chat_history is None:
        chat_history = []

    with lf.start_as_current_observation(
        as_type="span",
        name="agent_run",
    ) as agent_span:

        agent_span.update(input={"question": question})

        # Construir mensajes con historial de sesión
        messages_for_llm = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + chat_history
            + [{"role": "user", "content": question}]
        )

        # ── Span 1: decisión del orquestador ────────────────
        with lf.start_as_current_observation(
            as_type="span",
            name="agent_decision",
        ) as decision_span:

            decision = client.chat.completions.create(
                model=MODEL_ORCHESTRATOR,
                temperature=0,
                messages=messages_for_llm,
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
                            "name": "consultar_transacciones",
                            "description": "Consulta transacciones ficticias, fraude o riesgo de clientes vía MCP Server",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "accion": {
                                        "type": "string",
                                        "enum": [
                                            "get_transaction_by_id",
                                            "search_transactions",
                                            "get_customer_risk_summary",
                                            "get_recent_flagged_transactions",
                                            "create_fraud_case",
                                        ],
                                    },
                                    "justificacion": {
                                        "type": "string",
                                        "description": "Por qué se necesita esta consulta (mínimo 10 caracteres)",
                                    },
                                    "transaction_id": {"type": "integer"},
                                    "customer_id":    {"type": "integer"},
                                    "days":           {"type": "integer"},
                                    "reason":         {"type": "string"},
                                    "severity": {
                                        "type": "string",
                                        "enum": ["low", "medium", "high", "critical"],
                                    },
                                    "status": {
                                        "type": "string",
                                        "enum": ["approved", "failed", "pending"],
                                    },
                                    "is_flagged":  {"type": "boolean"},
                                    "min_amount":  {"type": "number"},
                                    "max_amount":  {"type": "number"},
                                    "start_date":  {"type": "string"},
                                    "end_date":    {"type": "string"},
                                },
                                "required": ["accion", "justificacion"],
                            }
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "buscar_en_web",
                            "description": (
                                            "Realiza una búsqueda en internet. "
                                            "SOLO debe usarse si: (a) el usuario lo pidió explícitamente con palabras "
                                            "como 'busca en internet' o 'qué dice la web', O (b) ya se intentó "
                                            "consultar_apuntes, no se encontró información, y el usuario confirmó "
                                            "que desea la búsqueda web. NUNCA usar como primera opción."
                                        ),
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "consulta": {
                                        "type": "string",
                                        "description": "La consulta de búsqueda web a realizar"
                                    }
                                },
                                "required": ["consulta"],
                            },
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "consultar_memoria",
                            "description": (
                                "Consulta el historial de conversaciones anteriores. "
                                "Úsala para preguntas sobre qué se preguntó antes, "
                                "resúmenes de sesiones pasadas, o si ya se consultó algo."
                            ),
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "tipo": {
                                        "type": "string",
                                        "enum": [
                                            "buscar",
                                            "sesion_actual",
                                            "hoy",
                                            "sesiones_recientes",
                                            "mensajes_usuario",
                                        ],
                                        "description": (
                                            "buscar: busca por texto en el historial. "
                                            "sesion_actual: mensajes de esta sesión. "
                                            "hoy: todo lo consultado hoy. "
                                            "sesiones_recientes: resumen de últimas sesiones."
                                            "mensajes_usuario: TODAS las preguntas del usuario en sesiones anteriores, "
                                            "sin filtro. Úsalo cuando pregunten '¿sobre qué he consultado?', "
                                            "'¿qué pregunté antes?', 'busca en todas las sesiones'."
                                        )
                                    },
                                    "query": {
                                        "type": "string",
                                        "description": "Texto a buscar (solo para tipo=buscar)"
                                    }
                                },
                                "required": ["tipo"],
                            },
                        }
                    },
                ],
            )

            message = decision.choices[0].message
            decision_span.update(
                output={
                    "tool_called": (
                        message.tool_calls[0].function.name
                        if message.tool_calls else "none"
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
                        tool_result  = ask_rag(question)
                        tool_content = tool_result["answer"]
                        last_tool_result = tool_result

                        if "No encontré información suficiente" in tool_content:
                            tool_content = (
                                "No encontré información sobre esto en los apuntes del curso. "
                                "¿Deseas que realice una búsqueda en internet?"
                            )

                    elif tool_name == "resumir_semana":
                        tool_result  = summarize_week(args["semana"])
                        tool_content = tool_result["answer"]
                        last_tool_result = tool_result

                    elif tool_name == "consultar_transacciones":
                        args["pregunta_original"] = question
                        tool_result  = run_transactional_agent(args)
                        tool_content = tool_result["answer"]

                    elif tool_name == "buscar_en_web":
                        tool_result  = ask_web_search(args["consulta"])
                        tool_content = tool_result["answer"]
                        last_tool_result = tool_result

                    elif tool_name == "consultar_memoria":
                        tool_result  = run_memory_query(args, session_id)
                        tool_content = tool_result["answer"]
                        last_tool_result = tool_result

                    else:
                        tool_content = "Herramienta no reconocida"
                        tool_result  = {
                            "answer": tool_content,
                            "retrieved_documents": [],
                            "metadatas": [],
                            "scores": [],
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
                        *messages_for_llm,
                        {
                            "role":       "assistant",
                            "content":    None,
                            "tool_calls": message.tool_calls,
                        },
                        *tool_messages,
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