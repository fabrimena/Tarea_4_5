import json
from langfuse import get_client
from langfuse.openai import openai
from config import MODEL_ORCHESTRATOR, MODEL_SUMMARIZER, MODEL_TRANSACTIONAL
from rag import ask_rag, search_documents, collection
from mcp_client import call_mcp_tool

client = openai.OpenAI()
lf = get_client()

SYSTEM_PROMPT = """
Eres un agente académico del curso de Inteligencia Artificial.

Herramientas disponibles:
- consultar_apuntes: preguntas sobre contenido del curso (RAG).
- resumir_semana: resumen de una semana específica.
- consultar_transacciones: datos transaccionales ficticios vía MCP (fraude, clientes, transacciones).

Usa consultar_transacciones solo para preguntas sobre transacciones, fraude o clientes ficticios.
Usa consultar_apuntes para temas académicos del curso.

Responde siempre en español y explica qué datos usaste.
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
# TRANSACTIONAL AGENT  →  consulta vía MCP (sin acceso directo a BD)
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
                                    "customer_id": {"type": "integer"},
                                    "days": {"type": "integer"},
                                    "reason": {"type": "string"},
                                    "severity": {
                                        "type": "string",
                                        "enum": ["low", "medium", "high", "critical"],
                                    },
                                    "status": {
                                        "type": "string",
                                        "enum": ["approved", "failed", "pending"],
                                    },
                                    "is_flagged": {"type": "boolean"},
                                    "min_amount": {"type": "number"},
                                    "max_amount": {"type": "number"},
                                    "start_date": {"type": "string"},
                                    "end_date": {"type": "string"},
                                },
                                "required": ["accion", "justificacion"],
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
            tool_name = message.tool_calls[0].function.name
            args      = json.loads(message.tool_calls[0].function.arguments)

            print("DEBUG - Tool name:", tool_name)
            print("DEBUG - Arguments:", args)

            with lf.start_as_current_observation(
                as_type="span",
                name=f"tool_call:{tool_name}",
            ) as tool_span:

                tool_span.update(input={"tool": tool_name, "args": args})

                if tool_name == "consultar_apuntes":
                    # ask_rag tiene spans internos que se anidarán aquí automáticamente.
                    rag_result   = ask_rag(args["pregunta"])
                    tool_content = rag_result["answer"]

                elif tool_name == "resumir_semana":
                    rag_result   = summarize_week(args["semana"])
                    tool_content = rag_result["answer"]

                elif tool_name == "consultar_transacciones":
                    args["pregunta_original"] = question
                    rag_result   = run_transactional_agent(args)
                    tool_content = rag_result["answer"]

                else:
                    tool_content = "Herramienta no reconocida"
                    rag_result   = {
                        "answer": tool_content,
                        "retrieved_documents": [],
                        "metadatas": [],
                        "scores": [],
                    }

                tool_span.update(output={"tool_content_preview": tool_content[:300]})

            # ── Span 3: respuesta final del orquestador ─────
            with lf.start_as_current_observation(
                as_type="span",
                name="agent_final_response",
            ):
                final = client.chat.completions.create(
                    model=MODEL_ORCHESTRATOR,
                    temperature=0,
                    messages=[
                        {"role": "system",  "content": SYSTEM_PROMPT},
                        {"role": "user",    "content": question},
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": message.tool_calls,
                        },
                        {
                            "role": "tool",
                            "tool_call_id": message.tool_calls[0].id,
                            "content": tool_content,
                        },
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