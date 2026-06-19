import streamlit as st
from openai import OpenAI

from agent import run_agent
from memory import (
    create_session,
    save_message,
    save_session_summary,
)

# ==================================================
# CONFIG
# ==================================================

st.set_page_config(
    page_title="Asistente IA",
    page_icon="🤖",
    layout="wide"
)

# ==================================================
# INICIALIZAR ESTADO DE SESIÓN
# ==================================================

if "session_id" not in st.session_state:
    st.session_state.session_id = create_session()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # [{"role": "user"|"assistant", "content": "..."}]

if "display_messages" not in st.session_state:
    st.session_state.display_messages = []  # para mostrar en la UI

# ==================================================
# HEADER
# ==================================================

st.title("🤖 Asistente Académico IA")
st.markdown("Consulta apuntes del curso (RAG) o transacciones ficticias (MCP + PostgreSQL).")
st.caption(f"Sesión #{st.session_state.session_id}")

# ==================================================
# BOTÓN NUEVA SESIÓN
# ==================================================

col1, col2 = st.columns([6, 1])
with col2:
    if st.button("Nueva sesión"):
        # Guardar resumen de la sesión actual antes de cerrar
        if st.session_state.chat_history:
            _client = OpenAI()
            history_text = "\n".join(
                f"{m['role']}: {m['content']}"
                for m in st.session_state.chat_history
            )
            try:
                resp = _client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0,
                    messages=[{
                        "role": "user",
                        "content": f"Resume en 2-3 oraciones esta conversación:\n\n{history_text}"
                    }]
                )
                summary = resp.choices[0].message.content
                save_session_summary(st.session_state.session_id, summary)
            except Exception:
                pass

        # Reiniciar estado
        st.session_state.session_id = create_session()
        st.session_state.chat_history = []
        st.session_state.display_messages = []
        st.rerun()

# ==================================================
# MOSTRAR HISTORIAL DE CHAT
# ==================================================

for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

        # Mostrar fuentes si las hay (solo para mensajes del asistente)
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Fuentes recuperadas"):
                for i, doc in enumerate(msg["sources"]):
                    st.markdown(f"**Documento {i+1}**")
                    st.write(doc)

        if msg["role"] == "assistant" and msg.get("metadatas"):
            with st.expander("Metadatos"):
                for i, meta in enumerate(msg["metadatas"]):
                    st.markdown(f"**Fuente {i+1}**")
                    st.json(meta)

# ==================================================
# INPUT DE CHAT
# ==================================================

if question := st.chat_input("Escribe tu pregunta..."):

    # Mostrar pregunta del usuario
    with st.chat_message("user"):
        st.write(question)

    st.session_state.display_messages.append({
        "role": "user",
        "content": question
    })

    # Guardar en memoria histórica
    save_message(st.session_state.session_id, "user", question)

    # Ejecutar agente con historial de sesión
    with st.chat_message("assistant"):
        with st.spinner("Buscando respuesta..."):
            result = run_agent(
                question=question,
                chat_history=st.session_state.chat_history,
                session_id=st.session_state.session_id,
            )

        answer = result["answer"]
        st.write(answer)

        if result.get("retrieved_documents"):
            with st.expander("Fuentes recuperadas"):
                for i, doc in enumerate(result["retrieved_documents"]):
                    st.markdown(f"**Documento {i+1}**")
                    st.write(doc)

        if result.get("metadatas"):
            with st.expander("Metadatos"):
                for i, meta in enumerate(result["metadatas"]):
                    st.markdown(f"**Fuente {i+1}**")
                    st.json(meta)

    # Actualizar historial de sesión (para pasarlo al agente en el próximo turno)
    st.session_state.chat_history.append({"role": "user",      "content": question})
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

    # Guardar respuesta en memoria histórica
    save_message(st.session_state.session_id, "assistant", answer)

    # Actualizar display
    st.session_state.display_messages.append({
        "role":      "assistant",
        "content":   answer,
        "sources":   result.get("retrieved_documents", []),
        "metadatas": result.get("metadatas", []),
    })