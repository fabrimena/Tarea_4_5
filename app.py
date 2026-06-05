import streamlit as st

from Tarea_4_5.vectordb.rag import ask_rag
from Tarea_4_5.vectordb.agent import run_agent

# ==================================================
# CONFIG
# ==================================================

st.set_page_config(
    page_title="Asistente IA",
    page_icon="🤖",
    layout="wide"
)

# ==================================================
# HEADER
# ==================================================

st.title("🤖 Asistente Académico IA")

st.markdown(
    """
Consulta los apuntes del curso usando RAG + OpenAI + ChromaDB.
"""
)

# ==================================================
# INPUT
# ==================================================

question = st.text_input(
    "Escribe tu pregunta:"
)

# ==================================================
# BOTON
# ==================================================

if st.button("Consultar"):

    if question.strip() == "":
        st.warning(
            "Ingrese una pregunta."
        )

    else:

        with st.spinner(
            "Buscando respuesta..."
        ):

            result = run_agent(question)

        # =====================================
        # RESPUESTA
        # =====================================

        st.subheader("Respuesta")

        st.write(
            result["answer"]
        )

        # =====================================
        # FUENTES
        # =====================================

        st.subheader(
            "Fuentes recuperadas"
        )

        for i, doc in enumerate(
            result["retrieved_documents"]
        ):

            with st.expander(
                f"Documento {i+1}"
            ):

                st.write(doc)

        # =====================================
        # METADATOS
        # =====================================

        st.subheader(
            "Metadatos"
        )

        for i, meta in enumerate(
            result["metadatas"]
        ):

            with st.expander(
                f"Fuente {i+1}"
            ):

                st.json(meta)