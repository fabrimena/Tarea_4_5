"""
build_raptor_index.py
---------------------
Agrega resúmenes jerárquicos RAPTOR al ChromaDB existente (./vectordb).
NO borra ni modifica los chunks originales — solo añade nodos nuevos.

Ejecutar una sola vez (o cuando cambien los apuntes):
    python build_raptor_index.py

Para forzar la regeneración de todas las semanas:
    python build_raptor_index.py --rebuild

Niveles que construye:
  Nivel 1 → un resumen por semana   (id: raptor_semana_<N>)
  Nivel 2 → un resumen global       (id: raptor_global)

Estructura de cada resumen de semana:
  - "los apuntes de la semana X fueron anotados por: [autores]."
  - "el quiz N fue respondido en esta semana." (según QUIZ_POR_SEMANA)
  - Descripción de temas, conceptos y técnicas vistas.
"""

import sys
from dotenv import load_dotenv
load_dotenv()

import chromadb
from openai import OpenAI
from config import MODEL_SUMMARIZER

# ----------------------------------------------------------
# Conexión
# ----------------------------------------------------------

client = OpenAI()

chroma_client = chromadb.PersistentClient(path="./vectordb")
collection = chroma_client.get_or_create_collection(name="apuntes_ai")

MAX_CHUNKS_PER_WEEK = 30

REBUILD = "--rebuild" in sys.argv

# ----------------------------------------------------------
# Mapeo hardcoded: semana → número de quiz (muy dificil inferir los quizzes desde los documentos)
# ----------------------------------------------------------

QUIZ_POR_SEMANA = {
    3:  1,
    6:  2,
    7:  3,
    8:  4,
    10: 5,
    14: 6,
}


# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------

def get_embedding(text: str) -> list[float]:
    return client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding


def summarize(prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_SUMMARIZER,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


def get_original_chunks(semana: int) -> tuple[list[str], list[dict]]:
    """
    Devuelve los chunks originales de una semana, excluyendo nodos RAPTOR.
    Filtra en Python para evitar el operador $and de ChromaDB.
    """
    result = collection.get(
        where={"semana": {"$eq": semana}},
        include=["documents", "metadatas"]
    )

    docs, metas = [], []
    for doc, meta in zip(result["documents"], result["metadatas"]):
        if meta.get("tipo") not in ("resumen_semana", "resumen_global"):
            docs.append(doc)
            metas.append(meta)

    return docs, metas


def extract_autores(metas: list[dict]) -> str:
    """Extrae autores únicos de los metadatos de una semana."""
    autores = set()
    for m in metas:
        valor = m.get("autor", "")
        if valor and valor not in ("Desconocido", "desconocido"):
            for nombre in valor.split(","):
                nombre = nombre.strip()
                if nombre:
                    autores.add(nombre)
    return ", ".join(sorted(autores)) if autores else "desconocido"


# ----------------------------------------------------------
# Nivel 1: resúmenes por semana
# ----------------------------------------------------------

def build_week_summaries() -> list[tuple[int, str]]:
    """
    Para cada semana en ChromaDB, genera un resumen de sus chunks
    y lo inserta como un nuevo documento con metadato nivel=1.
    Devuelve lista de (semana, resumen) para construir el nivel 2.
    """

    all_items = collection.get(include=["metadatas"])
    semanas = sorted(set(
        int(m["semana"])
        for m in all_items["metadatas"]
        if m.get("semana") not in (None, "", "None", 0)
        and m.get("tipo") not in ("resumen_semana", "resumen_global")
    ))

    print(f"Semanas encontradas: {semanas}\n")

    week_summaries = []

    for semana in semanas:
        raptor_id = f"raptor_semana_{semana}"

        # Si ya existe y no se pidió rebuild, usar el existente
        existing = collection.get(ids=[raptor_id])
        if existing["ids"] and not REBUILD:
            print(f"  Semana {semana}: resumen ya existe, saltando.")
            week_summaries.append((semana, existing["documents"][0]))
            continue

        # Obtener chunks originales
        chunks, metas = get_original_chunks(semana)

        if not chunks:
            print(f"  Semana {semana}: sin chunks, saltando.")
            continue

        autores_str = extract_autores(metas)
        chunks_para_resumir = chunks[:MAX_CHUNKS_PER_WEEK]
        context = "\n\n".join(chunks_para_resumir)

        # Obtener quiz de esta semana desde el mapeo
        numero_quiz = QUIZ_POR_SEMANA.get(semana)
        if numero_quiz:
            quiz_info = f"el quiz {numero_quiz} fue respondido en esta semana."
            print(f"  Semana {semana}: quiz {numero_quiz} asignado.")
        else:
            quiz_info = ""
            print(f"  Semana {semana}: sin quiz.")

        # Construir línea de quiz para el prompt
        quiz_line = (
            f"\n2. Inmediatamente después escribe en minúsculas: \"{quiz_info}\""
            if quiz_info else ""
        )
        paso_temas = "3" if quiz_info else "2"

        prompt = f"""Eres un asistente que resume apuntes académicos de un curso de Inteligencia Artificial.

Los apuntes de la semana {semana} fueron escritos por los siguientes estudiantes: {autores_str}.

Tu tarea: escribe un resumen completo de los temas cubiertos en esta semana siguiendo este formato:

1. Comienza SIEMPRE con esta frase exacta (en minúsculas):
   "los apuntes de la semana {semana} fueron anotados por: {autores_str}."{quiz_line}

{paso_temas}. Luego describe:
   - Los temas principales cubiertos en la semana {semana}
   - Conceptos clave, algoritmos y técnicas vistas
   - Cualquier actividad evaluativa adicional mencionada (tareas, proyectos)

El resumen debe permitir responder preguntas como:
   - "¿en qué semana se vio X tema?"
   - "¿quién anotó los apuntes de la semana Y?"
   - "¿qué quiz se realizó en la semana Z?"
   - "¿quién respondió el quiz N?"

Escribe en español, en minúsculas (los apuntes están normalizados así).
Sé específico con nombres de temas, técnicas y autores.

APUNTES DE LA SEMANA {semana}:
{context}

RESUMEN:"""

        print(f"  Semana {semana}: generando resumen ({len(chunks)} chunks, autores: {autores_str[:50]}...)...", end=" ")
        summary = summarize(prompt)
        print("✓")

        # Borrar versión anterior si existe (caso --rebuild)
        if existing["ids"]:
            collection.delete(ids=[raptor_id])

        # Insertar en ChromaDB
        embedding = get_embedding(summary)
        collection.add(
            ids=[raptor_id],
            documents=[summary],
            embeddings=[embedding],
            metadatas=[{
                "semana": semana,
                "tipo": "resumen_semana",
                "nivel": 1,
                "autor": autores_str,
                "nombre_archivo": f"raptor_semana_{semana}",
                "fecha": "",
                "tema_principal": f"resumen semana {semana}",
                "secciones": "",
                "chunk_numero": 0,
                "total_chunks": len(chunks),
            }]
        )

        week_summaries.append((semana, summary))

    return week_summaries


# ----------------------------------------------------------
# Nivel 2: resumen global del curso
# ----------------------------------------------------------

def build_global_summary(week_summaries: list[tuple[int, str]]) -> None:
    """
    Genera un resumen global del curso a partir de los resúmenes por semana
    e inserta un único nodo raíz con id 'raptor_global'.
    """
    raptor_id = "raptor_global"

    existing = collection.get(ids=[raptor_id])
    if existing["ids"] and not REBUILD:
        print("\nResumen global ya existe. Usa --rebuild para regenerarlo.")
        return

    if not week_summaries:
        print("\nNo hay resúmenes de semana para construir el resumen global.")
        return

    semanas_texto = "\n\n".join(
        f"=== SEMANA {s} ===\n{txt}"
        for s, txt in sorted(week_summaries)
    )

    prompt = f"""Eres un asistente que resume el contenido completo de un curso de Inteligencia Artificial.

A continuación tienes el resumen de cada semana del curso, incluyendo autores y quizzes.

Tu tarea: escribe un resumen global del curso que incluya:
- Lista de semanas con sus temas principales, autores y quizzes realizados
  (ej: "semana 6 (autores: David Blanco, Victor Aymerich | quiz 2): regresión logística, función sigmoide")
- Progresión temática del curso
- Información útil para responder:
  * "¿en qué semana se vio X?"
  * "¿quién escribió los apuntes de la semana Y?"
  * "¿en qué semana se realizó el quiz N?"
  * "¿quién respondió el quiz N?"

Escribe en español, en minúsculas.

RESÚMENES POR SEMANA:
{semanas_texto}

RESUMEN GLOBAL:"""

    print("\nGenerando resumen global del curso...", end=" ")
    summary = summarize(prompt)
    print("✓")

    if existing["ids"]:
        collection.delete(ids=[raptor_id])

    embedding = get_embedding(summary)
    collection.add(
        ids=[raptor_id],
        documents=[summary],
        embeddings=[embedding],
        metadatas=[{
            "semana": 0,
            "tipo": "resumen_global",
            "nivel": 2,
            "autor": "",
            "nombre_archivo": "raptor_global",
            "fecha": "",
            "tema_principal": "resumen global del curso",
            "secciones": "",
            "chunk_numero": 0,
            "total_chunks": len(week_summaries),
        }]
    )


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("RAPTOR — Construcción de índice jerárquico")
    if REBUILD:
        print("Modo: REBUILD (regenerando todos los resúmenes)")
    print("=" * 60)
    print(f"Documentos en ChromaDB antes: {collection.count()}\n")

    print("[ NIVEL 1 ] Resúmenes por semana")
    print("-" * 40)
    week_summaries = build_week_summaries()

    print("\n[ NIVEL 2 ] Resumen global del curso")
    print("-" * 40)
    build_global_summary(week_summaries)

    print("\n" + "=" * 60)
    print(f"Documentos en ChromaDB después: {collection.count()}")
    print("RAPTOR completado ✓")
    print("=" * 60)