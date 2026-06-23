"""
run_experiments.py
------------------
Runs the agent against every dataset in Langfuse and evaluates
the results using the configured LLM evaluator.

Also includes a manual conversation evaluation that chains
multi-turn exchanges and scores only the final follow-up answer.
"""

from dotenv import load_dotenv

load_dotenv()

from langfuse import get_client
from agent import run_agent

lf = get_client()

# ==========================================================
# DATASET EXPERIMENTS  (factual, comparacion, etc.)
# ==========================================================

def _question_from_item(item) -> str:
    if isinstance(item.input, dict):
        return item.input.get("question", str(item.input))
    return item.input


def task(item):
    """
    Called once per dataset item by run_experiment.
    Returns a string — becomes {{output}} in the evaluator prompt.
    """
    result = run_agent(question=_question_from_item(item))
    return result["answer"]


def evaluate_agent(
    run,
    dataset_names=("factual", "comparacion", "fuera_alcance", "websearch", "transactional"),
):
    for ds in dataset_names:
        dataset = lf.get_dataset(ds)

        if not dataset.items:
            print(f"\n  '{ds}' no tiene items, saltando.")
            continue

        print(f"\n  Ejecutando '{ds}' con {len(dataset.items)} items...")

        lf.run_experiment(
            name=ds,
            run_name=str(run),
            data=dataset.items,
            task=task,
        )

    print("\nDone.")

# ==========================================================
# CONVERSATIONAL TEST CASES
# ----------------------------------------------------------
# Each entry is a conversation with N turns.
# The agent's real output is fed as history to the next turn.
# Only the LAST turn is scored against expected_output.
# ==========================================================

CONVERSATION_DATASET = "conversacion"

CONVERSATIONS = [
    {
        "id": "conv-01",
        "turns": [
            {"question": "¿Qué es el descenso de gradiente?"},
            {"question": "¿Cuál es la diferencia entre el descenso de gradiente batch y el estocástico?"},
            {"question": "¿Cuál de los dos recomendarías para un dataset grande y por qué?"},  # ← scored
        ],
        "expected_output": (
            "Para datasets grandes se recomienda el descenso de gradiente estocástico (SGD) "
            "o mini-batch, ya que actualiza los parámetros con cada muestra o subconjunto, "
            "lo que es computacionalmente más eficiente y permite convergencia más rápida "
            "que el batch que requiere procesar todo el dataset en cada iteración."
        ),
    },
    {
        "id": "conv-02",
        "turns": [
            {"question": "Explícame qué es el overfitting."},
            {"question": "¿Qué técnicas existen para evitarlo?"},
            {"question": "De las técnicas que mencionaste, ¿cuál es la más efectiva cuando el dataset es pequeño?"},  # ← scored
        ],
        "expected_output": (
            "Cuando el dataset es pequeño, la regularización (L1 o L2) y el dropout son "
            "las técnicas más recomendadas, ya que penalizan la complejidad del modelo sin "
            "requerir más datos. El aumento de datos también puede ser útil si el dominio "
            "lo permite, pero depende del tipo de problema."
        ),
    },
    {
        "id": "conv-03",
        "turns": [
            {"question": "¿Qué es una red neuronal convolucional?"},
            {"question": "¿Para qué tipo de problemas se usa principalmente?"},
            {"question": "Compara su uso con una red neuronal densa clásica para clasificación de imágenes."},  # ← scored
        ],
        "expected_output": (
            "Las CNN son superiores a las redes densas para clasificación de imágenes porque "
            "explotan la estructura espacial mediante convoluciones y pooling, reduciendo la "
            "cantidad de parámetros y capturando patrones locales. Las redes densas tratan "
            "cada píxel de forma independiente, perdiendo información espacial y siendo "
            "computacionalmente más costosas para imágenes."
        ),
    },
    {
        "id": "conv-04",
        "turns": [
            {"question": "¿Qué es la regresión logística?"},
            {"question": "¿Cómo se interpreta la salida de una regresión logística?"},
            {"question": "Si la salida es 0.85, ¿qué significa y cómo tomarías una decisión de clasificación?"},  # ← scored
        ],
        "expected_output": (
            "Una salida de 0.85 significa que el modelo estima un 85% de probabilidad de "
            "que la muestra pertenezca a la clase positiva. Con un umbral estándar de 0.5, "
            "se clasificaría como clase positiva. El umbral puede ajustarse según el costo "
            "relativo de falsos positivos vs falsos negativos en el problema específico."
        ),
    },
    {
        "id": "conv-05",
        "turns": [
            {"question": "¿Qué es el aprendizaje por refuerzo?"},
            {"question": "¿Cuáles son sus componentes principales?"},
            {"question": "¿En qué se diferencia del aprendizaje supervisado que vimos antes?"},  # ← scored
        ],
        "expected_output": (
            "A diferencia del aprendizaje supervisado, que aprende de pares etiquetados "
            "(entrada, salida correcta), el aprendizaje por refuerzo aprende mediante "
            "interacción con un entorno: el agente toma acciones, recibe recompensas o "
            "penalizaciones, y ajusta su política para maximizar la recompensa acumulada. "
            "No requiere datos etiquetados sino señales de retroalimentación del entorno."
        ),
    },
]

# ==========================================================
# CONVERSATION EXPERIMENTS
# ----------------------------------------------------------
# Chains N turns, feeds real agent outputs as history,
# scores only the final turn against expected_output.
# Creates one dataset item + linked trace per conversation.
# ==========================================================

def evaluate_agent_conversation(run: str = "conv-run-001"):
    """
    Runs all CONVERSATIONS, feeds each turn's real output as
    chat_history to the next, and links the final trace to a
    Langfuse dataset item for LLM-as-judge scoring.
    """

    # Ensure the conversation dataset exists
    try:
        lf.get_dataset(CONVERSATION_DATASET)
    except Exception:
        lf.create_dataset(name=CONVERSATION_DATASET)

    print(f"\nRunning {len(CONVERSATIONS)} conversational tests (run='{run}')...\n")

    for conv in CONVERSATIONS:
        conv_id      = conv["id"]
        turns        = conv["turns"]
        expected     = conv["expected_output"]
        chat_history = []
        final_answer = ""

        print(f"  [{conv_id}] {len(turns)} turns")

        # ── Wrap the entire conversation in one root trace ────
        # This gives run_agent an active span context to nest into,
        # fixing the "No active span" warning.
        with lf.start_as_current_observation(
            name=f"conversation_{conv_id}",
            as_type="agent",
            input={"turns": [t["question"] for t in turns]},
        ):
            for i, turn in enumerate(turns):
                question = turn["question"]
                is_last  = (i == len(turns) - 1)

                result = run_agent(question=question, chat_history=chat_history)
                answer = result["answer"]

                if is_last:
                    final_answer = answer
                    lf.update_current_span(output=final_answer)
                else:
                    chat_history.append({"role": "user",      "content": question})
                    chat_history.append({"role": "assistant", "content": answer})
                    print(f"    turn {i+1}/{len(turns)}: ✓")

        # Capture trace_id after the context manager closes
        trace_id = lf.get_current_trace_id()

        # ── Create dataset item with final question + expected ─
        item = lf.create_dataset_item(
            dataset_name=CONVERSATION_DATASET,
            input=turns[-1]["question"],
            expected_output=expected,
        )

        # ── Link via run_experiment with a single cached item ─
        # task() must accept item as a keyword argument (v4 requirement)
        cached = final_answer
        lf.run_experiment(
            name=CONVERSATION_DATASET,
            run_name=run,
            data=[item],
            task=lambda *, item, _answer=cached: _answer,
        )

        print(f"    turn {len(turns)}/{len(turns)}: ✓  (trace: {str(trace_id)[:8]}...)")

    lf.flush()
    print(f"\nDone. Check Langfuse → Datasets → {CONVERSATION_DATASET} → Experiments → {run}")


if __name__ == "__main__":
    import sys

    run_name = sys.argv[1] if len(sys.argv) > 1 else "run-001"
    dataset_names = (
        tuple(sys.argv[2:])
        if len(sys.argv) > 2
        else ("factual", "comparacion", "fuera_alcance", "websearch", "transactional")
    )
    evaluate_agent(run_name, dataset_names=dataset_names)