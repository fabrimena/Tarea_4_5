"""
Carga los 5 ítems de evaluación en el dataset 'transactional' de Langfuse.
Basado en los datos ficticios de db/seed.py.

Uso:
    python seed_transactional_dataset.py
"""

from dotenv import load_dotenv

load_dotenv()

from langfuse import get_client

DATASET_NAME = "transactional"

ITEMS = [
    {
        "id": "txn-01",
        "input": "¿Cuál es la transacción de mayor monto y de qué cliente es?",
        "expected_output": (
            "La transacción de mayor monto es de 48,500 USD (transacción ID 4), "
            "del cliente Ana Lucía Pérez (Panamá, riesgo alto). Está marcada como sospechosa."
        ),
    },
    {
        "id": "txn-02",
        "input": "¿Qué transacciones fallidas tiene el cliente 3?",
        "expected_output": (
            "El cliente 3 (Ana Lucía Pérez) tiene 2 transacciones fallidas: "
            "una de 1,200 USD por 'Fondos insuficientes' y otra de 1,500 USD por "
            "'Límite diario excedido'. Ambas están marcadas como sospechosas."
        ),
    },
    {
        "id": "txn-03",
        "input": "¿Cuántas transacciones sospechosas tiene Carlos Méndez Vega?",
        "expected_output": (
            "Carlos Méndez Vega (cliente ID 2, riesgo medio) tiene 5 transacciones "
            "marcadas como sospechosas en un periodo corto, con montos entre 45 y 61 USD."
        ),
    },
    {
        "id": "txn-04",
        "input": "¿Laura Jiménez realizó transacciones en países distintos el mismo día?",
        "expected_output": (
            "Sí. Laura Jiménez Castro (cliente ID 5) tuvo transacciones en España, "
            "Estados Unidos y Reino Unido el mismo día. Las tres están marcadas como sospechosas."
        ),
    },
    {
        "id": "txn-05",
        "input": "Dame el resumen de riesgo de María Solano Rojas.",
        "expected_output": (
            "María Solano Rojas (cliente ID 1) tiene riesgo bajo y país Costa Rica. "
            "Tiene transacciones aprobadas normales y al menos una transacción sospechosa "
            "de 9,200 USD, inusual para un cliente de bajo riesgo."
        ),
    },
]


def main():
    lf = get_client()

    try:
        lf.get_dataset(DATASET_NAME)
        print(f"Dataset '{DATASET_NAME}' encontrado.")
    except Exception:
        lf.create_dataset(name=DATASET_NAME)
        print(f"Dataset '{DATASET_NAME}' creado.")

    for item in ITEMS:
        lf.create_dataset_item(
            id=item["id"],
            dataset_name=DATASET_NAME,
            input=item["input"],
            expected_output=item["expected_output"],
        )
        print(f"  ✓ {item['id']}: {item['input'][:50]}...")

    lf.flush()
    dataset = lf.get_dataset(DATASET_NAME)
    print(f"\nListo: {len(dataset.items)} items en '{DATASET_NAME}'.")


if __name__ == "__main__":
    main()
