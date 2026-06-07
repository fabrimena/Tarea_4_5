import json
import os
from contextlib import contextmanager

import psycopg2
from psycopg2.extras import RealDictCursor

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://fraud_user:fraud_pass@localhost:5432/fraud_db",
)

MAX_DATE_RANGE_DAYS = 90
MAX_SEARCH_RESULTS = 50


@contextmanager
def get_connection():
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
    finally:
        conn.close()


def mask_account_number(account_number: str) -> str:
    digits = "".join(ch for ch in account_number if ch.isdigit())
    if len(digits) < 4:
        return "****"
    return f"****{digits[-4:]}"


def log_tool_access(conn, tool_name: str, justification: str, request_data: dict):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO tool_access_logs (tool_name, justification, request_data)
            VALUES (%s, %s, %s)
            """,
            (tool_name, justification, json.dumps(request_data)),
        )
    conn.commit()


def validate_justification(justification: str):
    if not justification or not justification.strip():
        raise ValueError("Toda llamada al MCP debe incluir una justificación.")
    if len(justification.strip()) < 10:
        raise ValueError("La justificación debe tener al menos 10 caracteres.")


def validate_date_range(start_date, end_date):
    if not start_date or not end_date:
        raise ValueError("Las búsquedas históricas deben incluir rango de fechas.")
    delta = end_date - start_date
    if delta.days > MAX_DATE_RANGE_DAYS:
        raise ValueError(f"El rango de fechas no puede superar {MAX_DATE_RANGE_DAYS} días.")
    if delta.days < 0:
        raise ValueError("La fecha de inicio no puede ser posterior a la fecha final.")
