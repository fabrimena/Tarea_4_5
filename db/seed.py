"""
Pobla la base de datos ficticia con clientes, cuentas y transacciones.
Incluye patrones sospechosos requeridos por el enunciado.

Uso:
    cd db && docker compose up -d
    python seed.py
"""

import os
from datetime import datetime, timedelta

import psycopg2
from psycopg2.extras import execute_batch

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://fraud_user:fraud_pass@localhost:5432/fraud_db",
)


def connect():
    return psycopg2.connect(DATABASE_URL)


def clear_tables(conn):
    with conn.cursor() as cur:
        cur.execute("TRUNCATE fraud_cases, tool_access_logs, transactions, accounts, customers RESTART IDENTITY CASCADE")
    conn.commit()


def seed_customers(conn):
    customers = [
        ("María Solano Rojas", "maria.solano@mail.test", "Costa Rica", "low"),
        ("Carlos Méndez Vega", "carlos.mendez@mail.test", "Costa Rica", "medium"),
        ("Ana Lucía Pérez", "ana.perez@mail.test", "Panamá", "high"),
        ("Diego Herrera Mora", "diego.herrera@mail.test", "México", "low"),
        ("Laura Jiménez Castro", "laura.jimenez@mail.test", "España", "medium"),
    ]
    with conn.cursor() as cur:
        execute_batch(
            cur,
            "INSERT INTO customers (full_name, email, country, risk_level) VALUES (%s, %s, %s, %s)",
            customers,
        )
    conn.commit()


def seed_accounts(conn):
    accounts = [
        (1, "CR-1000-0001-4821", 12500.00, "active"),
        (2, "CR-1000-0002-7394", 8200.50, "active"),
        (3, "PA-2000-0003-1156", 450.00, "active"),
        (4, "MX-3000-0004-9023", 3100.75, "active"),
        (5, "ES-4000-0005-6678", 15600.00, "active"),
    ]
    with conn.cursor() as cur:
        execute_batch(
            cur,
            "INSERT INTO accounts (customer_id, account_number, balance, status) VALUES (%s, %s, %s, %s)",
            accounts,
        )
    conn.commit()


def seed_transactions(conn):
    now = datetime.now()
    rows = [
        # Normales
        (1, 120.00, "USD", "Costa Rica", "approved", False, None, now - timedelta(days=10, hours=10)),
        (2, 85.50, "USD", "Costa Rica", "approved", False, None, now - timedelta(days=8, hours=14)),
        (4, 200.00, "USD", "México", "approved", False, None, now - timedelta(days=5, hours=11)),
        # Monto inusualmente alto
        (3, 48500.00, "USD", "Panamá", "approved", True, None, now - timedelta(days=2, hours=9)),
        # Muchas transacciones en periodo corto (cuenta 2)
        (2, 45.00, "USD", "Costa Rica", "approved", True, None, now - timedelta(hours=6)),
        (2, 52.00, "USD", "Costa Rica", "approved", True, None, now - timedelta(hours=5, minutes=40)),
        (2, 48.00, "USD", "Costa Rica", "approved", True, None, now - timedelta(hours=5, minutes=20)),
        (2, 61.00, "USD", "Costa Rica", "approved", True, None, now - timedelta(hours=5)),
        (2, 55.00, "USD", "Costa Rica", "approved", True, None, now - timedelta(hours=4, minutes=45)),
        # Países distintos el mismo día (cuenta 5)
        (5, 300.00, "EUR", "España", "approved", True, None, now - timedelta(days=1, hours=15)),
        (5, 280.00, "USD", "Estados Unidos", "approved", True, None, now - timedelta(days=1, hours=16)),
        (5, 260.00, "GBP", "Reino Unido", "approved", True, None, now - timedelta(days=1, hours=17)),
        # Transacciones de madrugada
        (1, 900.00, "USD", "Costa Rica", "approved", True, None, now - timedelta(days=3) + timedelta(hours=2)),
        (4, 750.00, "USD", "México", "approved", True, None, now - timedelta(days=1, hours=3)),
        # Fallidas seguidas de aprobadas (cuenta 3 - alto riesgo)
        (3, 1200.00, "USD", "Panamá", "failed", True, "Fondos insuficientes", now - timedelta(days=4, hours=12)),
        (3, 1500.00, "USD", "Panamá", "failed", True, "Límite diario excedido", now - timedelta(days=4, hours=11, minutes=30)),
        (3, 1800.00, "USD", "Panamá", "approved", True, None, now - timedelta(days=4, hours=11)),
        # Actividad inusual para cliente low risk
        (1, 9200.00, "USD", "Costa Rica", "approved", True, None, now - timedelta(days=1, hours=20)),
    ]
    with conn.cursor() as cur:
        execute_batch(
            cur,
            """
            INSERT INTO transactions
                (account_id, amount, currency, country, status, is_flagged, failure_reason, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            rows,
        )
    conn.commit()


def main():
    conn = connect()
    try:
        clear_tables(conn)
        seed_customers(conn)
        seed_accounts(conn)
        seed_transactions(conn)
        print("Base de datos poblada correctamente.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
