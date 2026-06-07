import json
from datetime import datetime, timedelta

from psycopg2.extras import RealDictCursor

from mcp_server.db import (
    MAX_DATE_RANGE_DAYS,
    MAX_SEARCH_RESULTS,
    get_connection,
    log_tool_access,
    mask_account_number,
    validate_date_range,
    validate_justification,
)


def _anonymize_customer(row: dict) -> dict:
    email = row.get("email", "")
    if "@" in email:
        local, domain = email.split("@", 1)
        row["email"] = f"{local[:2]}***@{domain}"
    return row


def _format_transaction(row: dict) -> dict:
    row = dict(row)
    if "account_number" in row:
        row["account_number_masked"] = mask_account_number(row.pop("account_number"))
    return row


def get_transaction_by_id(transaction_id: int, justification: str) -> str:
    validate_justification(justification)

    with get_connection() as conn:
        log_tool_access(conn, "get_transaction_by_id", justification, {"transaction_id": transaction_id})

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT t.id, t.amount, t.currency, t.country, t.status,
                       t.is_flagged, t.failure_reason, t.created_at,
                       a.account_number, c.full_name, c.risk_level
                FROM transactions t
                JOIN accounts a ON a.id = t.account_id
                JOIN customers c ON c.id = a.customer_id
                WHERE t.id = %s
                """,
                (transaction_id,),
            )
            row = cur.fetchone()

    if not row:
        return json.dumps({"error": f"No existe la transacción {transaction_id}."}, ensure_ascii=False)

    data = _format_transaction(row)
    data["created_at"] = data["created_at"].isoformat()
    data["data_used"] = ["transactions", "accounts", "customers"]
    return json.dumps(data, ensure_ascii=False)


def search_transactions(
    justification: str,
    customer_id: int | None = None,
    status: str | None = None,
    is_flagged: bool | None = None,
    min_amount: float | None = None,
    max_amount: float | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> str:
    validate_justification(justification)

    filters = {
        "customer_id": customer_id,
        "status": status,
        "is_flagged": is_flagged,
        "min_amount": min_amount,
        "max_amount": max_amount,
        "start_date": start_date,
        "end_date": end_date,
    }
    active_filters = {k: v for k, v in filters.items() if v is not None}
    if not active_filters:
        raise ValueError("No se permiten consultas masivas sin filtros.")

    start_dt = datetime.fromisoformat(start_date) if start_date else None
    end_dt = datetime.fromisoformat(end_date) if end_date else None
    if start_dt and end_dt:
        validate_date_range(start_dt, end_dt)

    clauses = []
    params = []

    if customer_id is not None:
        clauses.append("c.id = %s")
        params.append(customer_id)
    if status is not None:
        clauses.append("t.status = %s")
        params.append(status)
    if is_flagged is not None:
        clauses.append("t.is_flagged = %s")
        params.append(is_flagged)
    if min_amount is not None:
        clauses.append("t.amount >= %s")
        params.append(min_amount)
    if max_amount is not None:
        clauses.append("t.amount <= %s")
        params.append(max_amount)
    if start_dt is not None:
        clauses.append("t.created_at >= %s")
        params.append(start_dt)
    if end_dt is not None:
        clauses.append("t.created_at <= %s")
        params.append(end_dt)

    where_sql = " AND ".join(clauses)

    with get_connection() as conn:
        log_tool_access(conn, "search_transactions", justification, active_filters)

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT t.id, t.amount, t.currency, t.country, t.status,
                       t.is_flagged, t.created_at, a.account_number, c.full_name
                FROM transactions t
                JOIN accounts a ON a.id = t.account_id
                JOIN customers c ON c.id = a.customer_id
                WHERE {where_sql}
                ORDER BY t.created_at DESC
                LIMIT %s
                """,
                (*params, MAX_SEARCH_RESULTS),
            )
            rows = cur.fetchall()

    results = []
    for row in rows:
        item = _format_transaction(row)
        item["created_at"] = item["created_at"].isoformat()
        results.append(item)

    return json.dumps(
        {
            "count": len(results),
            "filters_applied": active_filters,
            "transactions": results,
            "data_used": ["transactions", "accounts", "customers"],
        },
        ensure_ascii=False,
    )


def get_customer_risk_summary(customer_id: int, justification: str) -> str:
    validate_justification(justification)

    with get_connection() as conn:
        log_tool_access(conn, "get_customer_risk_summary", justification, {"customer_id": customer_id})

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT c.id, c.full_name, c.email, c.country, c.risk_level,
                       COUNT(t.id) AS total_transactions,
                       COUNT(*) FILTER (WHERE t.is_flagged) AS flagged_transactions,
                       COALESCE(SUM(t.amount) FILTER (WHERE t.status = 'approved'), 0) AS approved_volume
                FROM customers c
                LEFT JOIN accounts a ON a.customer_id = c.id
                LEFT JOIN transactions t ON t.account_id = a.id
                WHERE c.id = %s
                GROUP BY c.id
                """,
                (customer_id,),
            )
            row = cur.fetchone()

    if not row:
        return json.dumps({"error": f"No existe el cliente {customer_id}."}, ensure_ascii=False)

    data = _anonymize_customer(dict(row))
    data["approved_volume"] = float(data["approved_volume"])
    data["data_used"] = ["customers", "accounts", "transactions"]
    return json.dumps(data, ensure_ascii=False)


def get_recent_flagged_transactions(days: int, justification: str) -> str:
    validate_justification(justification)

    if days < 1 or days > MAX_DATE_RANGE_DAYS:
        raise ValueError(f"El parámetro days debe estar entre 1 y {MAX_DATE_RANGE_DAYS}.")

    since = datetime.now() - timedelta(days=days)

    with get_connection() as conn:
        log_tool_access(conn, "get_recent_flagged_transactions", justification, {"days": days})

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT t.id, t.amount, t.currency, t.country, t.status,
                       t.created_at, a.account_number, c.full_name, c.risk_level
                FROM transactions t
                JOIN accounts a ON a.id = t.account_id
                JOIN customers c ON c.id = a.customer_id
                WHERE t.is_flagged = TRUE AND t.created_at >= %s
                ORDER BY t.created_at DESC
                LIMIT %s
                """,
                (since, MAX_SEARCH_RESULTS),
            )
            rows = cur.fetchall()

    results = []
    for row in rows:
        item = _format_transaction(row)
        item["created_at"] = item["created_at"].isoformat()
        results.append(item)

    return json.dumps(
        {
            "days": days,
            "count": len(results),
            "flagged_transactions": results,
            "data_used": ["transactions", "accounts", "customers"],
        },
        ensure_ascii=False,
    )


def create_fraud_case(transaction_id: int, reason: str, severity: str, justification: str) -> str:
    validate_justification(justification)

    allowed_severity = {"low", "medium", "high", "critical"}
    if severity not in allowed_severity:
        raise ValueError(f"Severity inválida. Valores permitidos: {sorted(allowed_severity)}")

    with get_connection() as conn:
        log_tool_access(
            conn,
            "create_fraud_case",
            justification,
            {"transaction_id": transaction_id, "reason": reason, "severity": severity},
        )

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT id FROM transactions WHERE id = %s", (transaction_id,))
            if not cur.fetchone():
                return json.dumps({"error": f"No existe la transacción {transaction_id}."}, ensure_ascii=False)

            cur.execute(
                """
                INSERT INTO fraud_cases (transaction_id, reason, severity)
                VALUES (%s, %s, %s)
                RETURNING id, transaction_id, reason, severity, status, created_at
                """,
                (transaction_id, reason, severity),
            )
            case = cur.fetchone()
        conn.commit()

    case = dict(case)
    case["created_at"] = case["created_at"].isoformat()
    case["note"] = "No se modificó la transacción existente; solo se creó un caso de revisión."
    case["data_used"] = ["fraud_cases", "transactions"]
    return json.dumps(case, ensure_ascii=False)
