"""
MCP Server para consultas transaccionales controladas.
Ejecutar: python -m mcp_server.server
"""

from mcp.server.fastmcp import FastMCP

from mcp_server import tools

mcp = FastMCP("transaction-mcp-server")


@mcp.tool()
def get_transaction_by_id(transaction_id: int, justification: str) -> str:
    """Obtiene una transacción por ID. Requiere justificación."""
    return tools.get_transaction_by_id(transaction_id, justification)


@mcp.tool()
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
    """Busca transacciones con filtros obligatorios. No permite consultas masivas."""
    return tools.search_transactions(
        justification=justification,
        customer_id=customer_id,
        status=status,
        is_flagged=is_flagged,
        min_amount=min_amount,
        max_amount=max_amount,
        start_date=start_date,
        end_date=end_date,
    )


@mcp.tool()
def get_customer_risk_summary(customer_id: int, justification: str) -> str:
    """Resume el riesgo y actividad de un cliente. Datos sensibles anonimizados."""
    return tools.get_customer_risk_summary(customer_id, justification)


@mcp.tool()
def get_recent_flagged_transactions(days: int, justification: str) -> str:
    """Lista transacciones marcadas como sospechosas en los últimos N días."""
    return tools.get_recent_flagged_transactions(days, justification)


@mcp.tool()
def create_fraud_case(
    transaction_id: int,
    reason: str,
    severity: str,
    justification: str,
) -> str:
    """Crea un caso de fraude/revisión sin modificar la transacción original."""
    return tools.create_fraud_case(transaction_id, reason, severity, justification)


if __name__ == "__main__":
    mcp.run()
