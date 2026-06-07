"""
Cliente MCP para que el agente transaccional consulte la BD
sin acceso directo. Usa transporte stdio al MCP Server.
"""

import asyncio
import json
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_PARAMS = StdioServerParameters(
    command=sys.executable,
    args=["-m", "mcp_server.server"],
    env=os.environ.copy(),
)


async def _call_tool_async(tool_name: str, arguments: dict) -> dict:
    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)

    if result.isError:
        text = result.content[0].text if result.content else "Error desconocido en MCP"
        return {"error": text}

    text = result.content[0].text if result.content else "{}"
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"result": text}


def call_mcp_tool(tool_name: str, arguments: dict) -> dict:
    return asyncio.run(_call_tool_async(tool_name, arguments))
