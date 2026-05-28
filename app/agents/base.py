from __future__ import annotations

from typing import Any, Dict, List


AgentTrace = List[Dict[str, str]]


class BaseAgent:
    name = "BaseAgent"

    def add_trace(self, context: Dict[str, Any], status: str, message: str) -> None:
        context.setdefault("agent_trace", []).append(
            {"agent": self.name, "status": status, "message": message}
        )
