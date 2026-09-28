"""Tool definitions given to the agent, plus their local executors.

Each tool has:
  - a JSON schema (what the model sees, in Anthropic's tool-use format)
  - an executor function (what actually runs when the model calls it)

Keeping schema and executor side by side makes it easy to add a new tool
without touching agent.py at all.
"""
from __future__ import annotations

import ast
import operator
from typing import Any, Callable, Dict

from .vectorstore import VectorStore

# ---------------------------------------------------------------------------
# Tool schemas (Anthropic tool-use format)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "name": "search_documents",
        "description": (
            "Search the ingested knowledge base for passages relevant to a "
            "query. Use this when the user asks something that likely "
            "depends on the content of the uploaded documents, not for "
            "general knowledge or pure arithmetic."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query, phrased as a natural question or keyword set.",
                },
                "k": {
                    "type": "integer",
                    "description": "Number of passages to retrieve (default 4).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "calculator",
        "description": (
            "Evaluate a pure arithmetic expression (numbers, +, -, *, /, "
            "%, **, parentheses only). Use this for any math instead of "
            "computing it yourself, to avoid arithmetic mistakes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "e.g. '(120 * 1.08) / 3'",
                }
            },
            "required": ["expression"],
        },
    },
]


# ---------------------------------------------------------------------------
# Safe calculator (no eval(), restricted AST walk)
# ---------------------------------------------------------------------------

_ALLOWED_OPERATORS: Dict[type, Callable[..., Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants are allowed.")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](
            _eval_node(node.left), _eval_node(node.right)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Disallowed expression element: {type(node).__name__}")


def safe_calculate(expression: str) -> float:
    try:
        tree = ast.parse(expression, mode="eval")
        return _eval_node(tree.body)
    except Exception as exc:
        raise ValueError(f"Could not evaluate '{expression}': {exc}") from exc


# ---------------------------------------------------------------------------
# Executors
# ---------------------------------------------------------------------------


def make_tool_executors(store: VectorStore) -> Dict[str, Callable[[dict], str]]:
    """Bind tool executors to a specific vector store instance."""

    def _search_documents(args: dict) -> str:
        query = args.get("query", "")
        k = int(args.get("k", 4))
        if store.is_empty():
            return "The knowledge base is empty. No documents have been ingested yet."
        results = store.search(query, k=k)
        if not results:
            return "No relevant passages were found for that query."
        formatted = []
        for i, r in enumerate(results, start=1):
            formatted.append(
                f"[{i}] (source: {r.source}, relevance: {r.score:.3f})\n{r.text}"
            )
        return "\n\n".join(formatted)

    def _calculator(args: dict) -> str:
        expression = args.get("expression", "")
        try:
            result = safe_calculate(expression)
            return str(result)
        except ValueError as exc:
            return f"Error: {exc}"

    return {
        "search_documents": _search_documents,
        "calculator": _calculator,
    }
