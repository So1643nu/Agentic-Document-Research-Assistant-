"""The agentic loop: decide -> act -> observe -> repeat.

On every turn, the model either returns a final text answer or asks to
call one or more tools. When it asks for tools, we execute them locally,
feed the results back as `tool_result` blocks, and let the model continue.
This repeats until the model stops calling tools (or MAX_AGENT_STEPS is
hit, as a safety valve against infinite loops).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import anthropic

from .config import Settings
from .tools import TOOL_SCHEMAS, make_tool_executors
from .vectorstore import VectorStore

SYSTEM_PROMPT = """You are a careful research assistant with access to two tools:

1. search_documents - searches a user-provided knowledge base
2. calculator - evaluates arithmetic expressions safely

Rules:
- Only call search_documents when the question plausibly depends on the
  ingested documents. Do not call it for general knowledge or greetings.
- Only call calculator for arithmetic; never compute math yourself.
- If search_documents returns nothing relevant, say so plainly rather than
  guessing or inventing an answer.
- Cite which source(s) you used when you rely on retrieved passages.
- Keep answers concise and direct.
"""


@dataclass
class ToolCallRecord:
    tool_name: str
    tool_input: Dict[str, Any]
    result: str


@dataclass
class AgentResponse:
    text: str
    tool_calls: List[ToolCallRecord] = field(default_factory=list)


class AgenticRAGAgent:
    def __init__(self, settings: Settings, store: VectorStore) -> None:
        self.settings = settings
        self.store = store
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.executors = make_tool_executors(store)

    def run(self, user_query: str, history: List[Dict[str, Any]] | None = None) -> AgentResponse:
        messages: List[Dict[str, Any]] = list(history or [])
        messages.append({"role": "user", "content": user_query})

        tool_calls: List[ToolCallRecord] = []

        for _ in range(self.settings.max_agent_steps):
            response = self.client.messages.create(
                model=self.settings.model_name,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS,
                messages=messages,
            )

            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

            if not tool_use_blocks:
                final_text = "".join(
                    b.text for b in response.content if b.type == "text"
                )
                return AgentResponse(text=final_text, tool_calls=tool_calls)

            # Model wants to use one or more tools: execute them and feed
            # results back, then loop again.
            messages.append({"role": "assistant", "content": response.content})

            tool_result_blocks = []
            for block in tool_use_blocks:
                executor = self.executors.get(block.name)
                if executor is None:
                    result_text = f"Error: unknown tool '{block.name}'"
                else:
                    result_text = executor(block.input)

                tool_calls.append(
                    ToolCallRecord(
                        tool_name=block.name,
                        tool_input=block.input,
                        result=result_text,
                    )
                )
                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                    }
                )

            messages.append({"role": "user", "content": tool_result_blocks})

        return AgentResponse(
            text=(
                "I hit the maximum number of reasoning steps without reaching "
                "a final answer. Try rephrasing your question or breaking it "
                "into smaller parts."
            ),
            tool_calls=tool_calls,
        )
