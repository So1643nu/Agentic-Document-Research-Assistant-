import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools import make_tool_executors, safe_calculate
from src.vectorstore import TfidfStore


def test_safe_calculate_basic_arithmetic():
    assert safe_calculate("2 + 2") == 4
    assert safe_calculate("(120 * 1.08) / 3") == pytest.approx(43.2)
    assert safe_calculate("2 ** 10") == 1024


def test_safe_calculate_rejects_non_arithmetic():
    with pytest.raises(ValueError):
        safe_calculate("__import__('os').system('echo hi')")
    with pytest.raises(ValueError):
        safe_calculate("open('/etc/passwd').read()")


def test_calculator_executor_handles_bad_expression_gracefully():
    store = TfidfStore()
    executors = make_tool_executors(store)
    result = executors["calculator"]({"expression": "1 / 0"})
    assert "Error" in result


def test_search_documents_executor_on_empty_store():
    store = TfidfStore()
    executors = make_tool_executors(store)
    result = executors["search_documents"]({"query": "anything"})
    assert "empty" in result.lower()
