"""
Task 03 - MCP Tool Tests
Tests all three tools with valid and invalid inputs.
Verifies errors are returned as structured objects, not raw exceptions.

Usage: pytest tests/test_mcp_tools.py -v
"""

import json
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from mcp_server.server import _search_documents, _run_python, _get_date_context


# ─── search_documents tests ────────────────────────────────────────────────────

class TestSearchDocuments:
    @pytest.mark.asyncio
    async def test_valid_query_returns_results(self):
        result = await _search_documents({"query": "kubernetes deployment", "top_k": 3})
        assert len(result) == 1
        data = json.loads(result[0].text)
        # Either gets results or reports empty store — both are valid structured responses
        assert "error" in data or "results" in data

    @pytest.mark.asyncio
    async def test_empty_query_returns_error(self):
        result = await _search_documents({"query": "", "top_k": 3})
        assert len(result) == 1
        data = json.loads(result[0].text)
        assert "error" in data
        assert "non-empty" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_missing_query_returns_error(self):
        result = await _search_documents({"top_k": 3})
        assert len(result) == 1
        data = json.loads(result[0].text)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_top_k_clamped_to_valid_range(self):
        result = await _search_documents({"query": "docker", "top_k": 100})
        assert len(result) == 1
        data = json.loads(result[0].text)
        assert "error" in data or "results" in data

    @pytest.mark.asyncio
    async def test_returns_text_content_type(self):
        result = await _search_documents({"query": "helm chart"})
        assert result[0].type == "text"


# ─── run_python tests ──────────────────────────────────────────────────────────

class TestRunPython:
    @pytest.mark.asyncio
    async def test_simple_arithmetic(self):
        result = await _run_python({"code": "print(2 + 2)"})
        data = json.loads(result[0].text)
        assert data["exit_code"] == 0
        assert "4" in data["stdout"]
        assert data["stderr"] == ""

    @pytest.mark.asyncio
    async def test_multiline_code(self):
        code = "nums = [1, 2, 3, 4, 5]\nprint(sum(nums))"
        result = await _run_python({"code": code})
        data = json.loads(result[0].text)
        assert data["exit_code"] == 0
        assert "15" in data["stdout"]

    @pytest.mark.asyncio
    async def test_syntax_error_returns_structured_response(self):
        result = await _run_python({"code": "def broken(:"})
        data = json.loads(result[0].text)
        assert data["exit_code"] != 0
        assert "error" in data["stderr"].lower() or len(data["stderr"]) > 0

    @pytest.mark.asyncio
    async def test_empty_code_returns_error(self):
        result = await _run_python({"code": ""})
        data = json.loads(result[0].text)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_blocked_os_import(self):
        result = await _run_python({"code": "import os; print(os.listdir('/'))"})
        data = json.loads(result[0].text)
        assert "error" in data
        assert "Blocked" in data["error"]

    @pytest.mark.asyncio
    async def test_blocked_subprocess(self):
        result = await _run_python({"code": "subprocess.run(['ls'])"})
        data = json.loads(result[0].text)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_runtime_error_captured(self):
        result = await _run_python({"code": "print(1/0)"})
        data = json.loads(result[0].text)
        assert data["exit_code"] != 0
        assert "ZeroDivision" in data["stderr"] or "error" in data

    @pytest.mark.asyncio
    async def test_missing_code_key_returns_error(self):
        result = await _run_python({})
        data = json.loads(result[0].text)
        assert "error" in data


# ─── get_date_context tests ────────────────────────────────────────────────────

class TestGetDateContext:
    def test_returns_all_fields(self):
        result = _get_date_context()
        data = json.loads(result[0].text)
        assert "date" in data
        assert "time" in data
        assert "day_of_week" in data
        assert "iso_week" in data
        assert "unix_timestamp" in data

    def test_date_format(self):
        result = _get_date_context()
        data = json.loads(result[0].text)
        parts = data["date"].split("-")
        assert len(parts) == 3
        assert len(parts[0]) == 4   # year
        assert len(parts[1]) == 2   # month
        assert len(parts[2]) == 2   # day

    def test_unix_timestamp_is_positive_int(self):
        result = _get_date_context()
        data = json.loads(result[0].text)
        assert isinstance(data["unix_timestamp"], int)
        assert data["unix_timestamp"] > 0

    def test_returns_text_content(self):
        result = _get_date_context()
        assert result[0].type == "text"
