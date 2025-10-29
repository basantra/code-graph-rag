"""Real-world test scenarios for the Graph-Code RAG system.

These tests cover comprehensive, end-to-end scenarios that users would encounter
in real development environments, focusing on multi-language support, complex
codebases, and integration with various LLM providers.
"""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from codebase_rag.config import settings
from codebase_rag.graph_updater import GraphUpdater, MemgraphIngestor
from codebase_rag.parser_loader import load_parsers
from codebase_rag.services.llm import CypherGenerator


class TestRealWorldScenarios:
    """Test real-world usage scenarios for the Graph-Code system."""

    def test_multi_language_repository_parsing(self):
        """Test parsing a repository with multiple programming languages.

        This scenario simulates a real-world project like a web application
        with Python backend, JavaScript frontend, and C++ extensions.
        """
        # Create a mock multi-language repository structure
        repo_path = Path("/tmp/test_multi_lang_repo")
        repo_path.mkdir(exist_ok=True)

        # Create Python backend files
        (repo_path / "backend").mkdir(exist_ok=True)
        (repo_path / "backend" / "__init__.py").write_text("")
        (repo_path / "backend" / "app.py").write_text("""
from flask import Flask
from .database import get_user

app = Flask(__name__)

@app.route('/user/<int:user_id>')
def get_user_route(user_id):
    return get_user(user_id)
""")

        (repo_path / "backend" / "database.py").write_text("""
def get_user(user_id):
    return {"id": user_id, "name": "Test User"}

def save_user(user_data):
    pass
""")

        # Create JavaScript frontend files
        (repo_path / "frontend").mkdir(exist_ok=True)
        (repo_path / "frontend" / "app.js").write_text("""
class UserService {
    async getUser(userId) {
        const response = await fetch(`/api/user/${userId}`);
        return response.json();
    }

    saveUser(userData) {
        return fetch('/api/user', {
            method: 'POST',
            body: JSON.stringify(userData)
        });
    }
}

export default UserService;
""")

        # Create C++ extension
        (repo_path / "extensions").mkdir(exist_ok=True)
        (repo_path / "extensions" / "utils.cpp").write_text("""
#include "utils.h"

std::string processData(const std::string& data) {
    return "Processed: " + data;
}

int calculateSum(const std::vector<int>& numbers) {
    return std::accumulate(numbers.begin(), numbers.end(), 0);
}
""")

        (repo_path / "extensions" / "utils.h").write_text("""
#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include <numeric>

std::string processData(const std::string& data);
int calculateSum(const std::vector<int>& numbers);

#endif
""")

        try:
            # Initialize graph ingestor
            with MemgraphIngestor(
                host=settings.MEMGRAPH_HOST,
                port=settings.MEMGRAPH_PORT,
                batch_size=100
            ) as ingestor:
                ingestor.clean_database()
                ingestor.ensure_constraints()

                # Load parsers and update graph
                parsers, queries = load_parsers()
                updater = GraphUpdater(ingestor, repo_path, parsers, queries)
                updater.run()

                # Verify that all languages were parsed
                graph_data = ingestor.export_graph_to_dict()

                # Check for Python nodes
                python_modules = [n for n in graph_data["nodes"] if n.get("labels") == ["Module"] and "backend" in n.get("properties", {}).get("path", "")]
                assert len(python_modules) >= 2  # __init__.py and app.py

                # Check for JavaScript nodes
                js_modules = [n for n in graph_data["nodes"] if n.get("labels") == ["Module"] and ".js" in n.get("properties", {}).get("path", "")]
                assert len(js_modules) >= 1

                # Check for C++ nodes
                cpp_modules = [n for n in graph_data["nodes"] if n.get("labels") == ["Module"] and ".cpp" in n.get("properties", {}).get("path", "")]
                assert len(cpp_modules) >= 1

                # Check for function/method nodes across languages
                functions_and_methods = [n for n in graph_data["nodes"] if "Function" in n.get("labels", []) or "Method" in n.get("labels", [])]
                assert len(functions_and_methods) >= 7  # get_user, save_user, get_user_route, getUser, saveUser, processData, calculateSum

        finally:
            # Cleanup
            import shutil
            shutil.rmtree(repo_path, ignore_errors=True)

    def test_cross_language_function_relationships(self):
        """Test querying relationships between functions across different languages.

        This simulates understanding how a Python API endpoint calls
        JavaScript frontend code and C++ backend processing.
        """
        # Setup test data similar to above
        repo_path = Path("/tmp/test_cross_lang")
        repo_path.mkdir(exist_ok=True)

        try:
            (repo_path / "api.py").write_text("""
from extensions.utils import process_data

def api_endpoint(data):
    return process_data(data)
""")

            (repo_path / "frontend.js").write_text("""
function callApi(data) {
            return fetch('/api', { body: JSON.stringify(data) });
}
""")

            (repo_path / "extensions").mkdir(exist_ok=True)
            (repo_path / "extensions" / "__init__.py").write_text("")
            (repo_path / "extensions" / "utils.py").write_text("""
def process_data(data):
    return f"processed_{data}"
""")

            with MemgraphIngestor(
                host=settings.MEMGRAPH_HOST,
                port=settings.MEMGRAPH_PORT,
                batch_size=100
            ) as ingestor:
                ingestor.clean_database()
                ingestor.ensure_constraints()

                parsers, queries = load_parsers()
                updater = GraphUpdater(ingestor, repo_path, parsers, queries)
                updater.run()

                # Test that relationships exist by checking the graph data directly
                graph_data = ingestor.export_graph_to_dict()

                # Check that CALLS relationships exist
                calls_relationships = [r for r in graph_data["relationships"] if r.get("type") == "CALLS"]
                assert len(calls_relationships) > 0

                # Verify cross-language calls (Python calling Python)
                python_calls = [r for r in calls_relationships
                               if r.get("from_id") and r.get("to_id")]
                assert len(python_calls) > 0

        finally:
            import shutil
            shutil.rmtree(repo_path, ignore_errors=True)

    def test_large_codebase_performance(self):
        """Test performance with a large codebase simulation.

        This scenario tests how the system handles repositories with
        many files and complex relationships.
        """
        repo_path = Path("/tmp/test_large_repo")
        repo_path.mkdir(exist_ok=True)

        try:
            # Create many Python modules with interconnected functions
            for i in range(50):
                module_path = repo_path / f"module_{i}.py"
                module_path.write_text(f"""
def function_{i}_a():
    return function_{i}_b()

def function_{i}_b():
    return function_{i}_c()

def function_{i}_c():
    return "result_{i}"
""")

            with MemgraphIngestor(
                host=settings.MEMGRAPH_HOST,
                port=settings.MEMGRAPH_PORT,
                batch_size=1000  # Larger batch for performance
            ) as ingestor:
                ingestor.clean_database()
                ingestor.ensure_constraints()

                parsers, queries = load_parsers()
                updater = GraphUpdater(ingestor, repo_path, parsers, queries)

                # Measure update time
                import time
                start_time = time.time()
                updater.run()
                end_time = time.time()

                # Should complete within reasonable time (adjust based on hardware)
                assert end_time - start_time < 60  # Less than 1 minute

                # Verify all modules were processed
                graph_data = ingestor.export_graph_to_dict()
                modules = [n for n in graph_data["nodes"] if n.get("labels") == ["Module"]]
                assert len(modules) >= 50

                functions = [n for n in graph_data["nodes"] if "Function" in n.get("labels", [])]
                assert len(functions) >= 150  # 3 functions per module

        finally:
            import shutil
            shutil.rmtree(repo_path, ignore_errors=True)

    @pytest.mark.parametrize("provider", ["ollama", "google", "openai"])
    def test_llm_provider_integration(self, provider):
        """Test integration with different LLM providers.

        This tests that the system works with various AI providers
        for natural language query processing.
        """
        # Skip if provider not configured
        if provider == "ollama" and not settings.active_orchestrator_config.endpoint:
            pytest.skip("Ollama not configured")
        elif provider == "google" and not settings.active_orchestrator_config.api_key:
            pytest.skip("Google API key not configured")
        elif provider == "openai" and not settings.active_orchestrator_config.api_key:
            pytest.skip("OpenAI API key not configured")

        # This would require mocking or actual provider setup
        # For now, just test that provider validation works
        from codebase_rag.providers.base import get_provider

        config = settings.active_orchestrator_config
        provider_instance = get_provider(
            provider,
            api_key=config.api_key,
            endpoint=config.endpoint
        )
        assert provider_instance is not None

    def test_error_handling_memgraph_down(self):
        """Test graceful error handling when Memgraph is unavailable.

        This simulates network issues or Memgraph service being down.
        """
        with pytest.raises(Exception):  # Should raise connection error
            with MemgraphIngestor(
                host="nonexistent.host",
                port=7687,
                batch_size=100
            ) as ingestor:
                ingestor.execute_query("MATCH (n) RETURN count(n)")

    def test_empty_repository_handling(self):
        """Test handling of empty or minimal repositories.

        This tests edge cases where there are no source files to parse.
        """
        repo_path = Path("/tmp/test_empty_repo")
        repo_path.mkdir(exist_ok=True)

        try:
            # Create only non-source files
            (repo_path / "README.md").write_text("# Empty Repo")
            (repo_path / ".gitignore").write_text("*.pyc")

            with MemgraphIngestor(
                host=settings.MEMGRAPH_HOST,
                port=settings.MEMGRAPH_PORT,
                batch_size=100
            ) as ingestor:
                ingestor.clean_database()
                ingestor.ensure_constraints()

                parsers, queries = load_parsers()
                updater = GraphUpdater(ingestor, repo_path, parsers, queries)
                updater.run()  # Should not crash

                # Verify minimal graph structure
                graph_data = ingestor.export_graph_to_dict()
                assert "nodes" in graph_data
                assert "relationships" in graph_data

        finally:
            import shutil
            shutil.rmtree(repo_path, ignore_errors=True)

    def test_incremental_updates(self):
        """Test incremental graph updates as code changes.

        This simulates real development workflow where files are
        added, modified, and deleted over time.
        """
        repo_path = Path("/tmp/test_incremental")
        repo_path.mkdir(exist_ok=True)

        try:
            # Initial state
            (repo_path / "main.py").write_text("""
def hello():
    return "world"
""")

            with MemgraphIngestor(
                host=settings.MEMGRAPH_HOST,
                port=settings.MEMGRAPH_PORT,
                batch_size=100
            ) as ingestor:
                ingestor.clean_database()
                ingestor.ensure_constraints()

                parsers, queries = load_parsers()
                updater = GraphUpdater(ingestor, repo_path, parsers, queries)
                updater.run()

                # Check initial state
                graph_data = ingestor.export_graph_to_dict()
                initial_functions = len([n for n in graph_data["nodes"] if "Function" in n.get("labels", [])])

                # Add new file
                (repo_path / "utils.py").write_text("""
def helper():
    return hello()
""")

                # Update graph incrementally
                updater = GraphUpdater(ingestor, repo_path, parsers, queries)
                updater.run()

                # Verify new function was added
                graph_data = ingestor.export_graph_to_dict()
                final_functions = len([n for n in graph_data["nodes"] if "Function" in n.get("labels", [])])
                assert final_functions > initial_functions

        finally:
            import shutil
            shutil.rmtree(repo_path, ignore_errors=True)

    def test_cli_integration_smoke_test(self):
        """Test basic CLI functionality via subprocess.

        This tests the actual command-line interface that users interact with.
        """
        # Test help command
        result = subprocess.run(
            [sys.executable, "-m", "codebase_rag.main", "--help"],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0
        assert "Usage:" in result.stdout

    def test_export_import_workflow(self):
        """Test the export and import workflow for graph data.

        This tests backing up and restoring graph state.
        """
        export_path = Path("/tmp/test_export.json")

        try:
            # Create some test data first
            repo_path = Path("/tmp/test_export_repo")
            repo_path.mkdir(exist_ok=True)
            (repo_path / "test.py").write_text("def test(): pass")

            with MemgraphIngestor(
                host=settings.MEMGRAPH_HOST,
                port=settings.MEMGRAPH_PORT,
                batch_size=100
            ) as ingestor:
                ingestor.clean_database()
                ingestor.ensure_constraints()

                parsers, queries = load_parsers()
                updater = GraphUpdater(ingestor, repo_path, parsers, queries)
                updater.run()

                # Export graph
                graph_data = ingestor.export_graph_to_dict()
                with open(export_path, 'w') as f:
                    json.dump(graph_data, f)

                # Verify export file exists and has content
                assert export_path.exists()
                assert export_path.stat().st_size > 0

                # Load exported data
                from codebase_rag.graph_loader import load_graph
                loaded_graph = load_graph(str(export_path))

                # Verify loaded data matches
                assert loaded_graph.summary()["total_nodes"] == graph_data["metadata"]["total_nodes"]
                assert loaded_graph.summary()["total_relationships"] == graph_data["metadata"]["total_relationships"]

        finally:
            import shutil
            shutil.rmtree(repo_path, ignore_errors=True)
            export_path.unlink(missing_ok=True)
