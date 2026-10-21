"""FastMCP tool server (local stdio):  uv run python -m tools.mcp_server

Read-only, allowlisted tools. Tool outputs are untrusted data for the agent.
"""
from mcp.server.fastmcp import FastMCP

from tools import hf_client

mcp = FastMCP("hf-discovery")


@mcp.tool()
def search_models(query: str | None = None, task: str | None = None, library: str | None = None,
                  language: str | None = None, license: str | None = None,
                  sort: str = "downloads", limit: int = 10) -> dict:
    """Search Hugging Face models. Filters: task (pipeline tag), library, language code, license id.
    Returns live results with fetched_at; downloads and likes are time-sensitive."""
    return hf_client.search_models(query, task=task, library=library, language=language,
                                   license=license, sort=sort, limit=limit)


@mcp.tool()
def search_embedding_models(query: str | None = None, language: str | None = None,
                            sort: str = "downloads", limit: int = 10) -> dict:
    """Search sentence-embedding models (sentence-similarity, sentence-transformers library)."""
    return hf_client.search_embedding_models(query, language=language, sort=sort, limit=limit)


@mcp.tool()
def get_model_details(repo_id: str, revision: str | None = None) -> dict:
    """Get metadata for one model: revision (sha), license, languages, parameter count, files."""
    return hf_client.get_model_details(repo_id, revision)


@mcp.tool()
def search_datasets(query: str | None = None, task: str | None = None, language: str | None = None,
                    license: str | None = None, sort: str = "downloads", limit: int = 10) -> dict:
    """Search Hugging Face datasets, optionally filtered by language and license id."""
    return hf_client.search_datasets(query, language=language, license=license, sort=sort, limit=limit)


@mcp.tool()
def get_dataset_details(repo_id: str, revision: str | None = None) -> dict:
    """Get metadata for one dataset: revision (sha), license, files."""
    return hf_client.get_dataset_details(repo_id, revision)


@mcp.tool()
def search_docs(query: str, category: str | None = None, k: int = 5) -> list[dict]:
    """Search the indexed, license-checked Hugging Face documentation. Returns cited passages."""
    from src import retrieval

    if not 1 <= k <= 20:
        raise ValueError("k must be between 1 and 20")
    return retrieval.search_knowledge(query, category=category, k=k)


if __name__ == "__main__":
    mcp.run()
