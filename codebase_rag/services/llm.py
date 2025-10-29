from loguru import logger
from pydantic_ai import Agent, Tool

from ..config import settings
from ..prompts import (
    CYPHER_SYSTEM_PROMPT,
    LOCAL_CYPHER_SYSTEM_PROMPT,
    RAG_ORCHESTRATOR_SYSTEM_PROMPT,
)
from ..providers.base import get_provider


class LLMGenerationError(Exception):
    """Custom exception for LLM generation failures."""

    pass


def _clean_cypher_response(response_text: str) -> str:
    """Utility to clean up common LLM formatting artifacts from a Cypher query."""
    import re

    # Remove markdown code blocks
    # Look for ```cypher ... ``` or ``` ... ```
    cypher_block_pattern = r'```(?:cypher)?\s*\n?(.*?)\n?```'
    match = re.search(cypher_block_pattern, response_text, re.DOTALL | re.IGNORECASE)
    if match:
        query = match.group(1).strip()
    else:
        # No code block found, try to extract from text
        query = response_text.strip()

    # Remove backticks and other artifacts
    query = query.replace("`", "").strip()

    # Remove "cypher" prefix if present
    if query.lower().startswith("cypher"):
        query = query[6:].strip()

    # Try to find the actual query by looking for Cypher keywords
    lines = query.split('\n')
    cypher_lines = []
    in_query = False

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Stop if we hit explanation text (common patterns)
        if in_query and re.match(r'^(This|You can|The query|To list|Here is)', line, re.IGNORECASE):
            break
        # Check if line starts with Cypher keywords
        if re.match(r'^(MATCH|WITH|WHERE|RETURN|CREATE|DELETE|SET|MERGE|OPTIONAL|PROFILE|EXPLAIN|UNWIND|FOREACH|CALL|START|USING|ORDER BY|LIMIT|SKIP|DISTINCT)\b', line.upper()):
            in_query = True
        if in_query:
            cypher_lines.append(line)

    if cypher_lines:
        query = ' '.join(cypher_lines)
    else:
        # Fallback: just clean the original query
        query = query

    # Ensure it ends with semicolon
    query = query.rstrip(';').strip() + ";"

    return query


class CypherGenerator:
    """Generates Cypher queries from natural language."""

    def __init__(self) -> None:
        try:
            # Get active cypher model configuration
            config = settings.active_cypher_config

            # Create provider instance
            provider = get_provider(
                config.provider,
                api_key=config.api_key,
                endpoint=config.endpoint,
                project_id=config.project_id,
                region=config.region,
                provider_type=config.provider_type,
                thinking_budget=config.thinking_budget,
            )

            # Create model using provider
            llm = provider.create_model(config.model_id)

            # Select system prompt based on provider
            system_prompt = (
                LOCAL_CYPHER_SYSTEM_PROMPT
                if config.provider == "ollama"
                else CYPHER_SYSTEM_PROMPT
            )

            self.agent = Agent(
                model=llm,
                system_prompt=system_prompt,
                output_type=str,
            )
        except Exception as e:
            raise LLMGenerationError(
                f"Failed to initialize CypherGenerator: {e}"
            ) from e

    async def generate(self, natural_language_query: str) -> str:
        logger.info(
            f"  [CypherGenerator] Generating query for: '{natural_language_query}'"
        )
        try:
            result = await self.agent.run(natural_language_query)
            if not isinstance(result.output, str):
                raise LLMGenerationError(
                    f"LLM did not generate a valid query. Output: {result.output}"
                )

            query = _clean_cypher_response(result.output)

            # Validate the cleaned query - be very strict
            query_upper = query.upper()
            if not ("MATCH" in query_upper and "RETURN" in query_upper):
                raise LLMGenerationError(
                f"Output does not contain both MATCH and RETURN keywords. This is not a valid Cypher query. Output: {query}"
            )

            # Check that it's not explanatory text
            if any(phrase in query.lower() for phrase in [
                "the database is structured",
                "natural language prompts",
                "here are some examples",
                "you can use",
                "the query",
                "to find",
            ]):
                raise LLMGenerationError(
                    f"LLM returned explanatory text instead of a Cypher query. Output: {query}"
                )
            logger.info(f"  [CypherGenerator] Raw LLM output: {result.output}")
            logger.info(f"  [CypherGenerator] Cleaned Cypher: {query}")
            return query
        except Exception as e:
            logger.error(f"  [CypherGenerator] Error: {e}")
            raise LLMGenerationError(f"Cypher generation failed: {e}") from e


def create_rag_orchestrator(tools: list[Tool]) -> Agent:
    """Factory function to create the main RAG orchestrator agent."""
    try:
        # Get active orchestrator model configuration
        config = settings.active_orchestrator_config

        # Create provider instance
        provider = get_provider(
            config.provider,
            api_key=config.api_key,
            endpoint=config.endpoint,
            project_id=config.project_id,
            region=config.region,
            provider_type=config.provider_type,
            thinking_budget=config.thinking_budget,
        )

        # Create model using provider
        llm = provider.create_model(config.model_id)

        return Agent(
            model=llm,
            system_prompt=RAG_ORCHESTRATOR_SYSTEM_PROMPT,
            tools=tools,
        )
    except Exception as e:
        raise LLMGenerationError(f"Failed to initialize RAG Orchestrator: {e}") from e
