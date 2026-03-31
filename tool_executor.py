"""
tool_executor.py

Shared agentic loop used by all agents (Agent 1, 2, and 3).

To add a new tool:
  1. Define it as a ToolDefinition (schema + handler) below the built-ins.
  2. Pass it in the `tools` list when calling `run_agent_loop()`.

The loop handles all tool_use / tool_result turns automatically. Each
agent only needs to provide its system prompt, initial message, and the
subset of tools it is allowed to use.
"""

import anthropic
import json
from dataclasses import dataclass, field
from typing import Any, Callable

import docker

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 8096
MAX_ITERATIONS = 30  # hard cap — prevents runaway loops


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

@dataclass
class ToolDefinition:
    """
    A single tool available to an agent.

    Attributes:
        name        : must match the name in schema exactly
        description : shown to Claude — be explicit about side effects
        schema      : JSON Schema for the tool's input parameters
        handler     : callable(input: dict) -> str, returns output shown to Claude
    """
    name: str
    description: str
    schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], str]

    def to_api_dict(self) -> dict:
        """Serialize to the format the Anthropic API expects."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.schema,
        }


# ---------------------------------------------------------------------------
# Built-in tool: bash
# ---------------------------------------------------------------------------

def make_bash_tool(container) -> ToolDefinition:
    """
    Factory that binds a Docker container to the bash tool handler.

    All commands are run via bash -lc to ensure the full container
    environment is initialized (PATH, site-packages, editable installs).
    stdout and stderr are merged and prefixed with the exit code so
    Claude can reason about failures.

    Args:
        container: a running docker.models.containers.Container

    Returns:
        ToolDefinition ready to pass to run_agent_loop()
    """
    def handler(tool_input: dict[str, Any]) -> str:
        command = tool_input["command"]
        exit_code, output = container.exec_run(
            ["bash", "-lc", command],
            workdir="/testbed",
            demux=False,  # merge stdout + stderr
        )
        output_str = output.decode("utf-8", errors="replace").strip()
        return f"[exit_code={exit_code}]\n{output_str}"

    return ToolDefinition(
        name="bash",
        description=(
            "Run a shell command inside the Docker container at /testbed. "
            "The container has the repository checked out at the base commit. "
            "Commands are executed via bash -lc so the full environment is "
            "initialized (PATH, site-packages, editable installs, etc.). "
            "Use this to explore the repo, read files, run tests, and write "
            "files. Always read a file before assuming its contents. "
            "stdout and stderr are merged in the output."
        ),
        schema={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to run inside the container.",
                }
            },
            "required": ["command"],
        },
        handler=handler,
    )


# ---------------------------------------------------------------------------
# Agentic loop
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    """
    The outcome of a completed agent loop.

    Attributes:
        instance_id    : SWE-bench instance this run was for
        final_response : Claude's last text response (the agent's conclusion)
        messages       : full conversation history, useful for debugging
        iterations     : how many API calls were made
        success        : False if the loop hit MAX_ITERATIONS without end_turn
    """
    instance_id: str
    final_response: str
    messages: list[dict]
    iterations: int
    success: bool


def run_agent_loop(
    *,
    instance_id: str,
    system_prompt: str,
    initial_message: str,
    tools: list[ToolDefinition],
    max_iterations: int = MAX_ITERATIONS,
    verbose: bool = True,
) -> AgentResult:
    """
    Generic agentic loop shared by all agents.

    Sends the initial message to Claude with the provided tools, then
    processes tool_use / tool_result turns until Claude emits end_turn
    or the iteration cap is hit.

    Args:
        instance_id     : used for logging and the returned AgentResult
        system_prompt   : the agent's persona and instructions
        initial_message : the first user message (context bundle as text)
        tools           : list of ToolDefinition the agent may use
        max_iterations  : override the global cap if needed
        verbose         : print iteration/tool logs to stdout

    Returns:
        AgentResult with the final response and full message history
    """
    client = anthropic.Anthropic()
    api_tools = [t.to_api_dict() for t in tools]
    tool_map = {t.name: t.handler for t in tools}

    messages = [{"role": "user", "content": initial_message}]

    for iteration in range(max_iterations):
        if verbose:
            print(f"\n[{instance_id}] iteration {iteration + 1}/{max_iterations}")

        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            tools=api_tools,
            messages=messages,
        )

        if verbose:
            print(f"[{instance_id}] stop_reason={response.stop_reason}")

        # Append Claude's full response (may contain text + tool_use blocks)
        messages.append({"role": "assistant", "content": response.content})

        # Claude is done — extract the final text response
        if response.stop_reason == "end_turn":
            final_text = next(
                (block.text for block in response.content if hasattr(block, "text")),
                "",
            )
            if verbose:
                print(f"[{instance_id}] completed in {iteration + 1} iteration(s).")
            return AgentResult(
                instance_id=instance_id,
                final_response=final_text,
                messages=messages,
                iterations=iteration + 1,
                success=True,
            )

        # Claude wants to use tools — execute each and collect results
        if response.stop_reason == "tool_use":
            tool_results = []

            for block in response.content:
                if block.type != "tool_use":
                    continue

                if verbose:
                    print(f"[{instance_id}] tool_use : {block.name}({json.dumps(block.input)[:120]})")

                handler = tool_map.get(block.name)
                if handler is None:
                    result = f"[error] Unknown tool '{block.name}'"
                else:
                    result = handler(block.input)

                if verbose:
                    print(f"[{instance_id}] tool_result: {result[:200]}")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

            # Return all tool results in a single user turn
            messages.append({"role": "user", "content": tool_results})

    # Hit the iteration cap without a clean end_turn
    if verbose:
        print(f"[{instance_id}] WARNING: hit max_iterations ({max_iterations}) without end_turn.")

    return AgentResult(
        instance_id=instance_id,
        final_response="",
        messages=messages,
        iterations=max_iterations,
        success=False,
    )