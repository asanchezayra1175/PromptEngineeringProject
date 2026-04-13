"""
tool_executor.py

Shared agentic loop used by all agents (Agent 1, 2, and 3).

Supports two execution modes controlled by the BATCH_MODE env var:

  BATCH_MODE=false (default)
    Sequential mode. Each instance runs one full agentic loop at a time.
    Best for prompt development and debugging — you see results immediately.

  BATCH_MODE=true
    Turn-level batch mode. All instances run their Nth turn together in a
    single Batch API call, then wait for results before proceeding to turn N+1.
    Gives 50% cost reduction at the expense of throughput latency (~1hr/batch).
    Best for full dev/test set evaluation runs.

Context window cost control:
  SUMMARIZE_EVERY  : summarize and restart the conversation every N tool calls
                     to prevent the growing history from inflating input token
                     costs. Set to 0 to disable. Default: 8.

To add a new tool:
  1. Define it as a ToolDefinition below.
  2. Pass it in the `tools` list when calling run_agent_loop() or run_batch_agent_loop().
"""

from dotenv import load_dotenv
load_dotenv()

from logger import get_logger
import anthropic
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import docker

MODEL            = os.environ.get("MODEL", "claude-haiku-4-5-20251001")

# Per-agent model overrides — fall back to MODEL if not set
AGENT1_MODEL     = os.environ.get("AGENT1_MODEL", MODEL)
AGENT2_MODEL     = os.environ.get("AGENT2_MODEL", MODEL)
AGENT3_MODEL     = os.environ.get("AGENT3_MODEL", MODEL)
MAX_TOKENS       = 8096
MAX_ITERATIONS   = int(os.environ.get("MAX_ITERATIONS", 10))
BATCH_MODE       = os.environ.get("BATCH_MODE", "false").lower() == "true"
BATCH_POLL_INTERVAL = int(os.environ.get("BATCH_POLL_INTERVAL", 60))
SUMMARIZE_EVERY  = int(os.environ.get("SUMMARIZE_EVERY", 0))  # disabled by default — only useful for loops > 20 iterations  # 0 = disabled


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
    name:        str
    description: str
    schema:      dict[str, Any]
    handler:     Callable[[dict[str, Any]], str]

    def to_api_dict(self) -> dict:
        """Serialize to the format the Anthropic API expects."""
        return {
            "name":         self.name,
            "description":  self.description,
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
    """
    def handler(tool_input: dict[str, Any]) -> str:
        command = tool_input["command"]
        exit_code, output = container.exec_run(
            ["bash", "-lc", command],
            workdir="/testbed",
            demux=False,
        )
        output_str = output.decode("utf-8", errors="replace").strip()
        # Truncate very long outputs to avoid runaway context growth
        if len(output_str) > 8000:
            output_str = output_str[:5000] + "\n...[truncated]...\n" + output_str[-2000:]
        return f"[exit_code={exit_code}]\n{output_str}"

    return ToolDefinition(
        name="bash",
        description=(
            "Run a shell command inside the Docker container at /testbed. "
            "The container has the repository checked out at the base commit. "
            "Commands are executed via bash -lc so the full environment is "
            "initialized. Use this to explore the repo, read files, run tests, "
            "and write files. Always read a file before assuming its contents. "
            "stdout and stderr are merged in the output."
        ),
        schema={
            "type": "object",
            "properties": {
                "command": {
                    "type":        "string",
                    "description": "The shell command to run inside the container.",
                }
            },
            "required": ["command"],
        },
        handler=handler,
    )


# ---------------------------------------------------------------------------
# RepairAgent-inspired tools for the Architect
# ---------------------------------------------------------------------------

def make_get_classes_and_methods_tool(container) -> ToolDefinition:
    """
    List all class and method/function names with line numbers in a Python file.
    Lets the Architect understand file structure without reading everything.
    """
    def handler(tool_input: dict[str, Any]) -> str:
        file_path = tool_input["file_path"]
        script = (
            "import ast; "
            "src = open('" + file_path + "').read(); "
            "tree = ast.parse(src); "
            "nodes = sorted(ast.walk(tree), key=lambda n: getattr(n, 'lineno', 0)); "
            "[print(f'class {n.name} (line {n.lineno})') "
            " for n in nodes if isinstance(n, ast.ClassDef)]; "
            "[print(f'  def {n.name} (line {n.lineno})') "
            " for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]"
        )
        exit_code, output = container.exec_run(
            ["python3", "-c", script],
            workdir="/testbed",
            demux=False,
        )
        output_str = output.decode("utf-8", errors="replace").strip()
        if exit_code != 0 or not output_str:
            # Fallback to grep
            exit_code, output = container.exec_run(
                ["bash", "-lc", f"grep -n 'class \\|def ' {file_path}"],
                workdir="/testbed",
                demux=False,
            )
            output_str = output.decode("utf-8", errors="replace").strip()
        return f"[exit_code={exit_code}]\n{output_str}"

    return ToolDefinition(
        name="get_classes_and_methods",
        description=(
            "List all class and method/function names with line numbers from a Python file. "
            "Use this to orient yourself in a large file before deciding which method to read. "
            "More efficient than cat when you only need the file structure."
        ),
        schema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the Python file, e.g. /testbed/src/module/file.py",
                }
            },
            "required": ["file_path"],
        },
        handler=handler,
    )


def make_extract_method_tool(container) -> ToolDefinition:
    """
    Extract the full source of a named method/function from a file.
    Finds the correct line range automatically — no need to know line numbers.
    """
    def handler(tool_input: dict[str, Any]) -> str:
        file_path   = tool_input["file_path"]
        method_name = tool_input["method_name"]
        script = (
            "import ast; "
            "src = open('" + file_path + "').read(); "
            "lines = src.splitlines(keepends=True); "
            "tree = ast.parse(src); "
            "matches = [n for n in ast.walk(tree) "
            "           if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) "
            "           and n.name == '" + method_name + "']; "
            "[print(f'--- {m.name} (lines {m.lineno}-{m.end_lineno}) ---\\n' "
            "      + ''.join(lines[m.lineno-1:m.end_lineno])) for m in matches] "
            "or print('Method not found: " + method_name + "')"
        )
        exit_code, output = container.exec_run(
            ["python3", "-c", script],
            workdir="/testbed",
            demux=False,
        )
        output_str = output.decode("utf-8", errors="replace").strip()
        if len(output_str) > 6000:
            output_str = output_str[:6000] + "\n...[truncated]..."
        return f"[exit_code={exit_code}]\n{output_str}"

    return ToolDefinition(
        name="extract_method",
        description=(
            "Extract the full source code of a named method or function from a Python file. "
            "Returns the exact lines including the def signature and body. "
            "Use this instead of sed line ranges — it finds the correct range automatically "
            "even if line numbers have shifted. Handles multiple methods with the same name."
        ),
        schema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the Python file.",
                },
                "method_name": {
                    "type": "string",
                    "description": "Name of the method or function to extract (just the name, not class.method).",
                },
            },
            "required": ["file_path", "method_name"],
        },
        handler=handler,
    )


def make_find_similar_api_calls_tool(container) -> ToolDefinition:
    """
    Find all call sites of a method/function across the codebase.
    Prevents hallucinated API usage by showing real examples from the repo.
    """
    def handler(tool_input: dict[str, Any]) -> str:
        method_name = tool_input["method_name"]
        search_path = tool_input.get("search_path", "/testbed")
        command = (
            f"grep -rn '{method_name}(' {search_path} "
            f"--include='*.py' --exclude-dir='.git' -A 1 2>/dev/null | head -80"
        )
        exit_code, output = container.exec_run(
            ["bash", "-lc", command],
            workdir="/testbed",
            demux=False,
        )
        output_str = output.decode("utf-8", errors="replace").strip()
        if not output_str:
            output_str = f"No calls to {method_name}() found in {search_path}"
        if len(output_str) > 5000:
            output_str = output_str[:5000] + "\n...[truncated]..."
        return f"[exit_code={exit_code}]\n{output_str}"

    return ToolDefinition(
        name="find_similar_api_calls",
        description=(
            "Find all places in the codebase that call a specific method or function. "
            "Use this when you are about to change a function's return type or signature "
            "to find all downstream callers that may also need updating. "
            "Also useful to understand how an API is called in practice before proposing "
            "a fix. Shows up to 80 results with one line of context each."
        ),
        schema={
            "type": "object",
            "properties": {
                "method_name": {
                    "type": "string",
                    "description": "Name of the method or function to search for (e.g. 'fuentes', 'LintResult').",
                },
                "search_path": {
                    "type": "string",
                    "description": "Directory to search. Defaults to /testbed (entire repo). Narrow with a subdirectory.",
                },
            },
            "required": ["method_name"],
        },
        handler=handler,
    )


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    """
    The outcome of a completed agent loop (sequential or batch).

    Attributes:
        instance_id    : SWE-bench instance this run was for
        final_response : Claude's last text response (the agent's conclusion)
        messages       : full conversation history, useful for debugging
        iterations     : how many turns were made
        success        : False if the loop hit MAX_ITERATIONS without end_turn
    """
    instance_id:    str
    final_response: str
    messages:       list[dict]
    iterations:     int
    success:        bool


# ---------------------------------------------------------------------------
# Mid-loop summarization
# ---------------------------------------------------------------------------

def _summarize_conversation(
    client: anthropic.Anthropic,
    system_prompt: str,
    messages: list[dict],
    instance_id: str,
    verbose: bool,
) -> list[dict]:
    """
    Compress the current conversation history into a compact summary and
    return a fresh message list containing only the summary as context.

    This prevents the accumulated tool call history from inflating input
    token costs on every subsequent API call. Called automatically every
    SUMMARIZE_EVERY tool calls when SUMMARIZE_EVERY > 0.

    The summary asks Claude to capture:
      - What has been explored so far
      - Key findings (relevant files, functions, error messages)
      - What has been tried and what the results were
      - What still needs to be done

    The fresh conversation restarts with the system prompt intact and the
    summary as the first user message, so the agent retains full context
    of its progress without paying for the full tool history.
    """
    if verbose:
        get_logger().info(f"[{instance_id}] summarizing conversation to compress context...")

    # Build a summary request from the current history
    summary_messages = messages + [{
        "role": "user",
        "content": (
            "Please provide a concise summary of your progress so far. Include:\n"
            "1. What you have explored and key findings (relevant files, functions, classes)\n"
            "2. What you now understand about the bug\n"
            "3. What approaches you have tried and their results\n"
            "4. Exactly what still needs to be done to complete your task\n\n"
            "Be specific — include file paths, function names, and error messages. "
            "This summary will replace the conversation history to save context space."
        )
    }]

    summary_response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=system_prompt,
        messages=summary_messages,
    )

    summary_text = next(
        (block.text for block in summary_response.content if hasattr(block, "text")),
        "No summary available.",
    )

    if verbose:
        get_logger().info(f"[{instance_id}] context compressed. Summary ({len(summary_text)} chars):\n{summary_text[:300]}...")

    # Return a fresh message list with just the summary as context
    return [{
        "role": "user",
        "content": (
            f"## Progress summary (conversation history compressed to save context)\n\n"
            f"{summary_text}\n\n"
            "Continue from where you left off based on this summary."
        )
    }, {
        "role": "assistant",
        "content": "Understood. I'll continue from where I left off based on the summary above."
    }]


# ---------------------------------------------------------------------------
# Sequential agentic loop
# ---------------------------------------------------------------------------

def run_agent_loop(
    *,
    instance_id:     str,
    system_prompt:   str,
    initial_message: str,
    tools:           list[ToolDefinition],
    model:           str = MODEL,
    max_iterations:  int = MAX_ITERATIONS,
    verbose:         bool = True,
) -> AgentResult:
    """
    Sequential agentic loop — one instance, one turn at a time.

    Use this during prompt development. Each tool call is executed
    immediately and results are fed back before the next API call.

    When SUMMARIZE_EVERY > 0, the conversation history is compressed
    every SUMMARIZE_EVERY tool calls to control input token costs.

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
    client    = anthropic.Anthropic(
        max_retries=8,   # retry up to 8 times on 429/500 with exponential backoff
        timeout=120.0,   # longer timeout accommodates rate-limit retry delays
    )
    api_tools = [t.to_api_dict() for t in tools]
    tool_map  = {t.name: t.handler for t in tools}
    messages  = [{"role": "user", "content": initial_message}]
    tool_call_count = 0

    for iteration in range(max_iterations):
        if verbose:
            get_logger().info(f"\n[{instance_id}] iteration {iteration + 1}/{max_iterations}")

        response = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            tools=api_tools,
            messages=messages,
        )

        if verbose:
            get_logger().info(f"[{instance_id}] stop_reason={response.stop_reason}")

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            final_text = next(
                (block.text for block in response.content if hasattr(block, "text")),
                "",
            )
            if verbose:
                get_logger().info(f"[{instance_id}] completed in {iteration + 1} iteration(s).")
            return AgentResult(
                instance_id=instance_id,
                final_response=final_text,
                messages=messages,
                iterations=iteration + 1,
                success=True,
            )

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                if verbose:
                    get_logger().info(f"[{instance_id}] tool_use : {block.name}({json.dumps(block.input)[:120]})")
                handler = tool_map.get(block.name)
                result  = handler(block.input) if handler else f"[error] Unknown tool '{block.name}'"
                if verbose:
                    get_logger().info(f"[{instance_id}] tool_result: {result[:500]}")
                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": block.id,
                    "content":     result,
                })
                tool_call_count += 1

            messages.append({"role": "user", "content": tool_results})

            # Mid-loop summarization — compress history every SUMMARIZE_EVERY tool calls
            if SUMMARIZE_EVERY > 0 and tool_call_count > 0 and tool_call_count % SUMMARIZE_EVERY == 0:
                messages = _summarize_conversation(
                    client, system_prompt, messages, instance_id, verbose
                )

            # Forced output nudge — if we are 2 iterations from the cap and the
            # agent is still exploring, inject a message forcing it to produce output.
            # This prevents the agent spending all iterations on exploration and
            # hitting MAX_ITERATIONS without ever writing a test or patch.
            if iteration == max_iterations - 4:
                if verbose:
                    get_logger().info(f"[{instance_id}] WARNING: approaching iteration cap — injecting output nudge.")
                messages.append({
                    "role": "user",
                    "content": (
                        "IMPORTANT: You are running out of iterations. "
                        "Stop exploring immediately. "
                        "Based on everything you have found so far, write the reproduction test NOW "
                        "and output the required JSON block. Do not make any more exploratory tool calls."
                    )
                })

    if verbose:
        get_logger().info(f"[{instance_id}] WARNING: hit max_iterations ({max_iterations}).")

    return AgentResult(
        instance_id=instance_id,
        final_response="",
        messages=messages,
        iterations=max_iterations,
        success=False,
    )


# ---------------------------------------------------------------------------
# Batch agentic loop — turn-level batching across multiple instances
# ---------------------------------------------------------------------------

@dataclass
class _InstanceState:
    """
    Internal state for one instance during a batch run.
    Tracks the conversation history and whether the instance is done.
    """
    instance_id:    str
    system_prompt:  str
    tool_map:       dict[str, Callable]
    messages:       list[dict] = field(default_factory=list)
    iterations:     int = 0
    tool_call_count: int = 0
    done:           bool = False
    final_text:     str = ""
    success:        bool = False


def _poll_batch(client: anthropic.Anthropic, batch_id: str, verbose: bool) -> Any:
    """
    Poll the Batch API until processing ends, then return the completed batch.
    Checks every BATCH_POLL_INTERVAL seconds.
    """
    if verbose:
        get_logger().info(f"[batch] submitted batch {batch_id} — polling every {BATCH_POLL_INTERVAL}s...")

    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            return batch
        if verbose:
            counts = batch.request_counts
            get_logger().info(
                f"[batch] {batch_id} still processing — "
                f"processing={counts.processing} succeeded={counts.succeeded} "
                f"errored={counts.errored}"
            )
        time.sleep(BATCH_POLL_INTERVAL)


def run_batch_agent_loop(
    *,
    instances:      list[dict],
    max_iterations: int = MAX_ITERATIONS,
    verbose:        bool = True,
) -> list[AgentResult]:
    # Note: in batch mode, each instance dict can include a "model" key
    # to specify per-instance model. Falls back to global MODEL if not set.
    """
    Turn-level batch agentic loop — all instances advance one turn per batch.

    Includes mid-loop summarization: when an instance's tool_call_count
    reaches a multiple of SUMMARIZE_EVERY, its history is compressed
    before the next batch turn.

    Args:
        instances      : list of dicts, one per instance, each with:
                           - instance_id    : str
                           - system_prompt  : str
                           - initial_message: str
                           - tools          : list[ToolDefinition]
        max_iterations : max turns per instance before giving up
        verbose        : print batch/tool logs to stdout

    Returns:
        list of AgentResult, one per instance, in the same order as input
    """
    client = anthropic.Anthropic()

    states: dict[str, _InstanceState] = {}
    for inst in instances:
        instance_id = inst["instance_id"]
        tool_map    = {t.name: t.handler for t in inst["tools"]}
        state       = _InstanceState(
            instance_id=instance_id,
            system_prompt=inst["system_prompt"],
            tool_map=tool_map,
            messages=[{"role": "user", "content": inst["initial_message"]}],
        )
        states[instance_id] = state

    api_tools_map = {
        inst["instance_id"]: [t.to_api_dict() for t in inst["tools"]]
        for inst in instances
    }

    for turn in range(max_iterations):
        active = [s for s in states.values() if not s.done]
        if not active:
            break

        if verbose:
            get_logger().info(f"\n[batch] turn {turn + 1}/{max_iterations} — {len(active)} active instance(s)")

        # Apply mid-loop summarization for any instances due for compression
        if SUMMARIZE_EVERY > 0:
            for state in active:
                if state.tool_call_count > 0 and state.tool_call_count % SUMMARIZE_EVERY == 0:
                    state.messages = _summarize_conversation(
                        client, state.system_prompt, state.messages,
                        state.instance_id, verbose
                    )

        batch_requests = []
        for state in active:
            batch_requests.append({
                "custom_id": state.instance_id,
                "params": {
                    "model":      inst.get("model", MODEL),
                    "max_tokens": MAX_TOKENS,
                    "system":     state.system_prompt,
                    "tools":      api_tools_map[state.instance_id],
                    "messages":   state.messages,
                },
            })

        batch     = client.messages.batches.create(requests=batch_requests)
        _poll_batch(client, batch.id, verbose)

        for result in client.messages.batches.results(batch.id):
            instance_id = result.custom_id
            state       = states[instance_id]
            state.iterations += 1

            if result.result.type != "succeeded":
                if verbose:
                    get_logger().info(f"[batch] {instance_id} result type={result.result.type} — marking done.")
                state.done    = True
                state.success = False
                continue

            response = result.result.message
            state.messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                state.final_text = next(
                    (block.text for block in response.content if hasattr(block, "text")),
                    "",
                )
                state.done    = True
                state.success = True
                if verbose:
                    get_logger().info(f"[batch] {instance_id} completed at turn {turn + 1}.")
                continue

            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if not hasattr(block, "type") or block.type != "tool_use":
                        continue
                    if verbose:
                        get_logger().info(f"[batch] {instance_id} tool_use: {block.name}({json.dumps(block.input)[:120]})")
                    handler    = state.tool_map.get(block.name)
                    result_str = handler(block.input) if handler else f"[error] Unknown tool '{block.name}'"
                    if verbose:
                        get_logger().info(f"[batch] {instance_id} tool_result: {result_str[:500]}")
                    tool_results.append({
                        "type":        "tool_result",
                        "tool_use_id": block.id,
                        "content":     result_str,
                    })
                    state.tool_call_count += 1
                state.messages.append({"role": "user", "content": tool_results})

    for state in states.values():
        if not state.done:
            if verbose:
                get_logger().info(f"[batch] {state.instance_id} hit max_iterations ({max_iterations}) — marking failed.")
            state.done    = True
            state.success = False

    return [
        AgentResult(
            instance_id=s.instance_id,
            final_response=s.final_text,
            messages=s.messages,
            iterations=s.iterations,
            success=s.success,
        )
        for s in [states[inst["instance_id"]] for inst in instances]
    ]


# ---------------------------------------------------------------------------
# Unified entry point — respects BATCH_MODE env var
# ---------------------------------------------------------------------------

def run_agent_loop_auto(
    *,
    instance_id:     str,
    system_prompt:   str,
    initial_message: str,
    tools:           list[ToolDefinition],
    model:           str = MODEL,
    max_iterations:  int = MAX_ITERATIONS,
    verbose:         bool = True,
) -> AgentResult:
    """
    Drop-in replacement for run_agent_loop() that respects BATCH_MODE.

    In sequential mode (BATCH_MODE=false): delegates to run_agent_loop().
    In batch mode (BATCH_MODE=true): wraps the single instance in run_batch_agent_loop().
    """
    if not BATCH_MODE:
        return run_agent_loop(
            instance_id=instance_id,
            system_prompt=system_prompt,
            initial_message=initial_message,
            tools=tools,
            model=model,
            max_iterations=max_iterations,
            verbose=verbose,
        )
    else:
        results = run_batch_agent_loop(
            instances=[{
                "instance_id":     instance_id,
                "system_prompt":   system_prompt,
                "initial_message": initial_message,
                "tools":           tools,
                "model":           model,
            }],
            max_iterations=max_iterations,
            verbose=verbose,
        )
        return results[0]