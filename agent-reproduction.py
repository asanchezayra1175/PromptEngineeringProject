"""
agent1_reproduction.py

Agent 1 — Issue Reproduction

Responsibilities:
  - Explore the repository to understand the bug
  - Write a reproduction test that confirms the bug is present
  - Verify the test fails with an AssertionError (not an env/import error)
  - Persist all outputs to a JSON state file for crash recovery and auditing

Environment variables (set in .env):
  AGENT1_MAX_RETRIES   : max times Agent 1 will rewrite the test (default: 3)
  AGENT1_STATE_DIR     : directory where JSON state files are written
                         (default: /tmp/swebench-agent/states)
"""

from dotenv import load_dotenv
load_dotenv()  # must be called before any os.environ.get() reads

import json
import os
import time
from pathlib import Path

from tool_executor import run_agent_loop, make_bash_tool, AgentResult

# ---------------------------------------------------------------------------
# Configuration from .env
# ---------------------------------------------------------------------------

MAX_RETRIES = int(os.environ.get("AGENT1_MAX_RETRIES", 3))
STATE_DIR   = Path(os.environ.get("AGENT1_STATE_DIR", "/tmp/swebench-agent/states"))
REPRO_TEST_PATH = "/testbed/repro_test.py"


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are Agent 1 in a multi-agent software engineering pipeline. Your sole
responsibility is to reproduce a bug described in a GitHub issue.

## Your environment
- You are operating inside a Docker container.
- The repository is checked out at /testbed at the exact commit before the bug was fixed.
- Use the bash tool for ALL interactions with the repository — reading files,
  searching code, running tests, and writing files.
- Always run commands from /testbed as the working directory.
- The environment is fully initialized — you do not need to install anything.

## Your workflow
1. Read the problem statement carefully.
2. Use the hints to orient your exploration — they often contain stack traces
   or workarounds that point directly to the relevant code.
3. Explore the repository structure to find the relevant source files and
   existing test files related to the issue.
4. Use the test targets as a guide to locate relevant code paths and test
   conventions — do NOT copy them directly as your reproduction test.
5. Write a minimal reproduction test to /testbed/repro_test.py that triggers
   the bug. The test must use pytest conventions (a function starting with test_).
6. Run the test with: pytest /testbed/repro_test.py -v
7. Verify the failure is an AssertionError or a clearly bug-related error —
   NOT an ImportError, ModuleNotFoundError, or setup issue.
8. If the test errors due to environment issues (not the bug), fix the test
   and retry. You have a limited number of retries — be precise.

## What you must produce
When you have confirmed the bug is reproducible, end your response with a
JSON block in this exact format (and nothing after it):

```json
{
  "repro_test_path": "/testbed/repro_test.py",
  "failure_trace": "<the full pytest output showing the failure>",
  "bug_explanation": "<1-3 sentences explaining what the bug is and where it lives in the code>"
}
```

## Important rules
- Never modify any source files in the repository — only write /testbed/repro_test.py.
- The test must fail on the CURRENT code and be designed to pass once the bug is fixed.
- If you cannot reproduce the bug after exhausting your retries, still output
  the JSON block with your best attempt and explain the difficulty in bug_explanation.
"""


# ---------------------------------------------------------------------------
# Initial message builder
# ---------------------------------------------------------------------------

def build_initial_message(context: dict, previous_response: str | None = None) -> str:
    message = f"""
You have been given the following SWE-bench issue to reproduce.

<instance_id>{context['instance_id']}</instance_id>

<problem_statement>
{context['problem_statement']}
</problem_statement>

<hints>
{context['hints_text']}
</hints>

<repo>{context['repo']}</repo>
<repo_path>{context['repo_path']}</repo_path>
<base_commit>{context['base_commit']}</base_commit>
<version>{context['version']}</version>

<test_targets>
These are the tests that must pass once the issue is fixed. Use them to
locate relevant code paths and understand the expected behavior — do NOT
copy them as your reproduction test.
{json.dumps(context['fail_to_pass'], indent=2)}
</test_targets>

Begin by reading the problem statement and hints, then explore the repository.
Write your reproduction test to {REPRO_TEST_PATH} once you understand the bug.
"""

    if previous_response:
        message += f"""
## Note: previous attempt failed
Your previous attempt did not produce a valid output or the test did not
confirm the bug. Review your approach and try again.
Previous final response (last 500 chars):
{previous_response[-500:]}
"""

    return message


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def parse_agent1_output(final_response: str) -> dict | None:
    """
    Extract the JSON block from Agent 1's final response.

    Returns the parsed dict on success, or None if no valid JSON block
    was found (which means the agent didn't follow the output format).
    """
    try:
        start = final_response.rfind("```json")
        end   = final_response.rfind("```", start + 1)
        if start == -1 or end == -1:
            return None
        json_str = final_response[start + 7:end].strip()
        return json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def save_state(instance_id: str, state: dict) -> Path:
    """
    Persist the agent's state to a JSON file.

    Path: {STATE_DIR}/{instance_id}/agent1.json
    Overwrites on each save — the file always reflects the latest run.
    """
    state_path = STATE_DIR / instance_id / "agent1.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2))
    print(f"[agent1] state saved → {state_path}")
    return state_path


def load_state(instance_id: str) -> dict | None:
    """
    Load a previously saved Agent 1 state if it exists.
    Returns None if no state file is found.
    """
    state_path = STATE_DIR / instance_id / "agent1.json"
    if state_path.exists():
        return json.loads(state_path.read_text())
    return None


# ---------------------------------------------------------------------------
# Message serialization helper
# ---------------------------------------------------------------------------

def serialize_messages(messages: list) -> list:
    """
    Convert Anthropic SDK content blocks to JSON-serializable dicts
    so the full message history can be saved to the state file.
    """
    serialized = []
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            serialized.append({"role": m["role"], "content": content})
        elif isinstance(content, list):
            serialized.append({
                "role": m["role"],
                "content": [
                    b if isinstance(b, dict) else b.__dict__
                    for b in content
                ],
            })
    return serialized


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_agent1(context: dict, container) -> dict:
    """
    Run Agent 1 for the given SWE-bench instance.

    Checks for an existing successful state first — if one exists, skips
    the run entirely and returns the cached result (crash recovery).

    Retries up to AGENT1_MAX_RETRIES times if the agent fails to produce
    a valid output or the test doesn't confirm the bug correctly.

    Args:
        context   : the agent1 context bundle from the orchestrator
        container : the running Docker container for this instance

    Returns:
        A state dict with status, repro_test_path, failure_trace,
        bug_explanation, messages, iterations, and timestamp.
    """
    instance_id = context["instance_id"]

    # Crash recovery — skip if a successful run already exists
    existing = load_state(instance_id)
    if existing and existing.get("status") == "success":
        print(f"[agent1] found existing successful state for {instance_id} — skipping.")
        return existing

    bash_tool   = make_bash_tool(container)
    last_result: AgentResult | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\n[agent1] attempt {attempt}/{MAX_RETRIES} for {instance_id}")

        previous_response = last_result.final_response if last_result else None
        initial_message   = build_initial_message(context, previous_response)

        last_result = run_agent_loop(
            instance_id=instance_id,
            system_prompt=SYSTEM_PROMPT,
            initial_message=initial_message,
            tools=[bash_tool],
            verbose=True,
        )

        if not last_result.success:
            print(f"[agent1] attempt {attempt} hit MAX_ITERATIONS without end_turn.")
            continue

        parsed = parse_agent1_output(last_result.final_response)
        if parsed is None:
            print(f"[agent1] attempt {attempt} did not produce a valid JSON output block.")
            continue

        # Success — persist and return
        state = {
            "instance_id":    instance_id,
            "status":         "success",
            "attempt":        attempt,
            "repro_test_path": parsed["repro_test_path"],
            "failure_trace":  parsed["failure_trace"],
            "bug_explanation": parsed["bug_explanation"],
            "messages":       serialize_messages(last_result.messages),
            "iterations":     last_result.iterations,
            "timestamp":      time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        save_state(instance_id, state)
        return state

    # All retries exhausted
    print(f"[agent1] all {MAX_RETRIES} attempts failed for {instance_id}.")
    state = {
        "instance_id":    instance_id,
        "status":         "failed",
        "attempts":       MAX_RETRIES,
        "repro_test_path": None,
        "failure_trace":  None,
        "bug_explanation": None,
        "messages":       serialize_messages(last_result.messages) if last_result else [],
        "iterations":     last_result.iterations if last_result else 0,
        "timestamp":      time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    save_state(instance_id, state)
    return state