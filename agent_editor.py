"""
agent_editor.py

Stage 2 — Editor Agent

Responsibility: receive the Architect's solution proposal and implement it
by modifying source files. Produces a unified diff.

The Editor does not reason about the fix — it implements what the Architect
specified. If the Architect's proposal is unclear, the Editor asks no
questions — it implements its best interpretation and the Critic will catch
any issues.

Environment variables (set in .env):
  EDITOR_MAX_RETRIES   : max attempts (default: 2)
  EDITOR_MAX_ITERATIONS: max tool calls per attempt (default: 10)
  EDITOR_STATE_DIR     : where to cache results (default: ./states)
  EDITOR_MODEL         : model override (default: MODEL from tool_executor)
"""

from dotenv import load_dotenv
load_dotenv()

import json
import os
import time
from pathlib import Path

from tool_executor import run_agent_loop_auto, make_bash_tool, AgentResult, MAX_ITERATIONS
from logger import get_logger

MAX_RETRIES = int(os.environ.get("EDITOR_MAX_RETRIES", 2))
MAX_ITER    = int(os.environ.get("EDITOR_MAX_ITERATIONS", MAX_ITERATIONS))
STATE_DIR   = Path(os.environ.get("EDITOR_STATE_DIR", "./states"))
MODEL       = os.environ.get("EDITOR_MODEL", os.environ.get("MODEL", "claude-sonnet-4-20250514"))


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are the Editor Agent in a software engineering pipeline. Your sole
responsibility is to implement a solution proposal by modifying source files
and producing a clean unified diff.
 
## CRITICAL: How to end your response
After running git diff and confirming the patch is correct, you MUST end your
response with this JSON block. This is the only acceptable way to finish.
Without this block the pipeline cannot continue.
 
```json
{
  "patch": "<full verbatim output of git diff>",
  "implementation_notes": "<1-2 sentences describing what you changed>"
}
```
 
## Environment
- You are inside a Docker container. The repository is at /testbed.
- The Architect Agent has already analyzed the bug and specified what to change.
- Use the bash tool to read files and apply changes.
 
## Workflow
 
### Step 1: Read each file to modify (1 tool call per file)
- Read the exact file(s) listed in files_to_modify IN FULL.
- Locate the specific class/method/lines described in proposed_changes.
 
### Step 2: Apply the changes (1 tool call per file)
- Apply the proposed changes using Python or sed — whichever is cleaner.
- Keep changes minimal — only modify what the Architect specified.
- Do not reformat, rename, or refactor unrelated code.
 
### Step 3: Verify with git diff (1 tool call)
Run EXACTLY this command:
  cd /testbed && git diff
 
Check two things only:
- Is the output non-empty? If empty, the edit did not apply — fix and retry.
- Does it show only the lines you intended to change? If not, reset and reapply.
 
If both checks pass: your next action is to write the JSON output in Step 4.
Do NOT make any more tool calls. Do NOT run Python. Do NOT verify anything else.
The very next thing you write after this tool call must be the JSON block.
 
### Step 4: Output your result
Copy the full git diff output verbatim into the patch field.
End your response with ONLY this JSON block — nothing after it:
 
```json
{
  "patch": "<full verbatim output of git diff — must start with 'diff --git'>",
  "implementation_notes": "<1-2 sentences describing what you changed>"
}
```
 
## Hard rules
- Only modify files listed in files_to_modify.
- Never modify test files.
- Step 3 MUST be: cd /testbed && git diff. Not git status. Not git log. git diff.
- If git diff produces non-empty correct output: output the JSON NOW. No more tool calls.
- NEVER run Python, pytest, or any code execution — that is the Critic's job, not yours.
- The JSON block must be the VERY LAST thing in your response.
"""


# ---------------------------------------------------------------------------
# Initial message builder
# ---------------------------------------------------------------------------

def build_initial_message(
    context: dict,
    localization_state: dict,
    architect_state: dict,
) -> str:
    # Format proposed changes as readable text
    changes_text = ""
    for i, change in enumerate(architect_state.get("proposed_changes", []), 1):
        changes_text += (
            f"\nChange {i}:\n"
            f"  File    : {change.get('file', '')}\n"
            f"  Location: {change.get('location', '')}\n"
            f"  Change  : {change.get('change', '')}\n"
            f"  Rationale: {change.get('rationale', '')}\n"
        )

    message = (
        "Implement the following solution proposal by modifying the source files.\n\n"
        f"<instance_id>{context['instance_id']}</instance_id>\n\n"
        f"<problem_statement>\n{context['problem_statement']}\n</problem_statement>\n\n"
        "<localization_report>\n"
        f"Bug analysis: {localization_state['bug_analysis']}\n"
        f"Exploration notes: {localization_state['exploration_notes']}\n"
        "</localization_report>\n\n"
        "<solution_proposal>\n"
        f"Fix strategy: {architect_state['fix_strategy']}\n\n"
        f"Files to modify: {json.dumps(architect_state['files_to_modify'], indent=2)}\n"
        f"Proposed changes:{changes_text}"
        "</solution_proposal>\n\n"
        "Read each file, implement the proposed changes, then run git diff "
        "and output the JSON result.\n"
    )

    return message


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def parse_output(final_response: str) -> dict | None:
    try:
        start = final_response.rfind("```json")
        end   = final_response.rfind("```", start + 1)
        if start == -1 or end == -1:
            return None
        json_str = final_response[start + 7:end].strip()
        parsed = json.loads(json_str)
        if not parsed.get("patch") or not parsed.get("implementation_notes"):
            return None
        return parsed
    except (json.JSONDecodeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def save_state(instance_id: str, state: dict, cycle: int) -> Path:
    state_path = STATE_DIR / instance_id / f"editor_cycle{cycle}.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, cls=_SafeEncoder))
    get_logger().info(f"[editor] state saved → {state_path}")
    return state_path


def load_state(instance_id: str, cycle: int) -> dict | None:
    state_path = STATE_DIR / instance_id / f"editor_cycle{cycle}.json"
    if state_path.exists():
        try:
            text = state_path.read_text().strip()
            if not text:
                return None
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None
    return None


class _SafeEncoder(json.JSONEncoder):
    """JSON encoder that converts any non-serializable object to its string repr."""
    def default(self, obj):
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return str(obj)


def serialize_messages(messages: list) -> list:
    """
    Convert Anthropic SDK message history to a JSON-safe list.
    Uses _SafeEncoder to handle any SDK-specific types (TextBlock,
    ToolUseBlock, DirectCaller, etc.) that are not natively serializable.
    """
    raw = json.dumps(
        [{"role": m["role"], "content": m["content"]} for m in messages],
        cls=_SafeEncoder,
    )
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_editor(
    context: dict,
    container,
    localization_state: dict,
    architect_state: dict,
    cycle: int = 1,
) -> dict:
    """
    Run the Editor Agent for a SWE-bench instance.

    Args:
        context            : agent context bundle
        container          : running Docker container
        localization_state : output from run_localization()
        architect_state    : output from run_architect()
        cycle              : fix/verify cycle number (1-based)

    Returns:
        State dict with status, patch, implementation_notes
    """
    instance_id = context["instance_id"]

    existing = load_state(instance_id, cycle)
    if existing and existing.get("status") == "success":
        get_logger().info(f"[editor] found existing state for {instance_id} cycle {cycle} — skipping.")
        return existing

    bash_tool   = make_bash_tool(container)
    last_result: AgentResult | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        get_logger().info(f"[editor] attempt {attempt}/{MAX_RETRIES} for {instance_id} cycle {cycle}")

        initial = build_initial_message(context, localization_state, architect_state)

        last_result = run_agent_loop_auto(
            instance_id=instance_id,
            system_prompt=SYSTEM_PROMPT,
            initial_message=initial,
            tools=[bash_tool],
            model=MODEL,
            max_iterations=MAX_ITER,
            verbose=True,
        )

        if not last_result.success:
            get_logger().info(f"[editor] attempt {attempt} hit MAX_ITERATIONS.")
            continue

        parsed = parse_output(last_result.final_response)
        if parsed is None:
            get_logger().info(f"[editor] attempt {attempt} did not produce valid JSON.")
            get_logger().info("[editor] final response was: " + last_result.final_response)
            continue

        state = {
            "instance_id":          instance_id,
            "status":               "success",
            "cycle":                cycle,
            "attempt":              attempt,
            "patch":                parsed["patch"],
            "implementation_notes": parsed["implementation_notes"],
            "messages":             serialize_messages(last_result.messages),
            "iterations":           last_result.iterations,
            "timestamp":            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        save_state(instance_id, state, cycle)
        return state

    get_logger().info(f"[editor] all {MAX_RETRIES} attempts failed for {instance_id} cycle {cycle}.")
    state = {
        "instance_id":          instance_id,
        "status":               "failed",
        "cycle":                cycle,
        "attempts":             MAX_RETRIES,
        "patch":                None,
        "implementation_notes": None,
        "messages":             serialize_messages(last_result.messages) if last_result else [],
        "iterations":           last_result.iterations if last_result else 0,
        "timestamp":            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    save_state(instance_id, state, cycle)
    return state