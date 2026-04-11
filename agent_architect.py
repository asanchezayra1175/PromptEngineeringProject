"""
agent_architect.py

Stage 2 — Architect Agent

Responsibility: receive the localization report and produce a concrete
solution proposal — what to change, where, and why — without touching
any files. The Editor implements this proposal.

Output: a solution proposal containing:
  - files_to_modify : list of files that need changes
  - proposed_changes: description of each change with file, location, and rationale
  - fix_strategy    : overall approach in 2-3 sentences

On retry cycles, receives the previous proposal, the Editor's patch,
and the Critic's test output so it can refine its approach.

Environment variables (set in .env):
  ARCHITECT_MAX_RETRIES   : max attempts (default: 2)
  ARCHITECT_MAX_ITERATIONS: max tool calls per attempt (default: 10)
  ARCHITECT_STATE_DIR     : where to cache results (default: ./states)
  ARCHITECT_MODEL         : model override (default: MODEL from tool_executor)
"""

from dotenv import load_dotenv
load_dotenv()

import json
import os
import time
from pathlib import Path

from tool_executor import run_agent_loop_auto, make_bash_tool, AgentResult, MAX_ITERATIONS
from logger import get_logger

MAX_RETRIES = int(os.environ.get("ARCHITECT_MAX_RETRIES", 2))
MAX_ITER    = int(os.environ.get("ARCHITECT_MAX_ITERATIONS", MAX_ITERATIONS))
STATE_DIR   = Path(os.environ.get("ARCHITECT_STATE_DIR", "./states"))
MODEL       = os.environ.get("ARCHITECT_MODEL", os.environ.get("MODEL", "claude-sonnet-4-20250514"))


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are the Architect Agent in a software engineering pipeline. Your sole
responsibility is to produce a concrete solution proposal for a bug — what
to change, where, and why. You do NOT implement the fix yourself.
 
## Environment
- You are inside a Docker container. The repository is at /testbed.
- A Localization Agent has already identified the relevant files and symbols.
- Use the bash tool to read source code. Allowed commands: grep, cat, sed only.
- Target: 5 tool calls. Do not exceed without good reason.
 
## Workflow
 
### Step 1 — Understand the problem (no tool calls)
Before touching any code, reason about the problem from the issue description
and localization report alone:
 
<think>
1. What is the code SUPPOSED to do? Describe the intended behavior in plain English,
   ignoring the bug entirely. What value should it compute? What is it used for?
 
2. What goes wrong and under what condition? Be specific — what input or environment
   triggers the failure, and what error or wrong result does it produce?

3. Read the problem statement and hints given to you. Make sure you understand them and can connect them to the plan you create.
 
4. Now compare your first-principles answer to the localization report.
   Does the localization point to the same fix you would have chosen?
   If not — why not? Which is more likely to be correct?
</think>
 
This step exists because the localization report tells you WHERE the bug is,
not what the correct solution is. The correct solution comes from understanding
the problem. Only after you understand what the code should do can you evaluate
whether a proposed fix is correct.
 
### Step 2 — Read the source
Read every file in relevant_files IN FULL. Do not skim. Do not stop at the
buggy line — read the entire function or method it lives in.
 
### Step 3 — Quote the affected block, then reason
Before writing a single word of proposal, paste the exact lines of the buggy
symbol into a think block and answer these questions with specifics from the code:
 
<think>
1. Root cause check:
   Quote the buggy line(s) verbatim. Does this match the bug_analysis exactly?
 
2. Downstream contract check — THIS IS MANDATORY:
   Quote the 5 lines AFTER the buggy line verbatim.
   Does the variable you are changing appear in any of them?
   If yes, answer for each downstream use:
     a. Length  — does your fix produce the same number of elements?
     b. Nulls   — does your fix introduce NaN/None where the original had values?
     c. Index   — if pandas, is the index what downstream code expects?
     d. Semantics — does it compute the same logical result via a different path?
   If ANY property differs, you MUST also fix those downstream lines. List them now.
 
3. Simplicity check:
   Is your proposed fix the minimal idiomatic solution?
 
4. Completeness check:
   List every line that needs to change in a ordered list. Not just the first. All of them.
</think>
 
Do not proceed to Step 3 until you have answered all four questions with
direct quotes from the source code.
 
### Step 4 — Output your proposal
End your response with ONLY this JSON block — nothing after it:
 
```json
{
  "files_to_modify": [
    "/testbed/src/path/to/file.py"
  ],
  "proposed_changes": [
    {
      "file": "/testbed/src/path/to/file.py",
      "location": "ClassName.method_name, around line N",
      "change": "1.<every line that needs to change, including downstream lines — not just the first>, 2.<the next line if it also needs to change>, ...",
      "rationale": "<why this fixes the root cause and why downstream lines are also addressed>"
    }
  ],
  "fix_strategy": "<2-3 sentences: what changes, why it is minimal, and why it is safe>"
}
```
 
## Hard rules
- Read relevant_files IN FULL before proposing.
- The downstream contract check in Step 2 is not optional — do it every time.
- Never propose version-detection branches if a single idiomatic call exists.
- Never propose a partial fix — if downstream lines break, fix them too.
- Never modify any files — proposal only.
- The JSON block must be the VERY LAST thing in your response.
```
 
## Hard rules
- Read relevant_files IN FULL before proposing.
- The downstream contract check in Step 2 is not optional — do it every time.
- Never propose version-detection branches if a single idiomatic call exists.
- Never propose a partial fix — if downstream lines break, fix them too.
- Never modify any files — proposal only.
- The JSON block must be the VERY LAST thing in your response.
"""


# ---------------------------------------------------------------------------
# Initial message builder
# ---------------------------------------------------------------------------

def build_initial_message(
    context: dict,
    localization_state: dict,
    previous_proposal: str | None = None,
    previous_patch: str | None = None,
    critic_output: str | None = None,
) -> str:
    message = (
        "Produce a solution proposal for the following bug.\n\n"
        f"<instance_id>{context['instance_id']}</instance_id>\n\n"
        f"<problem_statement>\n{context['problem_statement']}\n</problem_statement>\n\n"
        f"<hints>\n{context['hints_text']}\n</hints>\n\n"
        "<localization_report>\n"
        f"Relevant files: {json.dumps(localization_state['relevant_files'], indent=2)}\n"
        f"Relevant symbols: {json.dumps(localization_state['relevant_symbols'], indent=2)}\n"
        f"Bug analysis: {localization_state['bug_analysis']}\n"
        f"Exploration notes: {localization_state['exploration_notes']}\n"
        "</localization_report>\n\n"
        "Start by reading the relevant files, then produce your solution proposal.\n"
    )

    # On retry cycles, include previous attempt context
    if previous_proposal and previous_patch and critic_output:
        message += (
            "\n## Previous fix attempt failed — use this to refine your proposal\n\n"
            f"<previous_proposal>\n{previous_proposal}\n</previous_proposal>\n\n"
            f"<previous_patch>\n{previous_patch}\n</previous_patch>\n\n"
            f"<critic_output>\n{critic_output}\n</critic_output>\n\n"
            "Analyze what went wrong with the previous fix and propose a better approach.\n"
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
        required = {"files_to_modify", "proposed_changes", "fix_strategy"}
        if not required.issubset(parsed.keys()):
            return None
        return parsed
    except (json.JSONDecodeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def save_state(instance_id: str, state: dict, cycle: int) -> Path:
    state_path = STATE_DIR / instance_id / f"architect_cycle{cycle}.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, cls=_SafeEncoder))
    get_logger().info(f"[architect] state saved → {state_path}")
    return state_path


def load_state(instance_id: str, cycle: int) -> dict | None:
    state_path = STATE_DIR / instance_id / f"architect_cycle{cycle}.json"
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

def run_architect(
    context: dict,
    container,
    localization_state: dict,
    cycle: int = 1,
    previous_proposal: str | None = None,
    previous_patch: str | None = None,
    critic_output: str | None = None,
) -> dict:
    """
    Run the Architect Agent for a SWE-bench instance.

    Args:
        context            : agent context bundle
        container          : running Docker container
        localization_state : output from run_localization()
        cycle              : fix/verify cycle number (1-based)
        previous_proposal  : Architect's previous proposal text (retry only)
        previous_patch     : Editor's previous patch (retry only)
        critic_output      : Critic's test output (retry only)

    Returns:
        State dict with status, files_to_modify, proposed_changes, fix_strategy
    """
    instance_id = context["instance_id"]

    existing = load_state(instance_id, cycle)
    if existing and existing.get("status") == "success":
        get_logger().info(f"[architect] found existing state for {instance_id} cycle {cycle} — skipping.")
        return existing

    bash_tool   = make_bash_tool(container)
    last_result: AgentResult | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        get_logger().info(f"[architect] attempt {attempt}/{MAX_RETRIES} for {instance_id} cycle {cycle}")

        initial = build_initial_message(
            context, localization_state,
            previous_proposal, previous_patch, critic_output
        )

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
            get_logger().info(f"[architect] attempt {attempt} hit MAX_ITERATIONS.")
            continue

        parsed = parse_output(last_result.final_response)
        if parsed is None:
            get_logger().info(f"[architect] attempt {attempt} did not produce valid JSON.")
            continue

        state = {
            "instance_id":      instance_id,
            "status":           "success",
            "cycle":            cycle,
            "attempt":          attempt,
            "files_to_modify":  parsed["files_to_modify"],
            "proposed_changes": parsed["proposed_changes"],
            "fix_strategy":     parsed["fix_strategy"],
            "full_response":    last_result.final_response,
            "messages":         serialize_messages(last_result.messages),
            "iterations":       last_result.iterations,
            "timestamp":        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        save_state(instance_id, state, cycle)
        return state

    get_logger().info(f"[architect] all {MAX_RETRIES} attempts failed for {instance_id} cycle {cycle}.")
    state = {
        "instance_id":      instance_id,
        "status":           "failed",
        "cycle":            cycle,
        "attempts":         MAX_RETRIES,
        "files_to_modify":  [],
        "proposed_changes": [],
        "fix_strategy":     "",
        "full_response":    last_result.final_response if last_result else "",
        "messages":         serialize_messages(last_result.messages) if last_result else [],
        "iterations":       last_result.iterations if last_result else 0,
        "timestamp":        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    save_state(instance_id, state, cycle)
    return state