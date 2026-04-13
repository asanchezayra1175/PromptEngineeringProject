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

from tool_executor import (
    run_agent_loop_auto, make_bash_tool, AgentResult, MAX_ITERATIONS,
    make_get_classes_and_methods_tool,
    make_extract_method_tool,
    make_find_similar_api_calls_tool,
)
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
- Use the specialist tools or bash to read source code.
- Target: 5 tool calls. Do not exceed without good reason.
 
## Specialist tools
- `get_classes_and_methods(file_path)` — list all classes/methods with line numbers.
- `extract_method(file_path, method_name)` — extract a single method by name.
- `find_similar_api_calls(method_name, search_path)` — find all call sites of a function.
 
## Maintainer constraints
If the hints contain maintainer statements, extract them first:
<think>
Maintainer constraints: [quote each verbatim]
- Does each rule out any fix strategies?
- Does each specify a required location or behavior?
</think>
A fix that contradicts a maintainer constraint is invalid.
 
## Workflow
 
### Step 1 — Understand (no tool calls)
<think>
1. What is the function SUPPOSED to do in plain English?
2. What input triggers the failure and what error results?
3. Which fix pattern from the library below matches this bug? Name it now.
4. Where should the fix live — inside the function that crashes, or at the call site?
   Fix inside the crashing function protects all callers. Fix at call site only if
   this caller needs different behavior from all others. Commit to one location.
5. How many lines need to change? If the answer is more than 3, reconsider —
   P1/P2/P4/P5 fixes are almost always 1-2 lines.
</think>
 
### Step 2 — Read the source (tool calls)
- Use `extract_method` to read the buggy method.
- Use `find_similar_api_calls` if you are changing a function's return type/shape.
- Read the 5 lines after the buggy line to check downstream contracts.
 
### Step 3 — Verify and propose
<think>
1. Quote the exact buggy line(s) verbatim.
2. Quote the 5 lines immediately after. Does your fix change what they receive?
   If yes, list every downstream line that also needs to change.
3. Re-read your pattern choice from Step 1. Does your fix follow it exactly?
   If P1: is the guard at the TOP of the method, not inside a branch?
   If P2: is it a single replacement call, not an if/hasattr branch?
   If it deviates, state why.
4. Count the lines your fix changes. If more than 5, simplify.
</think>
 
End with ONLY this JSON — nothing after it:
 
```json
{
  "files_to_modify": ["/testbed/path/to/file.py"],
  "proposed_changes": [
    {
      "file": "/testbed/path/to/file.py",
      "location": "ClassName.method_name, line N",
      "change": "INSERT after line N:\n  <exact new line(s) to insert>",
      "rationale": "<root cause, why this pattern, why this location>"
    }
  ],
  "fix_strategy": "<pattern name>: <1-2 sentences — what changes, why minimal, why safe>"
}
```
 
## change field format rules
The `change` field must use one of these formats — nothing else:
 
**For insertions (preferred):**
`INSERT after line N:\n  <exact indented lines to insert>`
 
**For single-line replacements:**
`REPLACE line N:\n  <old line>\nWITH:\n  <new line>`
 
**For multi-line replacements (only when insertion is impossible):**
`REPLACE lines N-M:\n  <old block>\nWITH:\n  <new block>`
 
Never use a REPLACE block larger than the minimal change. If you can insert
2 lines instead of replacing 30, insert.
 
## Fix pattern library
These patterns cover the failure modes where agents most commonly go wrong.
They are NOT ordered by importance — scan all of them before choosing.
If NO pattern fits, that is fine: write "no pattern match: <description>" in
fix_strategy. Do not force a pattern that does not apply.
 
---
 
**PA — None/empty guard placed at wrong level**
When to use: `data[key]` or `obj.attr` crashes when data/obj is None/empty.
Agent failure mode: guard is added at the call site instead of inside the function.
Fix: `if data is None: return` at the TOP of the crashing function — before any
  loop or branch — not in the caller. Protects all callers automatically.
Example:
  def process(self, items):
+     if items is None:
+         return
      for item in items:   # existing first line, unchanged
 
**PB — Deprecated or removed API replaced with version-detection branch**
When to use: `obj.old_method()` raises AttributeError in library >= X.0.
Agent failure mode: adds `if hasattr(obj, 'old_method'):` branch instead of single replacement.
Fix: Replace with a single idiomatic call that works across versions.
  Never use `if hasattr`, `isinstance`, or version-number checks as the fix.
Example:
  # Before (deprecated in numpy 2.0):
  - values = series.get_values()
  # After (works on all versions):
  + values = np.asarray(series)
 
**PC — Generic exception type used instead of domain-specific**
When to use: A domain error surfaces as TypeError, RuntimeError, ValueError, or Exception.
Agent failure mode: catches or raises `Exception` / `ValueError` without checking
  whether the library defines a more specific error class.
Fix: grep for domain exception classes first (`grep -r "class.*Error" /testbed/src`),
  then raise/catch the most specific one found.
Example:
  # Wrong — too broad:
  - raise ValueError("parse failed")
  # Right — domain-specific:
  + raise SQLParseError("parse failed")
 
**PD — Condition boundary wrong (too broad or too specific)**
When to use: An `if` condition misses valid cases or includes invalid ones.
Agent failure mode: changes the body of the branch but not the condition itself.
Fix: Adjust the comparator (`>` → `>=`, `==` → `in`, `is None` → `not`).
  One token in the condition changes, body stays the same.
Example:
  # Bug: skips zero-length match
  - if len(match) > 0:
  + if len(match) >= 0:
 
**PE — Missing type conversion wrapping an expression**
When to use: A value is passed or returned with the wrong type.
Agent failure mode: adds a new variable or branch instead of wrapping inline.
Fix: Wrap the expression in the appropriate conversion at the point of use.
Example:
  # Bug: returns generator where list is expected
  - return (item.value for item in self.nodes)
  + return [item.value for item in self.nodes]
 
**PF — Wrong function or method called (similar name)**
When to use: The right logic but a similarly-named function is called.
Agent failure mode: reads only one function and assumes it is correct without
  checking siblings. Use find_similar_api_calls to compare.
Fix: Replace with the correct function. Read both to confirm the difference.
Example:
  # Bug: extend() appends the list itself, append() adds items
  - self.results.append(new_items)
  + self.results.extend(new_items)
 
**PG — Shared mutable state modified in-place when copy needed**
When to use: A method modifies a shared object, corrupting state for other callers.
Agent failure mode: fixes the symptom in one caller instead of making a copy at source.
Fix: Add `.copy()`, `list()`, `dict()`, or equivalent at the point of assignment.
Example:
  # Bug: all callers share the same default dict
  - self.options = default_options
  + self.options = default_options.copy()
 
**PH — Wrong argument passed to a function call**
When to use: A function is called with the wrong argument value, order, or keyword.
Agent failure mode: changes the function signature instead of the call site.
Fix: Fix the call site — add missing arg, remove extra arg, fix kwarg name/value.
  Read the function signature with extract_method before proposing.
Example:
  # Bug: passes index instead of value
  - render(template, idx, context)
  + render(template, item, context)
 
---
 
**No pattern match**
If none of PA–PH apply, write in fix_strategy:
  "no pattern match: <one sentence describing what changes and why>"
The Critic will evaluate it on its own merits. This is always better than forcing
a pattern that does not fit.
 
## Complexity escalation
Cycle 1: use the matching pattern exactly, ≤5 lines changed total.
Retry cycles: may use a more complex approach, but fix_strategy must explain
  why the simpler cycle-1 approach was insufficient.
 
## Hard rules
- Never propose version-detection branches (if hasattr, isinstance checks).
- Never propose a partial fix — include all downstream lines that break.
- Never modify any files — proposal only.
- Never add error handling at a call site to paper over a bug inside a function.
- The JSON block must be the VERY LAST thing in your response.
- A fix contradicting a maintainer constraint is invalid.
- If exploration_notes quotes a PR diff, follow its pattern or explain why not.
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

    bash_tool                  = make_bash_tool(container)
    get_classes_tool           = make_get_classes_and_methods_tool(container)
    extract_method_tool        = make_extract_method_tool(container)
    find_similar_api_calls_tool = make_find_similar_api_calls_tool(container)
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
            tools=[bash_tool, get_classes_tool, extract_method_tool, find_similar_api_calls_tool],
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