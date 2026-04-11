"""
agent_critic.py

Stage 2 — Critic Agent

Responsibility: apply the Editor's patch, run the test suite, and evaluate
whether the fix is correct. If not, triggers a retry loop back to the
Architect with a detailed report of what went wrong.

Environment variables (set in .env):
  CRITIC_MAX_RETRIES   : max attempts (default: 2)
  CRITIC_MAX_ITERATIONS: max tool calls per attempt (default: 10)
  CRITIC_STATE_DIR     : where to cache results (default: ./states)
  CRITIC_MODEL         : model override (default: MODEL from tool_executor)
"""

from dotenv import load_dotenv
load_dotenv()

import json
import os
import time
from pathlib import Path

from tool_executor import run_agent_loop_auto, make_bash_tool, AgentResult, MAX_ITERATIONS
from logger import get_logger

MAX_RETRIES = int(os.environ.get("CRITIC_MAX_RETRIES", 2))
MAX_ITER    = int(os.environ.get("CRITIC_MAX_ITERATIONS", MAX_ITERATIONS))
STATE_DIR   = Path(os.environ.get("CRITIC_STATE_DIR", "./states"))
MODEL       = os.environ.get("CRITIC_MODEL", os.environ.get("MODEL", "claude-haiku-4-5-20251001"))


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are the Critic Agent in a software engineering pipeline. Your job is to
evaluate whether the implemented patch correctly and completely fixes the bug.
If it does not, you trigger a retry and tell the Architect exactly what to fix.

You do NOT run tests. Test execution is handled by the official SWE-bench
evaluation harness after the pipeline completes. Your job is code review.

## Environment
- You are inside a Docker container. The repository is at /testbed.
- The patch has already been applied to the source files.
- Allowed commands: grep, cat, sed only. No python, pytest, or code execution.

## Workflow

### Step 1: Read the patched code (1-2 tool calls)
Read each file modified by the patch in full. Focus on:
- The specific lines changed by the patch
- The 10-15 lines surrounding the change for context
- Whether the change addresses the root cause in the problem statement

### Step 2: Evaluate against four criteria

**Correctness** — does the patch fix the root cause?
- Does it address the bug described in the problem statement and bug_analysis?
- Or does it only fix a symptom while the underlying cause remains?

**Completeness** — is the fix whole?
- Are there downstream lines that depend on the changed code that also need updating?
- Are there edge cases the patch misses?

**Alignment** — does it match the Architect's proposal?
- Does the Editor's implementation match what the Architect specified?
- If it deviates, does the deviation make sense or introduce new problems?

**Safety** — does it avoid regressions?
- Does the patch modify only what is necessary?
- Could the change break other behavior in the surrounding code?

### Step 3: Output your verdict
End your response with ONLY this JSON block — nothing after it:

```json
{
  "verdict": "pass" | "fail",
  "correctness": "pass" | "fail",
  "completeness": "pass" | "fail",
  "alignment": "pass" | "fail",
  "safety": "pass" | "fail",
  "issues": ["<specific issue 1>", "<specific issue 2>"],
  "full_test_output": "n/a",
  "regressions": [],
  "regression_report": "<precise instructions for the Architect on what to fix, or 'none' if verdict is pass>"
}
```

## Verdict rules
- verdict = "pass" only if ALL four criteria pass.
- verdict = "fail" if ANY criterion fails.

## Hard rules
- NEVER run Python, pytest, or any code execution.
- NEVER modify any files.
- Be specific in regression_report — the Architect needs precise instructions.
- The JSON block must be the VERY LAST thing in your response.
"""


# ---------------------------------------------------------------------------
# Initial message builder
# ---------------------------------------------------------------------------

def build_initial_message(context: dict) -> str:
    loc = context.get("localization_state", {})
    arch = context.get("architect_state", {})

    localization_block = (
        f"<localization_report>\n"
        f"Relevant files: {json.dumps(loc.get('relevant_files', []))}\n"
        f"Bug analysis: {loc.get('bug_analysis', '')}\n"
        f"Exploration notes: {loc.get('exploration_notes', '')}\n"
        f"</localization_report>\n\n"
        if loc else ""
    )

    architect_block = (
        f"<solution_proposal>\n"
        f"Fix strategy: {arch.get('fix_strategy', '')}\n"
        f"Proposed changes: {json.dumps(arch.get('proposed_changes', []), indent=2)}\n"
        f"</solution_proposal>\n\n"
        if arch else ""
    )

    return (
        "Review the following patch and produce a verdict.\n\n"
        f"<instance_id>{context['instance_id']}</instance_id>\n\n"
        f"<problem_statement>\n{context['problem_statement']}\n</problem_statement>\n\n"
        f"{localization_block}"
        f"{architect_block}"
        f"<patch>\n{context.get('patch', '')}\n</patch>\n\n"
        "The patch has been applied to /testbed. Read the modified files, "
        "evaluate correctness, completeness, alignment, and safety, "
        "then output your verdict.\n"
    )


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
        required = {"verdict", "correctness", "completeness", "alignment",
                    "safety", "regression_report"}
        if not required.issubset(parsed.keys()):
            return None
        return parsed
    except (json.JSONDecodeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def save_state(instance_id: str, state: dict, cycle: int) -> Path:
    state_path = STATE_DIR / instance_id / f"critic_cycle{cycle}.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, cls=_SafeEncoder))
    get_logger().info(f"[critic] state saved → {state_path}")
    return state_path


def load_state(instance_id: str, cycle: int) -> dict | None:
    state_path = STATE_DIR / instance_id / f"critic_cycle{cycle}.json"
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
# Context builder (called by orchestrator)
# ---------------------------------------------------------------------------

def build_critic_context(
    agent1_context: dict,
    instance: dict,
    editor_state: dict,
    localization_state: dict | None = None,
    architect_state: dict | None = None,
) -> dict:
    """Build the context bundle for the Critic."""
    return {
        **agent1_context,
        "patch":              editor_state["patch"],
        "fail_to_pass":       json.loads(instance["FAIL_TO_PASS"]),
        "pass_to_pass":       json.loads(instance["PASS_TO_PASS"]),
        "localization_state": localization_state or {},
        "architect_state":    architect_state or {},
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_critic(
    context: dict,
    container,
    cycle: int = 1,
) -> dict:
    """
    Run the Critic Agent for a SWE-bench instance.

    Note: the patch must already be applied to the container before
    calling this — the orchestrator handles patch application.

    A previous "fail" verdict is NOT cached — the orchestrator may call
    the Critic again after a new patch from the Editor.

    Args:
        context   : critic context bundle from build_critic_context()
        container : running Docker container with patch applied
        cycle     : fix/verify cycle number (1-based)

    Returns:
        State dict with verdict, test results, regressions, regression_report
    """
    instance_id = context["instance_id"]

    # Only skip if previous run was a clean pass
    existing = load_state(instance_id, cycle)
    if existing and existing.get("status") == "success" and existing.get("verdict") == "pass":
        get_logger().info(f"[critic] found existing passing state for {instance_id} cycle {cycle} — skipping.")
        return existing

    bash_tool   = make_bash_tool(container)
    last_result: AgentResult | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        get_logger().info(f"[critic] attempt {attempt}/{MAX_RETRIES} for {instance_id} cycle {cycle}")

        initial = build_initial_message(context)

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
            get_logger().info(f"[critic] attempt {attempt} hit MAX_ITERATIONS.")
            continue

        parsed = parse_output(last_result.final_response)
        if parsed is None:
            get_logger().info(f"[critic] attempt {attempt} did not produce valid JSON.")
            continue

        state = {
            "instance_id":       instance_id,
            "status":            "success",
            "cycle":             cycle,
            "attempt":           attempt,
            "verdict":           parsed["verdict"],
            "correctness":       parsed.get("correctness", ""),
            "completeness":      parsed.get("completeness", ""),
            "alignment":         parsed.get("alignment", ""),
            "safety":            parsed.get("safety", ""),
            "issues":            parsed.get("issues", []),
            "regressions":       parsed.get("regressions", []),
            "regression_report": parsed.get("regression_report", "none"),
            "messages":          serialize_messages(last_result.messages),
            "iterations":        last_result.iterations,
            "timestamp":         time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        save_state(instance_id, state, cycle)
        return state

    get_logger().info(f"[critic] all {MAX_RETRIES} attempts failed for {instance_id} cycle {cycle}.")
    state = {
        "instance_id":       instance_id,
        "status":            "failed",
        "cycle":             cycle,
        "attempts":          MAX_RETRIES,
        "verdict":           "fail",
        "correctness":       "",
        "completeness":      "",
        "alignment":         "",
        "safety":            "",
        "issues":            [],
        "regressions":       [],
        "regression_report": "Critic failed to produce a valid output.",
        "messages":          serialize_messages(last_result.messages) if last_result else [],
        "iterations":        last_result.iterations if last_result else 0,
        "timestamp":            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    save_state(instance_id, state, cycle)
    return state