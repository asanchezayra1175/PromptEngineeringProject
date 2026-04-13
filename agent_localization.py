"""
agent_localization.py

Stage 1 — Localization Agent

Responsibility: explore the repository and produce a structured report
identifying which files, classes, and functions are relevant to the bug.
Does NOT write any tests or fixes — pure exploration only.

Output: a localization report containing:
  - relevant_files   : list of file paths most likely involved in the bug
  - relevant_symbols : key classes/functions/methods to look at
  - bug_analysis     : 2-3 sentence analysis of what is likely wrong and where
  - exploration_notes: anything else the fixing agents should know

Environment variables (set in .env):
  LOCALIZATION_MAX_RETRIES   : max attempts (default: 2)
  LOCALIZATION_MAX_ITERATIONS: max tool calls per attempt (default: 10)
  LOCALIZATION_STATE_DIR     : where to cache results (default: ./states)
  LOCALIZATION_MODEL         : model override (default: MODEL from tool_executor)
"""

from dotenv import load_dotenv
load_dotenv()

import json
import os
import time
from pathlib import Path

from tool_executor import run_agent_loop_auto, make_bash_tool, AgentResult, MAX_ITERATIONS
from logger import get_logger

MAX_RETRIES     = int(os.environ.get("LOCALIZATION_MAX_RETRIES", 2))
MAX_ITER        = int(os.environ.get("LOCALIZATION_MAX_ITERATIONS", MAX_ITERATIONS))
STATE_DIR       = Path(os.environ.get("LOCALIZATION_STATE_DIR", "./states"))
MODEL           = os.environ.get("LOCALIZATION_MODEL", os.environ.get("MODEL", "claude-sonnet-4-20250514"))


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
# Localization Agent Prompt

## Role

You are the **Localization Agent** in a software engineering pipeline. Your sole responsibility is to identify exactly which files, classes, and functions contain the bug described in the issue. You do **NOT** fix anything.

---

## CRITICAL: How to end your response

You MUST end every response with this exact JSON block. No exceptions. If you have used your tool calls and know the answer, output the JSON NOW. Do not write prose summaries. Do not explain your findings in text only. The pipeline cannot continue without this JSON block.

```json
```json
{
  "relevant_files": ["/testbed/src/path/to/file.py"],
  "relevant_symbols": ["ClassName.method_name"],
  "bug_analysis": "<2-3 sentences: what the code does, what input triggers the failure, and what error or wrong result is produced. No fix suggestions. No 'should be' or 'instead' language.>",
  "exploration_notes": "<import paths, call sites, and any other symbols a fixing agent will need to read. No fix suggestions. Stop after the last fact.>"
}
```
```

---

## Environment

- You are inside a Docker container. The repository is at `/testbed`.
- Use the bash tool for all exploration.
- Allowed commands: `grep`, `cat`, `sed` only. No python, pytest, or code execution.
- Allowed web fetches: GitHub PR and issue URLs referenced in the problem_statement or hints_text only. No other URLs.
- Your **target** is 7 tool calls or fewer. You may exceed this only if you can justify why additional calls are necessary to reach a confident answer. Each call beyond 7 must be preceded by a `<think>` block explaining why you have not yet reached confidence and what specifically the next call will resolve.

---

## Your information advantage

Every issue comes with strong signals that point directly to the bug location:

| Signal | What to look for |
|---|---|
| `test_targets` | Test IDs reveal the module/file. e.g. `test/rules/std_test.py::test_L031_foo` → rules module |
| `hints_text` | Stack traces, file paths, or function names |
| `problem_statement` | Explicitly named classes, methods, or modules |

---

## How to approach this task

### Step 1 — Plan

Before making any tool calls, read the issue carefully and think through your approach using `<think>` tags. Your plan must be entirely your own — derive it from the signals in the issue:

```
<think>
[Extract every useful signal from the issue: test IDs, stack traces, named symbols, file paths, hints, problem_statement]
[Pay attention to any "EXPECTED BEHAVIORS" present on the problem_statement, these will guide your strategy]
[Form a hypothesis: what file and symbol is most likely buggy, and why?]
[Devise your own plan for how to confirm it within the 5 tool call budget]
[Decide what to look for and in what order — commit to this plan]
[Are there any PR or issue links in the problem_statement or hints_text?
 If yes, list them. These are high-priority reads — a linked PR often contains
 the code change that introduced the regression, which identifies both the
 fix site and the fix pattern. You must answer, what specific change introduced the regression?]
</think>
```

This is your contract. Do not abandon it without reason.

---

### Step 2 — Execute

Carry out your plan. After each tool call, use a `<think>` block to assess what you found:

```
<think>
[Does this confirm or change my hypothesis?]
[Do I need to revise my plan? If so, why?]
[Have I found enough to produce the JSON, or do I need another tool call?]
</think>
```

Stop as soon as you have enough information. Do not use tool calls you don't need.

---

### Step 3 — Reflect

Before writing the JSON, do a final reasoning pass:

```
<think>
[What is the confirmed buggy file and symbol?]
[What is the root cause — not just the symptom?]
[Is my bug_analysis explaining why it fails, not just what fails?]
[Are my exploration_notes useful enough for the agent that will fix this?]
[Is my bug_analysis free of fix suggestions? It must not contain the words 
 "should", "instead", "gracefully", "handle", or any other prescriptive language.
 If it does, rewrite it to describe only what the code does and what goes wrong.]
[Are my exploration_notes limited to facts a fixing agent needs to navigate the code —
 call sites, import paths, related symbols — with no suggested approach?]
[If I fetched a linked PR, did I successfully read its code changes? If the fetch returned empty or metadata only, I have not read the PR — I must retry with a fallback URL before writing the JSON]
</think>
```

If it passes your reflection criteria, then output the JSON block immediately with **no prose after it**.


---

## Hard Rules

When fetching a PR diff, always attempt these URLs in order, stopping at the first non-empty response:

https://patch-diff.githubusercontent.com/raw/{repo}/pull/{number}.patch
https://github.com/{repo}/pull/{number}.diff
https://api.github.com/repos/{repo}/pulls/{number}/files (read patch field of each file)

Never use .diff or .patch directly on github.com — these return empty in this environment.

- Your **first** tool call must be a targeted search — never `find`, `ls`, or `tree`.
- Never re-read the same file twice.
- Never modify any files.
- **NEVER** run Python scripts, pytest, or any code execution — that is not your job.
- **NEVER** verify the bug behavior — trust the problem statement and read the source.
- Output as soon as you have identified the files and symbols — do not keep exploring.
- bug_analysis and exploration_notes must be descriptive only. They must not contain words like Also banned: "needs", "must", "requires", "missing", "fails to", "does not account for", "overlooked", "fix", "check", "guard", "handle", "prevent", "ensure" or any other language that implies a fix strategy. Describe what is broken, not how to repair it.
- If the given information references a PR or issue number (e.g. "#1323", "pull/1323", "issues/1458"), use web_fetch to retrieve it before writing the JSON. Extract any code changes in that PR that touch the relevant_files. Quote the relevant lines in exploration_notes verbatim — do not summarize or interpret them.
- If a linked PR was fetched successfully, exploration_notes MUST include: (1) a verbatim quote of the specific code change in the PR that is most relevant to the buggy symbol, and (2) a statement of whether the buggy code uses the same pattern as what the PR changed. After quoting PR diff lines, do not interpret them. Do not state what the diff "shows" or "means" or "indicates about the fix." Quote the before and after lines only, labeled as before: and after:. If you cannot find a relevant code change in the PR, state that explicitly — do not omit the PR content silently.
- Every sentence in bug_analysis and exploration_notes must be verifiable by reading the source or the PR diff alone. If a sentence requires knowing what the correct behavior should be, it does not belong in the localization report.
- The JSON block must be the **VERY LAST** thing in your response.
"""


# ---------------------------------------------------------------------------
# Initial message builder
# ---------------------------------------------------------------------------

def build_initial_message(context: dict, previous_response: str | None = None) -> str:
    message = (
        "Explore the following repository and produce a localization report "
        "identifying where the bug lives.\n\n"
        f"<instance_id>{context['instance_id']}</instance_id>\n\n"
        f"<problem_statement>\n{context['problem_statement']}\n</problem_statement>\n\n"
        f"<hints>\n{context['hints_text']}\n</hints>\n\n"
        f"<repo>{context['repo']}</repo>\n"
        f"<base_commit>{context['base_commit']}</base_commit>\n\n"
        f"<test_targets>\n"
        f"These tests must pass after the fix — use them to find the relevant module.\n"
        f"{json.dumps(context['fail_to_pass'], indent=2)}\n"
        f"</test_targets>\n\n"
        "Begin by reading the problem statement, then grep for the key term. "
        "Output the JSON report when you have identified the relevant files and symbols.\n"
    )

    if previous_response:
        message += (
            "\n## Previous attempt failed\n"
            "Your previous attempt did not produce a valid JSON report. "
            "Try again — be more systematic.\n"
            f"Previous response (last 300 chars): {previous_response[-300:]}\n"
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
        required = {"relevant_files", "relevant_symbols", "bug_analysis", "exploration_notes"}
        if not required.issubset(parsed.keys()):
            return None
        return parsed
    except (json.JSONDecodeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def save_state(instance_id: str, state: dict) -> Path:
    state_path = STATE_DIR / instance_id / "localization.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, cls=_SafeEncoder))
    get_logger().info(f"[localization] state saved → {state_path}")
    return state_path


def load_state(instance_id: str) -> dict | None:
    state_path = STATE_DIR / instance_id / "localization.json"
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

def run_localization(context: dict, container) -> dict:
    """
    Run the Localization Agent for a SWE-bench instance.

    Returns a state dict with:
      status, relevant_files, relevant_symbols, bug_analysis,
      exploration_notes, messages, iterations, timestamp
    """
    instance_id = context["instance_id"]

    existing = load_state(instance_id)
    if existing and existing.get("status") == "success":
        get_logger().info(f"[localization] found existing state for {instance_id} — skipping.")
        return existing

    bash_tool   = make_bash_tool(container)
    last_result: AgentResult | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        get_logger().info(f"[localization] attempt {attempt}/{MAX_RETRIES} for {instance_id}")

        previous = last_result.final_response if last_result else None
        initial  = build_initial_message(context, previous)

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
            get_logger().info(f"[localization] attempt {attempt} hit MAX_ITERATIONS.")
            continue

        parsed = parse_output(last_result.final_response)
        if parsed is None:
            get_logger().info(f"[localization] attempt {attempt} did not produce valid JSON.")
            get_logger().info("[localization] final response was: " + last_result.final_response)
            continue

        state = {
            "instance_id":       instance_id,
            "status":            "success",
            "attempt":           attempt,
            "relevant_files":    parsed["relevant_files"],
            "relevant_symbols":  parsed["relevant_symbols"],
            "bug_analysis":      parsed["bug_analysis"],
            "exploration_notes": parsed["exploration_notes"],
            "messages":          serialize_messages(last_result.messages),
            "iterations":        last_result.iterations,
            "timestamp":         time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        save_state(instance_id, state)
        return state

    get_logger().info(f"[localization] all {MAX_RETRIES} attempts failed for {instance_id}.")
    state = {
        "instance_id":       instance_id,
        "status":            "failed",
        "attempts":          MAX_RETRIES,
        "relevant_files":    [],
        "relevant_symbols":  [],
        "bug_analysis":      "",
        "exploration_notes": "",
        "messages":          serialize_messages(last_result.messages) if last_result else [],
        "iterations":        last_result.iterations if last_result else 0,
        "timestamp":         time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    save_state(instance_id, state)
    return state