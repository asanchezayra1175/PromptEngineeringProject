"""
format_predictions.py

Collects pipeline state files across all completed instances and formats
them into the JSONL predictions file required by the SWE-bench evaluator.

Output format (one JSON object per line):
  {
    "instance_id":        "repo_owner__repo_name-issue_number",
    "model_name_or_path": "your-model-name",
    "model_patch":        "unified diff string, or empty string for failures"
  }

Usage:
  python format_predictions.py

  # Then evaluate with:
  python -m swebench.harness.run_evaluation \
      --dataset_name princeton-nlp/SWE-bench_Lite \
      --split dev \
      --predictions_path predictions.jsonl \
      --max_workers 4 \
      --run_id agentloop

Environment variables (set in .env):
  EDITOR_STATE_DIR  : where state files are stored (default: ./states)
  PREDICTIONS_PATH  : output path for the JSONL file (default: predictions.jsonl)
  MODEL_NAME        : model name written into each prediction (default: claude-sonnet-swe-agent)
"""

from dotenv import load_dotenv
load_dotenv()

import json
import os
from pathlib import Path
from datetime import datetime, timezone

STATE_DIR        = Path(os.environ.get("EDITOR_STATE_DIR", "./states"))
PREDICTIONS_PATH = Path(os.environ.get("PREDICTIONS_PATH", "predictions.jsonl"))
MODEL_NAME       = os.environ.get("MODEL_NAME", "claude-sonnet-swe-agent")


def load_editor_state(instance_dir: Path) -> dict | None:
    """Load the most recent successful editor state (highest cycle number)."""
    last = None
    for cycle in range(1, 6):
        path = instance_dir / f"editor_cycle{cycle}.json"
        if path.exists():
            try:
                text = path.read_text().strip()
                if not text:
                    continue
                state = json.loads(text)
                if state.get("status") == "success" and state.get("patch"):
                    last = state
            except (json.JSONDecodeError, ValueError):
                continue
    return last


def load_critic_state(instance_dir: Path) -> dict | None:
    """Load the most recent critic state for summary reporting."""
    last = None
    for cycle in range(1, 6):
        path = instance_dir / f"critic_cycle{cycle}.json"
        if path.exists():
            try:
                text = path.read_text().strip()
                if not text:
                    continue
                last = json.loads(text)
            except (json.JSONDecodeError, ValueError):
                continue
    return last


def build_prediction(instance_id: str, editor_state: dict | None) -> dict:
    patch = ""
    if editor_state and editor_state.get("status") == "success":
        patch = editor_state.get("patch", "")
    return {
        "instance_id":        instance_id,
        "model_name_or_path": MODEL_NAME,
        "model_patch":        patch,
    }


def format_predictions() -> Path:
    if not STATE_DIR.exists():
        raise FileNotFoundError(
            f"State directory not found: {STATE_DIR}\n"
            "Have you run the pipeline yet? Check your EDITOR_STATE_DIR env var."
        )

    instance_dirs = sorted([d for d in STATE_DIR.iterdir() if d.is_dir()])
    if not instance_dirs:
        raise ValueError(f"No instance directories found in {STATE_DIR}.")

    print(f"Found {len(instance_dirs)} instance(s) in {STATE_DIR}")

    predictions       = []
    total_with_patch  = 0
    total_empty       = 0
    critic_pass       = 0
    critic_fail       = 0
    critic_no_verdict = 0

    for instance_dir in instance_dirs:
        instance_id  = instance_dir.name
        editor_state = load_editor_state(instance_dir)
        critic_state = load_critic_state(instance_dir)

        prediction = build_prediction(instance_id, editor_state)
        predictions.append(prediction)

        patch_status = "patch" if prediction["model_patch"] else "empty"
        verdict      = critic_state.get("verdict", "no verdict") if critic_state else "no verdict"
        cycle        = editor_state.get("cycle", "?") if editor_state else "?"

        if prediction["model_patch"]:
            total_with_patch += 1
        else:
            total_empty += 1

        if critic_state:
            if verdict == "pass":
                critic_pass += 1
            else:
                critic_fail += 1
        else:
            critic_no_verdict += 1

        print(f"  {instance_id}: {patch_status} (cycle {cycle}) | critic={verdict}")

    # Write JSONL
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PREDICTIONS_PATH.open("w") as f:
        for prediction in predictions:
            f.write(json.dumps(prediction) + "\n")

    # Write summary
    summary_data = {
        "total_instances":   len(predictions),
        "with_patch":        total_with_patch,
        "empty_patch":       total_empty,
        "critic_pass":       critic_pass,
        "critic_fail":       critic_fail,
        "critic_no_verdict": critic_no_verdict,
        "predictions_path":  str(PREDICTIONS_PATH),
        "model_name":        MODEL_NAME,
        "state_dir":         str(STATE_DIR),
        "generated_at":      datetime.now(timezone.utc).isoformat(),
    }
    summary_path = PREDICTIONS_PATH.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary_data, indent=2))

    print(f"\n{'=' * 60}")
    print("PREDICTIONS SUMMARY")
    print(f"{'=' * 60}")
    print(f"Output           : {PREDICTIONS_PATH}")
    print(f"Total instances  : {len(predictions)}")
    print(f"With patch       : {total_with_patch}")
    print(f"Empty patch      : {total_empty}")
    print(f"Critic pass      : {critic_pass}")
    print(f"Critic fail      : {critic_fail}")
    print(f"Critic no verdict: {critic_no_verdict}")
    print(f"\nTo evaluate:")
    print(
        f"  python -m swebench.harness.run_evaluation \\\n"
        f"      --dataset_name princeton-nlp/SWE-bench_Lite \\\n"
        f"      --split dev \\\n"
        f"      --predictions_path {PREDICTIONS_PATH} \\\n"
        f"      --max_workers 4 \\\n"
        f"      --run_id {MODEL_NAME}"
    )

    return PREDICTIONS_PATH


if __name__ == "__main__":
    format_predictions()