"""
run_localization_eval.py

Runs the Localization Agent across all dev set instances and produces
a summary report showing how well the prompt generalizes.

Usage:
    python run_localization_eval.py

    # Skip instances that already have a cached state:
    # (default behavior — re-run by deleting ./states/{instance_id}/localization.json)

    # Run only specific instances:
    # Set INSTANCE_IDS=id1,id2 in .env

Output:
    - Console summary table
    - ./localization_eval.json  — full results for inspection
    - ./localization_eval.md    — markdown summary table

Environment variables:
    EVAL_MAX_WORKERS : number of parallel workers (default: 1)
                       Increase carefully — each worker needs its own container.
"""

from dotenv import load_dotenv
load_dotenv()

from datasets import load_dataset
import docker
import docker.errors
import json
import logging
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from swebench.harness.test_spec.test_spec import make_test_spec
from swebench.harness.docker_build import build_env_images, build_container

from logger import make_pipeline_logger, make_agent_logger
from agent_localization import run_localization, load_state as load_localization_state

DATASET_NAME  = os.environ.get("DATASET_NAME", "princeton-nlp/SWE-bench_Lite")
SPLIT         = os.environ.get("SPLIT", "dev")
RUN_ID        = os.environ.get("RUN_ID", "agentloop")
EVAL_MAX_WORKERS = int(os.environ.get("EVAL_MAX_WORKERS", 1))
INSTANCE_IDS_FILTER = [
    i.strip() for i in os.environ.get("INSTANCE_IDS", "").split(",") if i.strip()
]


# ---------------------------------------------------------------------------
# Swebench-compatible logger
# ---------------------------------------------------------------------------

class SwebenchCompatLogger(logging.Logger):
    def __init__(self, name: str, log_path: str):
        super().__init__(name, logging.INFO)
        self.log_file = log_path
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.addHandler(handler)
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.addHandler(fh)

    def _shutdown(self):
        for h in self.handlers:
            try: h.flush(); h.close()
            except: pass


def make_swlogger(instance_id: str) -> SwebenchCompatLogger:
    log_dir = Path("./logs") / RUN_ID / instance_id
    log_dir.mkdir(parents=True, exist_ok=True)
    return SwebenchCompatLogger(f"swebench.{instance_id}", str(log_dir / "setup.log"))


# ---------------------------------------------------------------------------
# Docker setup
# ---------------------------------------------------------------------------

def setup_container(instance: dict, client: docker.DockerClient) -> object:
    swlogger = make_swlogger(instance["instance_id"])

    build_env_images(
        client=client, dataset=[instance], force_rebuild=False,
        max_workers=1, namespace=None,
        instance_image_tag="latest", env_image_tag="latest",
    )

    test_spec = make_test_spec(
        instance, namespace=None, base_image_tag="latest",
        env_image_tag="latest", instance_image_tag="latest",
    )

    container_name = f"sweb.eval.{instance['instance_id']}.{RUN_ID}"
    try:
        existing = client.containers.get(container_name)
        existing.remove(force=True)
    except docker.errors.NotFound:
        pass

    container = build_container(
        test_spec=test_spec, client=client, run_id=RUN_ID,
        logger=swlogger, nocache=False, force_rebuild=False,
    )
    container.start()
    return container


def teardown_container(container):
    try:
        container.stop()
        container.remove()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def build_context(instance: dict, container) -> dict:
    return {
        "instance_id":       instance["instance_id"],
        "problem_statement": instance["problem_statement"],
        "hints_text":        instance["hints_text"],
        "repo":              instance["repo"],
        "repo_path":         "/testbed",
        "base_commit":       instance["base_commit"],
        "version":           instance["version"],
        "fail_to_pass":      json.loads(instance["FAIL_TO_PASS"]),
        "container_name":    container.name,
        "localization":      "",
    }


# ---------------------------------------------------------------------------
# Single instance runner
# ---------------------------------------------------------------------------

def run_instance(instance: dict, client: docker.DockerClient) -> dict:
    """
    Run localization for a single instance. Returns a result dict.
    Handles its own container setup and teardown.
    """
    instance_id = instance["instance_id"]
    start_time  = time.time()
    container   = None

    # Check cache first — skip if already successful
    existing = load_localization_state(instance_id)
    if existing and existing.get("status") == "success":
        return {
            "instance_id":      instance_id,
            "repo":             instance["repo"],
            "status":           "cached",
            "iterations":       existing.get("iterations", 0),
            "relevant_files":   existing.get("relevant_files", []),
            "bug_analysis":     existing.get("bug_analysis", ""),
            "elapsed_seconds":  0,
            "error":            None,
        }

    try:
        make_agent_logger(RUN_ID, instance_id, "localization")
        container = setup_container(instance, client)
        context   = build_context(instance, container)
        state     = run_localization(context, container)

        return {
            "instance_id":      instance_id,
            "repo":             instance["repo"],
            "status":           state["status"],
            "iterations":       state.get("iterations", 0),
            "relevant_files":   state.get("relevant_files", []),
            "bug_analysis":     state.get("bug_analysis", ""),
            "elapsed_seconds":  round(time.time() - start_time, 1),
            "error":            None,
        }

    except Exception as e:
        return {
            "instance_id":      instance_id,
            "repo":             instance["repo"],
            "status":           "exception",
            "iterations":       0,
            "relevant_files":   [],
            "bug_analysis":     "",
            "elapsed_seconds":  round(time.time() - start_time, 1),
            "error":            str(e),
        }

    finally:
        if container is not None:
            teardown_container(container)


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def print_summary(results: list[dict]):
    """Print a formatted summary table to the console."""
    print("\n" + "=" * 100)
    print("LOCALIZATION EVAL SUMMARY")
    print("=" * 100)
    print(f"{'Instance ID':<45} {'Repo':<25} {'Status':<10} {'Iters':<7} {'Files found'}")
    print("-" * 100)

    for r in sorted(results, key=lambda x: x["instance_id"]):
        files_found = len(r["relevant_files"])
        files_str   = str(files_found) + (" ✓" if files_found > 0 else " ✗")
        status      = r["status"]
        iters       = str(r["iterations"]) if r["iterations"] > 0 else "-"
        repo        = r["repo"].split("/")[-1][:24]
        print(f"{r['instance_id']:<45} {repo:<25} {status:<10} {iters:<7} {files_str}")

    print("-" * 100)

    total     = len(results)
    succeeded = sum(1 for r in results if r["status"] == "success")
    cached    = sum(1 for r in results if r["status"] == "cached")
    failed    = sum(1 for r in results if r["status"] == "failed")
    exception = sum(1 for r in results if r["status"] == "exception")

    active_results = [r for r in results if r["status"] in ("success", "cached")]
    avg_iters = (
        sum(r["iterations"] for r in active_results) / len(active_results)
        if active_results else 0
    )

    print(f"\nTotal     : {total}")
    print(f"Success   : {succeeded}")
    print(f"Cached    : {cached}")
    print(f"Failed    : {failed}")
    print(f"Exception : {exception}")
    print(f"Avg iters : {avg_iters:.1f} (successful/cached only)")
    print(f"Success % : {(succeeded + cached) / total * 100:.0f}%")


def save_markdown(results: list[dict], path: str = "./localization_eval.md"):
    lines = [
        "# Localization eval results\n",
        f"Split: `{SPLIT}` | Total instances: `{len(results)}`\n",
        "| Instance ID | Repo | Status | Iterations | Files found |",
        "|---|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda x: x["instance_id"]):
        files = ", ".join(f"`{f.replace('/testbed/', '')}`" for f in r["relevant_files"])
        lines.append(
            f"| {r['instance_id']} | {r['repo']} | {r['status']} "
            f"| {r['iterations']} | {files or '—'} |"
        )

    succeeded = sum(1 for r in results if r["status"] in ("success", "cached"))
    lines.append(f"\n**Success rate: {succeeded}/{len(results)} "
                 f"({succeeded/len(results)*100:.0f}%)**")

    Path(path).write_text("\n".join(lines))
    print(f"\nMarkdown saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    pipeline_logger = make_pipeline_logger(RUN_ID)
    pipeline_logger.info(f"Starting localization eval on {SPLIT} split")

    dataset   = load_dataset(DATASET_NAME, split=SPLIT)
    instances = list(dataset)

    if INSTANCE_IDS_FILTER:
        instances = [i for i in instances if i["instance_id"] in INSTANCE_IDS_FILTER]
        pipeline_logger.info(f"Filtered to {len(instances)} instance(s)")

    total = len(instances)
    pipeline_logger.info(f"Running localization on {total} instance(s)")

    client  = docker.from_env()
    results = []

    if EVAL_MAX_WORKERS == 1:
        # Sequential — simpler, easier to debug
        for idx, instance in enumerate(instances, 1):
            instance_id = instance["instance_id"]
            pipeline_logger.info(f"[{idx}/{total}] {instance_id}")
            result = run_instance(instance, client)
            results.append(result)
            pipeline_logger.info(
                f"[{idx}/{total}] {instance_id} → "
                f"{result['status']} in {result['iterations']} iterations"
            )
    else:
        # Parallel — faster but uses more Docker resources
        pipeline_logger.info(f"Running with {EVAL_MAX_WORKERS} parallel workers")
        with ThreadPoolExecutor(max_workers=EVAL_MAX_WORKERS) as executor:
            futures = {
                executor.submit(run_instance, instance, client): instance["instance_id"]
                for instance in instances
            }
            completed = 0
            for future in as_completed(futures):
                instance_id = futures[future]
                completed  += 1
                try:
                    result = future.result()
                    results.append(result)
                    pipeline_logger.info(
                        f"[{completed}/{total}] {instance_id} → "
                        f"{result['status']} in {result['iterations']} iterations"
                    )
                except Exception as e:
                    pipeline_logger.error(f"[{completed}/{total}] {instance_id} → exception: {e}")
                    results.append({
                        "instance_id":     instance_id,
                        "repo":            "",
                        "status":          "exception",
                        "iterations":      0,
                        "relevant_files":  [],
                        "bug_analysis":    "",
                        "elapsed_seconds": 0,
                        "error":           str(e),
                    })

    # Save full results
    eval_path = Path("./localization_eval.json")
    eval_path.write_text(json.dumps(results, indent=2))
    pipeline_logger.info(f"Full results saved → {eval_path}")

    # Print and save summary
    print_summary(results)
    save_markdown(results)


if __name__ == "__main__":
    main()