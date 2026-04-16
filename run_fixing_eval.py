"""
run_fixing_eval.py

Runs the Architect, Editor, and/or Critic agents in isolation across instances,
reading localization results from a directory of JSON files rather than
running localization again.

This lets you evaluate and tune each agent prompt independently before running
the full pipeline.

Usage:
    # Run all three agents (architect + editor + critic):
    python run_fixing_eval.py --localization-dir ./states --agent all

    # Run only the Architect:
    python run_fixing_eval.py --localization-dir ./states --agent architect

    # Run only the Editor (requires existing architect state):
    python run_fixing_eval.py --localization-dir ./states --agent editor

    # Run only the Critic (requires existing editor state with patch):
    python run_fixing_eval.py --localization-dir ./states --agent critic

    # Run Architect + Editor (no critic):
    python run_fixing_eval.py --localization-dir ./states --agent both

    # Run specific instances only:
    python run_fixing_eval.py --localization-dir ./states --instance-ids sqlfluff__sqlfluff-1625

Output:
    - Console summary table
    - ./architect_eval.json / ./editor_eval.json / ./critic_eval.json
    - ./architect_eval.md  / ./editor_eval.md  / ./critic_eval.md

Environment variables:
    EVAL_MAX_WORKERS        : parallel workers (default: 1)
    SKIP_INSTANCES          : comma-separated IDs to skip
    CONTAINER_BUILD_TIMEOUT : max seconds for container build (default: 600)
"""

from dotenv import load_dotenv
load_dotenv()

from datasets import load_dataset
import argparse
import docker
import docker.errors
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from swebench.harness.test_spec.test_spec import make_test_spec
from swebench.harness.docker_build import build_env_images, build_container

from logger import make_pipeline_logger, make_agent_logger, get_logger
from agent_architect import run_architect, load_state as load_architect_state
from agent_editor import run_editor, load_state as load_editor_state
from agent_critic import run_critic, build_critic_context, load_state as load_critic_state


def apply_patch(container, patch: str, logger) -> bool:
    """Apply a unified diff patch to the container."""
    reset_code, reset_out = container.exec_run(
        ["bash", "-lc", "git checkout -- ."],
        workdir="/testbed",
    )
    if reset_code != 0:
        logger.warning(f"git checkout failed: {reset_out.decode().strip()}")

    container.exec_run(
        ["bash", "-lc", f"cat > /tmp/fix.patch << 'PATCHEOF'\n{patch}\nPATCHEOF"],
        workdir="/testbed",
    )

    apply_code, apply_out = container.exec_run(
        ["bash", "-lc", "git apply /tmp/fix.patch"],
        workdir="/testbed",
    )
    apply_out_str = apply_out.decode().strip()

    if apply_code != 0:
        logger.error(f"Patch application failed:\n{apply_out_str}")
        return False

    logger.info("Patch applied successfully.")
    return True

DATASET_NAME            = os.environ.get("DATASET_NAME", "princeton-nlp/SWE-bench_Lite")
SPLIT                   = os.environ.get("SPLIT", "dev")
RUN_ID                  = os.environ.get("RUN_ID", "agentloop")
EVAL_MAX_WORKERS        = int(os.environ.get("EVAL_MAX_WORKERS", 1))
SKIP_INSTANCES          = [
    i.strip() for i in os.environ.get("SKIP_INSTANCES", "").split(",") if i.strip()
]
CONTAINER_BUILD_TIMEOUT = int(os.environ.get("CONTAINER_BUILD_TIMEOUT", 600))


# ---------------------------------------------------------------------------
# Swebench-compatible logger
# ---------------------------------------------------------------------------

class SwebenchCompatLogger(logging.Logger):
    def __init__(self, name: str, log_path: str):
        super().__init__(name, logging.INFO)
        self.log_file = log_path
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.addHandler(h)
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.addHandler(fh)


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
        client.containers.get(container_name).remove(force=True)
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
# Localization state loader
# ---------------------------------------------------------------------------

def load_localization(localization_dir: Path, instance_id: str) -> dict | None:
    """
    Load localization state for an instance from the given directory.

    Looks for:
      {localization_dir}/{instance_id}/localization.json
    """
    path = localization_dir / instance_id / "localization.json"
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return None
        state = json.loads(text)
        if state.get("status") != "success":
            return None
        return state
    except (json.JSONDecodeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Agent context builder
# ---------------------------------------------------------------------------

def build_agent_context(instance: dict, container) -> dict:
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

def run_instance(
    instance: dict,
    client: docker.DockerClient,
    localization_dir: Path,
    run_agent: str,  # "architect", "editor", "critic", "both", or "all"
) -> dict:
    instance_id = instance["instance_id"]
    start_time  = time.time()
    container   = None

    if instance_id in SKIP_INSTANCES:
        return {
            "instance_id": instance_id,
            "repo":        instance["repo"],
            "status":      "skipped",
            "architect":   None,
            "editor":      None,
            "critic":      None,
            "elapsed":     0,
            "error":       "In SKIP_INSTANCES list",
        }

    # Load localization — required for all agents
    localization_state = load_localization(localization_dir, instance_id)
    if localization_state is None:
        return {
            "instance_id": instance_id,
            "repo":        instance["repo"],
            "status":      "no_localization",
            "architect":   None,
            "editor":      None,
            "critic":      None,
            "elapsed":     0,
            "error":       f"No successful localization.json found in {localization_dir / instance_id}",
        }

    try:
        # Boot container with timeout
        import concurrent.futures as _cf
        with _cf.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(setup_container, instance, client)
            try:
                container = future.result(timeout=CONTAINER_BUILD_TIMEOUT)
            except _cf.TimeoutError:
                return {
                    "instance_id": instance_id,
                    "repo":        instance["repo"],
                    "status":      "timeout",
                    "architect":   None,
                    "editor":      None,
                    "elapsed":     round(time.time() - start_time, 1),
                    "error":       f"Container build timed out after {CONTAINER_BUILD_TIMEOUT}s",
                }

        context = build_agent_context(instance, container)

        architect_result = None
        editor_result    = None
        critic_result    = None
        architect_state  = None
        editor_state     = None

        # Run Architect
        if run_agent in ("architect", "both", "all"):
            make_agent_logger(RUN_ID, instance_id, "architect_cycle1")
            architect_state  = run_architect(
                context=context,
                container=container,
                localization_state=localization_state,
                cycle=1,
            )
            architect_result = {
                "status":           architect_state["status"],
                "fix_strategy":     architect_state.get("fix_strategy", ""),
                "files_to_modify":  architect_state.get("files_to_modify", []),
                "proposed_changes": architect_state.get("proposed_changes", []),
                "iterations":       architect_state.get("iterations", 0),
            }

        # Run Editor (requires Architect to have succeeded)
        if run_agent in ("editor", "both", "all"):
            if run_agent in ("editor",):
                # Load existing architect state from disk
                architect_state = load_architect_state(instance_id, cycle=1)
                if architect_state is None or architect_state.get("status") != "success":
                    return {
                        "instance_id": instance_id,
                        "repo":        instance["repo"],
                        "status":      "no_architect",
                        "architect":   None,
                        "editor":      None,
                        "critic":      None,
                        "elapsed":     round(time.time() - start_time, 1),
                        "error":       "No successful architect_cycle1.json found — run architect first",
                    }

            if architect_state and architect_state.get("status") == "success":
                make_agent_logger(RUN_ID, instance_id, "editor_cycle1")
                editor_state  = run_editor(
                    context=context,
                    container=container,
                    localization_state=localization_state,
                    architect_state=architect_state,
                    cycle=1,
                )
                editor_result = {
                    "status":               editor_state["status"],
                    "patch":                editor_state.get("patch", ""),
                    "implementation_notes": editor_state.get("implementation_notes", ""),
                    "iterations":           editor_state.get("iterations", 0),
                    "patch_lines":          len((editor_state.get("patch") or "").splitlines()),
                }
            else:
                editor_result = {
                    "status":  "skipped",
                    "reason":  "Architect did not succeed",
                }

        # Run Critic (requires Editor to have produced a patch)
        if run_agent in ("critic", "all"):
            if run_agent == "critic":
                # Load existing editor state from disk
                editor_state = load_editor_state(instance_id, cycle=1)
                if editor_state is None or editor_state.get("status") != "success":
                    return {
                        "instance_id": instance_id,
                        "repo":        instance["repo"],
                        "status":      "no_editor",
                        "architect":   None,
                        "editor":      None,
                        "critic":      None,
                        "elapsed":     round(time.time() - start_time, 1),
                        "error":       "No successful editor_cycle1.json found — run editor first",
                    }

            if editor_state and editor_state.get("status") == "success":
                # Apply the patch before running Critic
                patch = editor_state.get("patch", "")
                if patch:
                    import logging as _logging
                    _plog = _logging.getLogger("patch")
                    patch_ok = apply_patch(container, patch, _plog)
                    if not patch_ok:
                        critic_result = {
                            "status":  "patch_failed",
                            "reason":  "Patch could not be applied to container",
                        }
                    else:
                        # Load localization/architect state for Critic context
                        _loc_state  = load_localization(localization_dir, instance_id) or {}
                        _arch_state = load_architect_state(instance_id, cycle=1) or {}

                        make_agent_logger(RUN_ID, instance_id, "critic_cycle1")
                        critic_ctx   = build_critic_context(
                            context, instance, editor_state,
                            localization_state=_loc_state,
                            architect_state=_arch_state,
                        )
                        critic_state = run_critic(critic_ctx, container, cycle=1)
                        critic_result = {
                            "status":            critic_state["status"],
                            "verdict":           critic_state.get("verdict", ""),
                            "correctness":       critic_state.get("correctness", ""),
                            "completeness":      critic_state.get("completeness", ""),
                            "alignment":         critic_state.get("alignment", ""),
                            "safety":            critic_state.get("safety", ""),
                            "issues":            critic_state.get("issues", []),
                            "regression_report": critic_state.get("regression_report", ""),
                            "iterations":        critic_state.get("iterations", 0),
                        }
                else:
                    critic_result = {
                        "status": "skipped",
                        "reason": "Editor patch was empty",
                    }
            else:
                critic_result = {
                    "status": "skipped",
                    "reason": "Editor did not succeed",
                }

        # Determine overall status
        if run_agent == "architect":
            overall = architect_result["status"] if architect_result else "skipped"
        elif run_agent == "editor":
            overall = editor_result["status"] if editor_result else "skipped"
        elif run_agent == "critic":
            c = critic_result or {}
            overall = "pass" if c.get("verdict") == "pass" else c.get("status", "skipped")
        elif run_agent == "both":
            a_ok = architect_result and architect_result["status"] == "success"
            e_ok = editor_result and editor_result["status"] == "success"
            overall = "success" if (a_ok and e_ok) else "partial" if (a_ok or e_ok) else "failed"
        else:  # all
            a_ok = architect_result and architect_result["status"] == "success"
            e_ok = editor_result and editor_result["status"] == "success"
            c_ok = critic_result and critic_result.get("verdict") == "pass"
            overall = "pass" if c_ok else "fixed_no_pass" if (a_ok and e_ok) else "partial" if (a_ok or e_ok) else "failed"

        return {
            "instance_id": instance_id,
            "repo":        instance["repo"],
            "status":      overall,
            "architect":   architect_result,
            "editor":      editor_result,
            "critic":      critic_result,
            "elapsed":     round(time.time() - start_time, 1),
            "error":       None,
        }

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[run_fixing_eval] EXCEPTION for {instance_id}: {e}")
        print(tb)
        try:
            get_logger().error(f"[run_fixing_eval] exception for {instance_id}: {e}")
            get_logger().error(tb)
        except Exception:
            pass
        return {
            "instance_id": instance_id,
            "repo":        instance["repo"],
            "status":      "exception",
            "architect":   None,
            "editor":      None,
            "critic":      None,
            "elapsed":     round(time.time() - start_time, 1),
            "error":       str(e),
        }

    finally:
        if container is not None:
            teardown_container(container)


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def print_summary(results: list, run_agent: str):
    label = run_agent.capitalize()
    print(f"\n{'=' * 100}")
    print(f"{label.upper()} EVAL SUMMARY")
    print("=" * 100)

    if run_agent in ("architect", "both"):
        print(f"{'Instance ID':<45} {'Status':<12} {'A.Iters':<9} {'Files':<6} {'Strategy snippet'}")
        print("-" * 100)
        for r in sorted(results, key=lambda x: x["instance_id"]):
            a    = r.get("architect") or {}
            stat = r["status"]
            iters = str(a.get("iterations", "-"))
            files = str(len(a.get("files_to_modify", [])))
            strat = (a.get("fix_strategy") or "")[:50]
            print(f"{r['instance_id']:<45} {stat:<12} {iters:<9} {files:<6} {strat}")

    if run_agent in ("editor", "both", "all"):
        print(f"\n{'Instance ID':<45} {'Status':<12} {'E.Iters':<9} {'Patch lines':<13} {'Notes snippet'}")
        print("-" * 100)
        for r in sorted(results, key=lambda x: x["instance_id"]):
            e     = r.get("editor") or {}
            stat  = r["status"]
            iters = str(e.get("iterations", "-"))
            lines = str(e.get("patch_lines", "-"))
            notes = (e.get("implementation_notes") or "")[:50]
            print(f"{r['instance_id']:<45} {stat:<12} {iters:<9} {lines:<13} {notes}")

    if run_agent in ("critic", "all"):
        print(f"\n{'Instance ID':<45} {'Verdict':<10} {'C.Iters':<9} {'Regressions':<13} {'Report snippet'}")
        print("-" * 100)
        for r in sorted(results, key=lambda x: x["instance_id"]):
            c       = r.get("critic") or {}
            verdict = c.get("verdict") or c.get("status", "-")
            iters   = str(c.get("iterations", "-"))
            regs    = str(len(c.get("regressions", [])))
            report  = (c.get("regression_report") or "")[:50]
            print(f"{r['instance_id']:<45} {verdict:<10} {iters:<9} {regs:<13} {report}")

    print("-" * 100)
    total     = len(results)
    passed    = sum(1 for r in results if r["status"] in ("pass",))
    succeeded = sum(1 for r in results if r["status"] in ("success", "pass", "fixed_no_pass"))
    partial   = sum(1 for r in results if r["status"] == "partial")
    failed    = sum(1 for r in results if r["status"] in ("failed", "exception"))
    skipped   = sum(1 for r in results if r["status"] in ("skipped", "no_localization",
                                                            "no_architect", "no_editor", "timeout"))

    print(f"\nTotal      : {total}")
    if run_agent in ("critic", "all"):
        print(f"Passed     : {passed}  (all tests pass)")
    print(f"Succeeded  : {succeeded}")
    print(f"Partial    : {partial}")
    print(f"Failed     : {failed}")
    print(f"Skipped    : {skipped}")
    if run_agent in ("critic", "all"):
        print(f"Pass %     : {passed / total * 100:.0f}%")
    else:
        print(f"Success %  : {succeeded / total * 100:.0f}%")


def save_results(results: list, run_agent: str, output_dir: Path):
    agent_list = {
        "architect": ["architect"],
        "editor":    ["editor"],
        "critic":    ["critic"],
        "both":      ["architect", "editor"],
        "all":       ["architect", "editor", "critic"],
    }.get(run_agent, [run_agent])
    for agent in agent_list:
        agent_results = []
        for r in results:
            agent_results.append({
                "instance_id": r["instance_id"],
                "repo":        r["repo"],
                "status":      r["status"],
                "elapsed":     r["elapsed"],
                "error":       r["error"],
                **({agent: r[agent]} if r.get(agent) else {}),
            })

        json_path = output_dir / f"{agent}_eval.json"
        json_path.write_text(json.dumps(agent_results, indent=2))
        print(f"{agent.capitalize()} results saved -> {json_path}")

        # Markdown
        lines = [
            f"# {agent.capitalize()} eval results\n",
            f"Split: `{SPLIT}` | Total: `{len(results)}`\n",
            f"| Instance ID | Status | Iterations | Details |",
            f"|---|---|---|---|",
        ]
        for r in sorted(agent_results, key=lambda x: x["instance_id"]):
            a     = r.get(agent) or {}
            iters = a.get("iterations", "-")
            if agent == "architect":
                detail = (a.get("fix_strategy") or "")[:80]
            elif agent == "editor":
                detail = f"{a.get('patch_lines', '-')} patch lines"
            else:  # critic
                verdict = a.get("verdict") or a.get("status", "-")
                regs    = len(a.get("regressions", []))
                detail  = f"verdict={verdict} regressions={regs}"
            lines.append(f"| {r['instance_id']} | {r['status']} | {iters} | {detail} |")

        succeeded = sum(1 for r in agent_results if r["status"] == "success")
        lines.append(f"\n**Success rate: {succeeded}/{len(results)} "
                     f"({succeeded / len(results) * 100:.0f}%)**")
        md_path = output_dir / f"{agent}_eval.md"
        md_path.write_text("\n".join(lines))
        print(f"{agent.capitalize()} markdown saved -> {md_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate Architect and/or Editor agents in isolation.")
    parser.add_argument(
        "--localization-dir",
        type=Path,
        default=Path("./states"),
        help="Directory containing {instance_id}/localization.json files (default: ./states)",
    )
    parser.add_argument(
        "--agent",
        choices=["architect", "editor", "critic", "both", "all"],
        default="both",
        help="Which agent(s) to run: architect, editor, critic, both (arch+editor), all (default: both)",
    )
    parser.add_argument(
        "--instance-ids",
        type=str,
        default="",
        help="Comma-separated instance IDs to run (default: all with localization)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Where to save eval JSON and markdown (default: current directory)",
    )
    args = parser.parse_args()

    make_pipeline_logger(RUN_ID)

    localization_dir = args.localization_dir
    if not localization_dir.exists():
        print(f"Error: localization directory '{localization_dir}' does not exist.")
        return

    # Load dataset for instance metadata
    dataset   = load_dataset(DATASET_NAME, split=SPLIT)
    instances = list(dataset)

    # Filter by --instance-ids arg or INSTANCE_IDS env var
    filter_ids = [i.strip() for i in args.instance_ids.split(",") if i.strip()]
    if not filter_ids:
        filter_ids = [
            i.strip() for i in os.environ.get("INSTANCE_IDS", "").split(",") if i.strip()
        ]

    if filter_ids:
        instances = [i for i in instances if i["instance_id"] in filter_ids]
    else:
        # Only run instances that have a localization file
        instances = [
            i for i in instances
            if (localization_dir / i["instance_id"] / "localization.json").exists()
        ]

    if not instances:
        print(f"No instances found. Check --localization-dir and --instance-ids.")
        return

    total = len(instances)
    print(f"Running [{args.agent}] agent(s) on {total} instance(s)")
    print(f"Localization dir : {localization_dir}")
    print(f"Output dir       : {args.output_dir}")

    client  = docker.from_env()
    results = []

    if EVAL_MAX_WORKERS == 1:
        for idx, instance in enumerate(instances, 1):
            instance_id = instance["instance_id"]
            print(f"[{idx}/{total}] {instance_id}")
            result = run_instance(instance, client, localization_dir, args.agent)
            results.append(result)
            print(f"[{idx}/{total}] {instance_id} -> {result['status']} ({result['elapsed']}s)")
    else:
        with ThreadPoolExecutor(max_workers=EVAL_MAX_WORKERS) as executor:
            futures = {
                executor.submit(run_instance, inst, client, localization_dir, args.agent): inst["instance_id"]
                for inst in instances
            }
            completed = 0
            for future in as_completed(futures):
                instance_id = futures[future]
                completed  += 1
                try:
                    result = future.result()
                    results.append(result)
                    print(f"[{completed}/{total}] {instance_id} -> {result['status']}")
                except Exception as e:
                    print(f"[{completed}/{total}] {instance_id} -> exception: {e}")
                    results.append({
                        "instance_id": instance_id,
                        "repo":        "",
                        "status":      "exception",
                        "architect":   None,
                        "editor":      None,
                        "critic":      None,
                        "elapsed":     0,
                        "error":       str(e),
                    })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print_summary(results, args.agent)
    save_results(results, args.agent, args.output_dir)


if __name__ == "__main__":
    main()