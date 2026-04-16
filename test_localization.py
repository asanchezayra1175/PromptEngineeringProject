"""
test_localization.py

Runs the Localization Agent in isolation against a single SWE-bench instance.
Use this to validate and tune the localization prompt before running the
full pipeline.

Usage:
    python test_localization.py

Output:
    - Prints the localization report to the console
    - Saves state to ./states/{instance_id}/localization.json
    - Saves logs to ./logs/{RUN_ID}/{instance_id}/localization.log

To test a different instance, set INSTANCE_IDS in your .env:
    INSTANCE_IDS=astropy__astropy-12907
"""

from dotenv import load_dotenv
load_dotenv()

from datasets import load_dataset
import docker
import json
import logging
import os
from pathlib import Path

from swebench.harness.test_spec.test_spec import make_test_spec
from swebench.harness.docker_build import build_env_images, build_container

from logger import make_pipeline_logger, make_agent_logger
from agent_localization import run_localization

DATASET_NAME = os.environ.get("DATASET_NAME", "princeton-nlp/SWE-bench_Lite")
SPLIT        = os.environ.get("SPLIT", "dev")
RUN_ID       = os.environ.get("RUN_ID", "agentloop")
INSTANCE_IDS = [
    i.strip() for i in os.environ.get("INSTANCE_IDS", "").split(",") if i.strip()
]


class SwebenchCompatLogger(logging.Logger):
    def __init__(self, name: str, log_path: str):
        super().__init__(name, logging.INFO)
        self.log_file = log_path
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        self.addHandler(handler)
        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.addHandler(file_handler)

    def _shutdown(self):
        for h in self.handlers:
            try:
                h.flush(); h.close()
            except Exception:
                pass


def make_swebench_logger(run_id: str) -> SwebenchCompatLogger:
    log_dir = Path("./logs") / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    return SwebenchCompatLogger("swebench", str(log_dir / "setup.log"))


def setup_container(instance: dict, client: docker.DockerClient, swlogger):
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


def main():
    make_pipeline_logger(RUN_ID)

    dataset   = load_dataset(DATASET_NAME, split=SPLIT)
    instances = list(dataset)

    if INSTANCE_IDS:
        instances = [i for i in instances if i["instance_id"] in INSTANCE_IDS]
        if not instances:
            print(f"No instances found matching INSTANCE_IDS={INSTANCE_IDS}")
            return

    instance    = instances[0]
    instance_id = instance["instance_id"]
    print(f"\nTesting localization on: {instance_id}")
    print(f"Problem: {instance['problem_statement'][:200]}...\n")

    client    = docker.from_env()
    swlogger  = make_swebench_logger(RUN_ID)
    container = setup_container(instance, client, swlogger)
    print(f"Container started: {container.name}\n")

    make_agent_logger(RUN_ID, instance_id, "localization")

    context = build_context(instance, container)
    state   = run_localization(context, container)

    print("\n" + "=" * 60)
    print("LOCALIZATION RESULT")
    print("=" * 60)
    print(f"Status     : {state['status']}")
    print(f"Iterations : {state['iterations']}")

    if state["status"] == "success":
        print(f"\nRelevant files:")
        for f in state["relevant_files"]:
            print(f"  {f}")
        print(f"\nRelevant symbols:")
        for s in state["relevant_symbols"]:
            print(f"  {s}")
        print(f"\nBug analysis:\n  {state['bug_analysis']}")
        print(f"\nExploration notes:\n  {state['exploration_notes'][:500]}")
    else:
        print("Localization failed — check the log for details.")

    print(f"\nFull state saved to: ./states/{instance_id}/localization.json")
    print(f"Log saved to       : ./logs/{RUN_ID}/{instance_id}/localization.log")

    try:
        container.stop()
        container.remove()
        print(f"Container cleaned up.")
    except Exception as e:
        print(f"Container cleanup warning: {e}")

    return state


if __name__ == "__main__":
    main()