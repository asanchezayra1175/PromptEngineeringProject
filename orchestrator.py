"""
orchestrator.py

Boots the Docker environment for a SWE-bench instance and builds the
context bundle passed to Agent 1.

Environment variables are loaded from a .env file in the project root.
See .env.example for all available variables.
"""

from dotenv import load_dotenv
load_dotenv()  # must be called before any os.environ.get() reads

from datasets import load_dataset
import docker
import docker.errors
import logging
import json
import os
import tempfile

from swebench.harness.test_spec.test_spec import make_test_spec
from swebench.harness.docker_build import build_env_images, build_container

DATASET_NAME = os.environ.get("DATASET_NAME", "princeton-nlp/SWE-bench_Lite")
SPLIT        = os.environ.get("SPLIT", "dev")
RUN_ID       = os.environ.get("RUN_ID", "agentloop")


class SwebenchCompatLogger(logging.Logger):
    """
    Thin wrapper around stdlib Logger that adds the log_file attribute
    swebench's BuildImageError expects. Without this, any container build
    failure raises AttributeError before the real error message is surfaced.
    """
    def __init__(self, name: str, log_path: str):
        super().__init__(name, logging.INFO)
        self.log_file = log_path
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        self.addHandler(handler)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.addHandler(file_handler)


def make_logger(run_id: str) -> SwebenchCompatLogger:
    log_dir = os.path.join(tempfile.gettempdir(), "swebench-agent", run_id)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "orchestrator.log")
    return SwebenchCompatLogger("swebench-agent", log_path)


def remove_existing_container(client: docker.DockerClient, container_name: str, logger: SwebenchCompatLogger):
    """
    Remove a container by name if it already exists. This handles the 409
    conflict that occurs when re-running against the same instance_id + run_id.
    """
    try:
        existing = client.containers.get(container_name)
        logger.info(f"Found existing container '{container_name}' (status={existing.status}) — removing it.")
        existing.remove(force=True)
        logger.info(f"Removed existing container '{container_name}'.")
    except docker.errors.NotFound:
        pass


def sanity_check(container, repo_module: str, logger: SwebenchCompatLogger) -> bool:
    """
    Verify the package imports cleanly inside the container's shell environment.

    Swebench containers install packages in editable mode under /testbed.
    A bare exec_run("python -c ...") bypasses the shell's PATH and site-packages
    setup, causing spurious ModuleNotFoundError for dependencies like 'erfa'.
    Running via ["bash", "-lc", "..."] picks up the full environment correctly.
    """
    exit_code, output = container.exec_run(
        ["bash", "-lc", f"python -c 'import {repo_module}; print(\"ok\")'"],
        workdir="/testbed",
    )
    output_str = output.decode().strip()

    if exit_code != 0 or "ok" not in output_str:
        logger.warning(
            f"Sanity import check failed for '{repo_module}':\n{output_str}\n"
            "The environment may be broken. Proceeding — Agent 1 should flag this."
        )
        return False

    logger.info(f"Sanity check passed: '{repo_module}' imports cleanly inside the container.")
    return True


def setup_environment(instance: dict, client: docker.DockerClient, logger: SwebenchCompatLogger):
    """
    Build the environment image using environment_setup_commit, then build
    and start a container checked out at base_commit.
    """
    logger.info(
        f"Building env image for {instance['instance_id']} "
        f"(environment_setup_commit={instance['environment_setup_commit']})"
    )
    build_env_images(
        client=client,
        dataset=[instance],
        force_rebuild=False,
        max_workers=1,
        namespace=None,
        instance_image_tag="latest",
        env_image_tag="latest",
    )

    logger.info(f"Building container at base_commit={instance['base_commit']}")

    test_spec = make_test_spec(
        instance,
        namespace=None,
        base_image_tag="latest",
        env_image_tag="latest",
        instance_image_tag="latest",
    )

    expected_container_name = f"sweb.eval.{instance['instance_id']}.{RUN_ID}"
    remove_existing_container(client, expected_container_name, logger)

    container = build_container(
        test_spec=test_spec,
        client=client,
        run_id=RUN_ID,
        logger=logger,
        nocache=False,
        force_rebuild=False,
    )

    container.start()
    logger.info(f"Container '{container.name}' started.")

    repo_module = instance["repo"].split("/")[-1].replace("-", "_")
    sanity_check(container, repo_module, logger)

    return container, test_spec


def build_agent1_context(instance: dict, container, test_spec) -> dict:
    """
    Assemble the structured context bundle passed to Agent 1.

    Intentionally withheld:
      - patch        : gold fix — never exposed to any agent
      - test_patch   : gold test — never exposed to any agent
      - pass_to_pass : reserved for Agent 3 only
    """
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
    }


def build_agent3_context(instance: dict, agent1_context: dict, patch: str) -> dict:
    """
    Context bundle for Agent 3. Adds pass_to_pass and the Agent 2 patch.
    """
    return {
        **agent1_context,
        "patch":        patch,
        "fail_to_pass": json.loads(instance["FAIL_TO_PASS"]),
        "pass_to_pass": json.loads(instance["PASS_TO_PASS"]),
    }


def main():
    logger = make_logger(RUN_ID)

    dataset = load_dataset(DATASET_NAME, split=SPLIT)
    instance = dataset[0]
    logger.info(f"Loaded instance: {instance['instance_id']} (split={SPLIT})")

    client = docker.from_env()

    container, test_spec = setup_environment(instance, client, logger)
    print(container.name)

    agent1_context = build_agent1_context(instance, container, test_spec)
    logger.info("Agent 1 context bundle:\n" + json.dumps(agent1_context, indent=2))

    return agent1_context


if __name__ == "__main__":
    main()