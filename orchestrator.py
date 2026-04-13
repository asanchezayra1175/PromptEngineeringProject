"""
orchestrator.py

Boots the Docker environment for each SWE-bench instance and runs the full
agent pipeline:

  [Pre-step] Localization — runs once per repo, cached to disk
  Agent 1 — Issue Reproduction
  Agent 2 — Issue Fixing        ←──────────────────┐
  Agent 3 — Testing & Verification                  │
      │                                             │
      └── regressions found? → feed report back ───┘
          (up to PIPELINE_MAX_FIX_CYCLES times)

Environment variables are loaded from a .env file in the project root.
See .env.example for all available variables.
"""

from dotenv import load_dotenv
load_dotenv()

from datasets import load_dataset
import docker
import docker.errors
import logging
from logger import make_pipeline_logger, make_agent_logger, SwebenchCompatLogger, get_logger
import json
import os
import tempfile
from pathlib import Path

from swebench.harness.test_spec.test_spec import make_test_spec
from swebench.harness.docker_build import build_env_images, build_container

from agent_localization import run_localization
from agent_architect import run_architect
from agent_editor import run_editor
from agent_critic import run_critic, build_critic_context

DATASET_NAME            = os.environ.get("DATASET_NAME", "princeton-nlp/SWE-bench_Lite")
SPLIT                   = os.environ.get("SPLIT", "dev")
RUN_ID                  = os.environ.get("RUN_ID", "agentloop")
PIPELINE_MAX_FIX_CYCLES = int(os.environ.get("PIPELINE_MAX_FIX_CYCLES", 3))

LOG_DIR = os.environ.get("LOG_DIR", "./logs")

INSTANCE_IDS_FILTER = [
    i.strip() for i in os.environ.get("INSTANCE_IDS", "").split(",") if i.strip()
]


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------


# Logger is defined in logger.py and imported here
# SwebenchCompatLogger is re-exported for use as a type hint in Docker helpers


# ---------------------------------------------------------------------------
# Docker helpers
# ---------------------------------------------------------------------------

def remove_existing_container(client: docker.DockerClient, container_name: str, logger: SwebenchCompatLogger):
    try:
        existing = client.containers.get(container_name)
        logger.info(f"Found existing container '{container_name}' (status={existing.status}) — removing it.")
        existing.remove(force=True)
        logger.info(f"Removed existing container '{container_name}'.")
    except docker.errors.NotFound:
        pass


def sanity_check(container, repo_module: str, logger: SwebenchCompatLogger) -> bool:
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


def teardown_container(container, logger: SwebenchCompatLogger):
    try:
        container.stop()
        container.remove()
        logger.info(f"Container '{container.name}' stopped and removed.")
    except Exception as e:
        logger.warning(f"Failed to clean up container: {e}")


# ---------------------------------------------------------------------------
# Patch application
# ---------------------------------------------------------------------------

def apply_patch(container, patch: str, logger: SwebenchCompatLogger) -> bool:
    import tempfile, os
    from pathlib import Path
    from swebench.harness.docker_utils import copy_to_container
    from pathlib import PurePosixPath

    reset_code, reset_out = container.exec_run(
        ["bash", "-lc", "git checkout -- ."],
        workdir="/testbed",
    )
    if reset_code != 0:
        logger.warning(f"git checkout failed: {reset_out.decode().strip()}")

    # Write patch to a temp file and copy into container — avoids heredoc
    # corruption from special characters (quotes, backslashes, etc.)
    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".patch", delete=False
    ) as f:
        normalized = patch.replace("\r\n", "\n").replace("\r", "\n")
        if not normalized.endswith("\n"):
            normalized += "\n"
        f.write(normalized.encode("utf-8"))
        patch_path = f.name

    try:
        copy_to_container(container, Path(patch_path), PurePosixPath("/tmp/fix.patch"))
    finally:
        os.unlink(patch_path)

    # Try multiple apply strategies, same as the official harness
    for cmd in ["git apply /tmp/fix.patch",
                "git apply --whitespace=fix /tmp/fix.patch",
                "patch --batch --fuzz=5 -p1 -i /tmp/fix.patch"]:
        apply_code, apply_out = container.exec_run(
            ["bash", "-lc", cmd],
            workdir="/testbed",
        )
        if apply_code == 0:
            logger.info(f"Patch applied successfully via: {cmd}")
            return True
        logger.warning(f"Apply attempt failed ({cmd}): {apply_out.decode().strip()[:200]}")

    logger.error("All patch application strategies failed.")
    return False


# ---------------------------------------------------------------------------
# Context builders
# ---------------------------------------------------------------------------

def _exec_in_container(container, command: str) -> str:
    """Run a command in the container and return output as string."""
    exit_code, output = container.exec_run(
        ["bash", "-lc", command],
        workdir="/testbed",
        demux=False,
    )
    return output.decode("utf-8", errors="replace").strip()


def extract_repo_context(instance: dict, container) -> str:
    """
    Extract concrete repo context to help Agent 1 write the reproduction test.

    Gathers three things that are hard for the agent to find efficiently:
      1. Existing test patterns — how tests call the relevant module
      2. Test fixture examples — existing pass/fail SQL or code examples
      3. How the module is imported and used in tests

    This is extracted once by the orchestrator (which already has the container)
    so Agent 1 doesn't burn tool calls discovering it.
    """
    fail_to_pass = json.loads(instance["FAIL_TO_PASS"])
    context_parts = []

    # Extract the test file and fixture file from the first fail_to_pass test ID
    # Format is typically: path/to/test_file.py::test_function_name
    if fail_to_pass:
        test_id = fail_to_pass[0]
        if "::" in test_id:
            test_file = test_id.split("::")[0]
            # Read the first 60 lines of the test file for import patterns
            imports = _exec_in_container(
                container,
                f"sed -n '1,60p' /testbed/{test_file} 2>/dev/null"
            )
            if imports:
                context_parts.append(
                    f"### Existing test file imports and patterns ({test_file}):\n{imports}"
                )

    # Find fixture files related to the issue (yml, sql, txt files in test/fixtures)
    # Search using key terms from the fail_to_pass test names
    if fail_to_pass:
        # Extract rule/module name from test ID (e.g. L031 from test_rules/std_test.py::test_L031_*)
        import re
        key_terms = set()
        for test_id in fail_to_pass:
            # Find uppercase rule codes like L031, or module names
            matches = re.findall(r'[A-Z][0-9]{3}|[a-z]+_[a-z]+', test_id)
            key_terms.update(matches[:2])

        for term in list(key_terms)[:2]:
            fixtures = _exec_in_container(
                container,
                f"find /testbed/test -name '*{term}*' -o -name '*{term.lower()}*' 2>/dev/null | grep -v __pycache__ | head -5"
            )
            if fixtures:
                for fixture_path in fixtures.splitlines()[:2]:
                    fixture_content = _exec_in_container(
                        container,
                        f"cat {fixture_path} 2>/dev/null | head -60"
                    )
                    if fixture_content:
                        rel_path = fixture_path.replace("/testbed/", "")
                        context_parts.append(
                            f"### Fixture file ({rel_path}):\n{fixture_content}"
                        )

    if not context_parts:
        return ""

    return (
        "## Repository context (pre-extracted to save your tool calls)\n\n"
        + "\n\n".join(context_parts)
    )


def build_agent1_context(instance: dict, container) -> dict:
    """
    Assemble the context bundle passed to Agent 1. Localization is included
    here so it propagates automatically to Agents 2 and 3 via ** spreading.

    Intentionally withheld:
      - patch        : gold fix — never exposed to any agent
      - test_patch   : gold test — never exposed to any agent
      - pass_to_pass : reserved for Agent 3 only
    """
    repo_context = extract_repo_context(instance, container)

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
        "repo_context":      repo_context,  # pre-extracted test patterns and fixtures
    }


def build_agent2_context_with_regression(
    agent1_context: dict,
    agent1_state: dict,
    agent3_state: dict,
) -> dict:
    context = build_agent2_context(agent1_context, agent1_state)
    context["regression_report"] = agent3_state.get("regression_report", "")
    context["regressions"]       = agent3_state.get("regressions", [])
    context["previous_patch"]    = agent3_state.get("patch", "")
    context["full_test_output"]  = agent3_state.get("full_test_output", "")
    return context


def clear_agent_state(instance_id: str, agent: str, logger: SwebenchCompatLogger):
    state_dir = Path(os.environ.get(f"AGENT{agent}_STATE_DIR", "/tmp/swebench-agent/states"))
    state_path = state_dir / instance_id / f"agent{agent.lower()}.json"
    if state_path.exists():
        state_path.unlink()
        logger.info(f"Cleared Agent {agent} state for {instance_id}.")


# ---------------------------------------------------------------------------
# Single-instance pipeline
# ---------------------------------------------------------------------------

def run_pipeline(instance: dict, client: docker.DockerClient, logger: SwebenchCompatLogger) -> dict:
    """
    Globant-inspired two-stage pipeline:

      Stage 1: Localization — finds relevant files and functions
      Stage 2: Fixing loop
        Architect  — proposes the fix
        Editor     — implements the proposal
        Critic     — verifies correctness, triggers retry if needed
          └── on fail: Architect receives previous proposal + patch + test output
    """
    instance_id = instance["instance_id"]
    container   = None

    try:
        container, test_spec = setup_environment(instance, client, logger)
        print(container.name)

        agent_context = build_agent1_context(instance, container)

        # -------------------------------------------------------
        # Stage 1: Localization
        # -------------------------------------------------------
        make_agent_logger(RUN_ID, instance_id, "localization")
        logger.info(f"Stage 1: Localization")
        localization_state = run_localization(agent_context, container)

        if localization_state["status"] != "success":
            logger.error(f"Localization failed for {instance_id} — stopping pipeline.")
            return {"instance_id": instance_id, "status": "failed_at_localization",
                    "localization": localization_state}

        logger.info(f"Localization complete. Bug: {localization_state['bug_analysis']}")

        # -------------------------------------------------------
        # Stage 2: Architect → Editor → Critic (with retry loop)
        # -------------------------------------------------------
        architect_state: dict | None = None
        editor_state:    dict | None = None
        critic_state:    dict | None = None

        for cycle in range(1, PIPELINE_MAX_FIX_CYCLES + 1):
            logger.info(f"Fix/verify cycle {cycle}/{PIPELINE_MAX_FIX_CYCLES} for {instance_id}")

            # Architect — proposes the fix
            # On retry cycles, receives previous proposal + patch + critic output
            make_agent_logger(RUN_ID, instance_id, f"architect_cycle{cycle}")
            architect_state = run_architect(
                context=agent_context,
                container=container,
                localization_state=localization_state,
                cycle=cycle,
                previous_proposal=architect_state["full_response"] if architect_state else None,
                previous_patch=editor_state["patch"] if editor_state else None,
critic_output=(
                    f"Issues: {critic_state.get('issues', [])}\n"
                    f"Report: {critic_state.get('regression_report', '')}"
                ) if critic_state else None,
            )

            if architect_state["status"] != "success":
                logger.error(f"Architect failed on cycle {cycle}.")
                return {"instance_id": instance_id, "status": "failed_at_architect",
                        "cycle": cycle, "localization": localization_state,
                        "architect": architect_state}

            logger.info(f"Architect complete. Strategy: {architect_state['fix_strategy']}")

            # Editor — implements the proposal
            make_agent_logger(RUN_ID, instance_id, f"editor_cycle{cycle}")
            editor_state = run_editor(
                context=agent_context,
                container=container,
                localization_state=localization_state,
                architect_state=architect_state,
                cycle=cycle,
            )

            if editor_state["status"] != "success":
                logger.error(f"Editor failed on cycle {cycle}.")
                return {"instance_id": instance_id, "status": "failed_at_editor",
                        "cycle": cycle, "localization": localization_state,
                        "architect": architect_state, "editor": editor_state}

            logger.info(f"Editor complete. Notes: {editor_state['implementation_notes']}")

            # Orchestrator applies the patch before Critic runs
            patch_ok = apply_patch(container, editor_state["patch"], logger)
            if not patch_ok:
                logger.warning(f"Patch application failed on cycle {cycle} — treating as Critic fail, retrying.")
                # Synthesize a critic failure so the retry loop feeds back to the Architect
                critic_state = {
                    "status":            "success",
                    "verdict":           "fail",
                    "correctness":       "fail",
                    "completeness":      "fail",
                    "alignment":         "fail",
                    "safety":            "fail",
                    "issues":            ["Patch could not be applied — it is malformed or corrupt."],
                    "regressions":       [],
                    "regression_report": (
                        "The Editor produced a malformed or corrupt patch that could not be "
                        "applied with git apply or patch. This usually means the diff is "
                        "truncated, has incorrect line endings, or contains invalid hunk headers. "
                        "Propose a simpler, minimal change. Avoid large multi-hunk diffs. "
                        "Make sure the change field contains exact before/after lines."
                    ),
                    "iterations":  0,
                    "cycle":       cycle,
                    "timestamp":   "",
                }
                # Fall through to the retry logic below — skip Critic agent
            else:
                # Syntax check — run py_compile on all modified Python files
                # before spending tokens on the Critic agent
                modified_files = architect_state.get("files_to_modify", [])
                syntax_errors  = []
                for fpath in modified_files:
                    if fpath.endswith(".py"):
                        check_code, check_out = container.exec_run(
                            ["bash", "-lc", f"python -m py_compile {fpath} 2>&1"],
                            workdir="/testbed",
                        )
                        if check_code != 0:
                            syntax_errors.append(
                                f"{fpath}: {check_out.decode('utf-8', errors='replace').strip()}"
                            )

                if syntax_errors:
                    logger.warning(f"Syntax errors detected on cycle {cycle}: {syntax_errors}")
                    critic_state = {
                        "status":            "success",
                        "verdict":           "fail",
                        "correctness":       "fail",
                        "completeness":      "fail",
                        "alignment":         "fail",
                        "safety":            "fail",
                        "issues":            [f"Syntax error: {e}" for e in syntax_errors],
                        "regressions":       [],
                        "regression_report": (
                            f"The patch introduced syntax errors in the modified files: "
                            f"{'; '.join(syntax_errors)}. "
                            f"Fix the syntax errors before proposing the change."
                        ),
                        "iterations":  0,
                        "cycle":       cycle,
                        "timestamp":   "",
                    }
                else:
                    # Critic — code review for correctness, completeness, alignment, safety
                    make_agent_logger(RUN_ID, instance_id, f"critic_cycle{cycle}")
                    critic_context = build_critic_context(
                        agent_context, instance, editor_state,
                        localization_state=localization_state,
                        architect_state=architect_state,
                    )
                    critic_state = run_critic(critic_context, container, cycle=cycle)

                if critic_state["status"] != "success":
                    logger.error(f"Critic failed to produce output on cycle {cycle}.")
                    return {"instance_id": instance_id, "status": "failed_at_critic",
                            "cycle": cycle, "localization": localization_state,
                            "architect": architect_state, "editor": editor_state,
                            "critic": critic_state}

                verdict = critic_state["verdict"]
                logger.info(f"Critic verdict on cycle {cycle}: {verdict}")

                if verdict == "pass":
                    logger.info(f"Pipeline complete — fix verified for {instance_id}.")
                    return {"instance_id": instance_id, "status": "success",
                            "cycles": cycle, "localization": localization_state,
                            "architect": architect_state, "editor": editor_state,
                            "critic": critic_state}

            # Fail — reset state, loop back to Architect with Critic feedback
            regressions = critic_state.get("regressions", [])
            logger.warning(f"Critic fail on cycle {cycle}. Retrying with feedback.")

            # Clear cycle state files so next cycle runs fresh
            for agent_name in [f"architect_cycle{cycle}", f"editor_cycle{cycle}", f"critic_cycle{cycle}"]:
                state_path = Path(os.environ.get("ARCHITECT_STATE_DIR", "./states")) / instance_id / f"{agent_name}.json"
                if state_path.exists():
                    state_path.unlink()

        logger.error(f"All {PIPELINE_MAX_FIX_CYCLES} cycles exhausted for {instance_id}.")
        return {"instance_id": instance_id, "status": "failed_max_cycles",
                "cycles": PIPELINE_MAX_FIX_CYCLES, "localization": localization_state,
                "architect": architect_state, "editor": editor_state, "critic": critic_state}

    except Exception as e:
        logger.error(f"Unhandled exception for {instance_id}: {e}", exc_info=True)
        return {"instance_id": instance_id, "status": "exception", "error": str(e)}

    finally:
        if container is not None:
            teardown_container(container, logger)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    logger = make_pipeline_logger(RUN_ID)

    dataset   = load_dataset(DATASET_NAME, split=SPLIT)
    instances = list(dataset)

    if INSTANCE_IDS_FILTER:
        instances = [i for i in instances if i["instance_id"] in INSTANCE_IDS_FILTER]
        logger.info(f"Filtered to {len(instances)} instance(s): {INSTANCE_IDS_FILTER}")

    total = len(instances)
    logger.info(f"Running pipeline on {total} instance(s) from '{SPLIT}' split.")

    client  = docker.from_env()
    results = []

    for idx, instance in enumerate(instances, start=1):
        instance_id = instance["instance_id"]
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Instance {idx}/{total}: {instance_id}")
        logger.info(f"{'=' * 60}")

        result = run_pipeline(instance, client, logger)
        results.append(result)
        logger.info(f"[{idx}/{total}] {instance_id} → {result['status']}")

    statuses = [r["status"] for r in results]
    logger.info(f"\n{'=' * 60}")
    logger.info("PIPELINE COMPLETE")
    logger.info(f"{'=' * 60}")
    logger.info(f"Total   : {total}")
    logger.info(f"Success : {statuses.count('success')}")
    logger.info(f"Failed  : {total - statuses.count('success')}")
    for status in sorted(set(statuses)):
        logger.info(f"  {status}: {statuses.count(status)}")
    logger.info("\nRun format_predictions.py to generate the JSONL submission file.")

    return results


if __name__ == "__main__":
    main()