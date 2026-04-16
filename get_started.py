from swebench.harness.docker_build import build_env_images, build_container
from swebench.harness.test_spec.test_spec import make_test_spec
from datasets import load_dataset
from logger import make_pipeline_logger
import docker
import logging

# Create a logger for the build process
logger = make_pipeline_logger("manual_build")

client = docker.from_env()

# Load real instances from SWE-bench dataset
dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="dev")
instances = list(dataset)

# Filter to specific instance IDs
target_ids = ["pylint-dev__astroid-1866", "pylint-dev__astroid-1978"]
instances = [i for i in instances if i["instance_id"] in target_ids]

logger.info(f"Building images for {len(instances)} instance(s)...")

# Build environment images
build_env_images(
    client=client,
    dataset=instances,
    force_rebuild=False,
    max_workers=1,
    namespace=None,
    instance_image_tag="latest",
    env_image_tag="latest",
)

logger.info("Environment images built successfully!")

# Build instance containers
for instance in instances:
    logger.info(f"Building container for {instance['instance_id']}...")
    test_spec = make_test_spec(instance, namespace=None)
    container = build_container(
        test_spec=test_spec,
        client=client,
        run_id="manual_build",
        logger=logger,
        nocache=False,
        force_rebuild=False,
    )
    logger.info(f"Container created: {container.name}")