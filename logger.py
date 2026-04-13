"""
logger.py

Shared logging utility for the SWE-bench agent pipeline.

Produces two levels of logs:

  Pipeline log (one per run):
    {LOG_DIR}/{run_id}/pipeline.log
    High-level orchestrator output — which instance is running, pipeline
    status, errors. Written by the orchestrator.

  Agent logs (one per instance per agent):
    {LOG_DIR}/{run_id}/{instance_id}/agent1.log
    {LOG_DIR}/{run_id}/{instance_id}/agent2.log
    {LOG_DIR}/{run_id}/{instance_id}/agent3.log
    Detailed per-agent output — every iteration, tool call, tool result,
    and state save for that specific instance. Written by each agent.

Usage:
    # Orchestrator — create the pipeline logger once at startup:
    from logger import make_pipeline_logger
    logger = make_pipeline_logger(run_id)

    # Agents — create an agent logger when starting work on an instance:
    from logger import make_agent_logger
    logger = make_agent_logger(run_id, instance_id, agent_name="agent1")

    # tool_executor — retrieve the current agent logger:
    from logger import get_logger
    logger = get_logger()

Environment variables:
    LOG_DIR : root directory for all log files (default: ./logs)
"""

from dotenv import load_dotenv
load_dotenv()

import logging
import os
import atexit
from pathlib import Path

LOG_DIR = os.environ.get("LOG_DIR", "./logs")

# Module-level logger — set by make_pipeline_logger or make_agent_logger,
# retrieved by get_logger(). Always holds the most recently created logger.
_logger: logging.Logger | None = None


# ---------------------------------------------------------------------------
# Core logger class
# ---------------------------------------------------------------------------

class SwebenchCompatLogger(logging.Logger):
    """
    Logger that satisfies swebench's BuildImageError (needs log_file attr)
    and guarantees handlers are flushed/closed on process exit.
    """
    def __init__(self, name: str, log_path: str, also_print: bool = True):
        super().__init__(name, logging.INFO)
        self.log_file = log_path

        # Console handler — always on so output is visible in terminal
        if also_print:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            self.addHandler(stream_handler)

        # File handler
        self._file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        self._file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        self.addHandler(self._file_handler)

        atexit.register(self._shutdown)

    def _shutdown(self):
        for handler in self.handlers:
            try:
                handler.flush()
                handler.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def make_pipeline_logger(run_id: str) -> SwebenchCompatLogger:
    """
    Create the top-level pipeline logger. Call once in the orchestrator.

    Writes to: {LOG_DIR}/{run_id}/pipeline.log

    Also sets the module-level _logger so get_logger() works in any module
    that hasn't yet created its own agent logger.
    """
    global _logger

    log_dir = Path(LOG_DIR) / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = str(log_dir / "pipeline.log")

    logger = SwebenchCompatLogger("pipeline", log_path)
    logger.info(f"Pipeline logger initialized → {log_path}")
    print(f"Pipeline log: {log_path}")

    _logger = logger
    return logger


def make_agent_logger(
    run_id: str,
    instance_id: str,
    agent_name: str,
) -> SwebenchCompatLogger:
    """
    Create a per-instance per-agent logger.

    Writes to: {LOG_DIR}/{run_id}/{instance_id}/{agent_name}.log

    Also sets the module-level _logger so get_logger() returns this logger
    for the duration of this agent's run. The orchestrator should call
    make_agent_logger() before calling run_agent1/2/3(), and restore the
    pipeline logger afterwards if needed.

    Args:
        run_id      : the pipeline run ID (matches orchestrator RUN_ID)
        instance_id : the SWE-bench instance ID
        agent_name  : "agent1", "agent2", or "agent3"

    Returns:
        A SwebenchCompatLogger writing to the agent-specific log file.
    """
    global _logger

    log_dir = Path(LOG_DIR) / run_id / instance_id
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = str(log_dir / f"{agent_name}.log")

    logger = SwebenchCompatLogger(
        name=f"{agent_name}.{instance_id}",
        log_path=log_path,
        also_print=True,
    )
    logger.info(f"{agent_name} logger initialized for {instance_id} → {log_path}")

    _logger = logger
    return logger


def get_logger() -> logging.Logger:
    """
    Retrieve the current active logger.

    Returns the most recently created logger (pipeline or agent).
    Falls back to a stdout-only logger if none has been created yet
    (e.g. when running an agent module standalone for testing).
    """
    global _logger
    if _logger is None:
        fallback = logging.getLogger("swebench-fallback")
        if not fallback.handlers:
            fallback.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            fallback.addHandler(handler)
        return fallback
    return _logger
