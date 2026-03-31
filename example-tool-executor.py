"""
example_usage.py

Concrete example of how to use tool_executor.py with a real Docker container.

This is NOT one of the real agents — it's a simple demo that asks Claude
to explore the /testbed directory and report what it finds. Run this after
your orchestrator has started the container to verify the tool loop works
end-to-end before wiring up the real agents.

Usage:
    python example_usage.py
"""
from dotenv import load_dotenv
load_dotenv()
import json
from tool_executor import run_agent_loop, make_bash_tool
import docker


# ---------------------------------------------------------------------------
# Step 1: Get a handle to your already-running container
# ---------------------------------------------------------------------------
# In the real pipeline the orchestrator passes the container object directly.
# Here we fetch it by name so this script can run standalone.

CONTAINER_NAME = "sweb.eval.sqlfluff__sqlfluff-1625.agentloop"  # from orchestrator output

client = docker.from_env()
container = client.containers.get(CONTAINER_NAME)


# ---------------------------------------------------------------------------
# Step 2: Define the system prompt for this agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are a code exploration agent operating inside a Docker container.
The repository is located at /testbed.

You have access to a bash tool. Use it to explore the repository and
answer the user's question. Always read files before assuming their contents.
Run all commands as bash -lc commands — the environment is already initialized.
"""


# ---------------------------------------------------------------------------
# Step 3: Build the initial user message
# ---------------------------------------------------------------------------

INITIAL_MESSAGE = """
Please explore the /testbed directory and tell me:
1. What repository is this?
2. What is the top-level directory structure?
3. What Python version is available?
"""


# ---------------------------------------------------------------------------
# Step 4: Call run_agent_loop() with the bash tool bound to the container
# ---------------------------------------------------------------------------

result = run_agent_loop(
    instance_id=CONTAINER_NAME,      # used for log prefixes
    system_prompt=SYSTEM_PROMPT,
    initial_message=INITIAL_MESSAGE,
    tools=[make_bash_tool(container)],  # bind container to the bash handler
    verbose=True,                    # print iteration/tool logs to stdout
)


# ---------------------------------------------------------------------------
# Step 5: Inspect the result
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("AGENT RESULT")
print("=" * 60)
print(f"success    : {result.success}")
print(f"iterations : {result.iterations}")
print(f"\nfinal_response:\n{result.final_response}")

# The full message history is available for debugging
print(f"\nmessages   : {len(result.messages)} turns")
for i, msg in enumerate(result.messages):
    role = msg["role"]
    content = msg["content"]
    if isinstance(content, str):
        preview = content[:80].replace("\n", " ")
    elif isinstance(content, list):
        types = [b.get("type") if isinstance(b, dict) else type(b).__name__ for b in content]
        preview = f"[{', '.join(types)}]"
    else:
        preview = repr(content)[:80]
    print(f"  [{i}] {role}: {preview}")