import os
import subprocess
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv(override=True)

# Get teh Github token
cr_pat = os.getenv("CR_PAT")
if not cr_pat:
    raise ValueError("CR_PAT is not set.")

cr_user = os.getenv("CR_USER")
if not cr_user:
    raise ValueError("CR_USER is not set")

# Define the version for the Docker Image
version = os.getenv("AGENT_GRID_VERSION", "1.1.2")
print(f"Building Agent Grid container version {version}")

# Login to Github Container Registry using podman
login_command = f"echo {cr_pat} | podman login ghcr.io -u {cr_user} --password-stdin"
subprocess.run(login_command, shell=True, check=True)

# Build the image with podman
build_command = f"podman build -t ghcr.io/reenvision-ai/agent-grid:{version} ."
subprocess.run(build_command, shell=True, check=True)

# Push the image to the registry
push_command = f"podman push ghcr.io/reenvision-ai/agent-grid:{version}"
subprocess.run(push_command, shell=True, check=True)

if version != "latest" and "beta" not in version:

    tag_command = f"podman tag ghcr.io/reenvision-ai/agent-grid:{version} ghcr.io/reenvision-ai/agent-grid:latest"
    subprocess.run(tag_command, shell=True, check=True)

    push_command = f"podman push ghcr.io/reenvision-ai/agent-grid:latest"
    subprocess.run(push_command, shell=True, check=True)