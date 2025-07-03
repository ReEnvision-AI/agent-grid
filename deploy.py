import os
import subprocess
from dotenv import load_dotenv

def run_command(command, input_data=None):
    """Runs a command, prints its output, and checks for errors."""
    print(f"Running: {' '.join(command)}")
    try:
        kwargs = {
            "check": True,
            "text": True,
        }
        if input_data:
            kwargs["input"] = input_data
            kwargs["capture_output"] = True

        process = subprocess.run(command, **kwargs)

        if input_data:  # Only print if we captured it
            if process.stdout:
                print(process.stdout)
            if process.stderr:
                print(process.stderr)

    except subprocess.CalledProcessError as e:
        print(f"Error running command: {' '.join(command)}")
        # If output was captured, it's in e.stderr.
        if e.stderr:
            print(e.stderr)
        raise

def main():
    """Main function to build and push the Docker image."""
    # Load environment variables from the .env file
    load_dotenv(override=True)

    # Get the GitHub token
    cr_pat = os.getenv("CR_PAT")
    if not cr_pat:
        raise ValueError("CR_PAT is not set.")

    cr_user = os.getenv("CR_USER")
    if not cr_user:
        raise ValueError("CR_USER is not set")

    # Define the version for the Docker Image
    version = os.getenv("AGENT_GRID_VERSION", "1.1.2")
    print(f"Building Agent Grid container version {version}")

    image_name = "ghcr.io/reenvision-ai/agent-grid"
    image_with_version = f"{image_name}:{version}"
    latest_image = f"{image_name}:latest"

    # Login to Github Container Registry using podman
    login_command = ["podman", "login", "ghcr.io", "-u", cr_user, "--password-stdin"]
    run_command(login_command, input_data=cr_pat)

    # Build the image with podman
    build_command = ["podman", "build", "-t", image_with_version, "."]
    run_command(build_command)

    # Push the image to the registry
    push_command = ["podman", "push", image_with_version]
    run_command(push_command)

    if version != "latest" and "beta" not in version:
        # Tag the image as latest
        tag_command = ["podman", "tag", image_with_version, latest_image]
        run_command(tag_command)

        # Push the latest image
        push_latest_command = ["podman", "push", latest_image]
        run_command(push_latest_command)

if __name__ == "__main__":
    main()
