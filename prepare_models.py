import shutil
import tomllib
from pathlib import Path
import os
# Constants
MODELS_DIR = Path("models").resolve()              # Absolute path to local models directory
ENTRYPOINT = Path("entrypoint.sh").resolve()       # Path to entrypoint script to be generated
MODELS_TOML = Path("models.toml")                  # Config file defining model metadata

def load_config():
    """
    Load the TOML configuration file and return the parsed dictionary.
    """
    with open(MODELS_TOML, "rb") as f:
        return tomllib.load(f)

def prepare_custom(source_path: Path) -> Path:
    """
    Copy a file or directory from source_path into the models directory.
    Returns the new local path under MODELS_DIR.
    """
    source_path = source_path.expanduser().resolve()
    target = MODELS_DIR / source_path.name

    if target.exists() and source_path.samefile(target):
        print(f"⚠️  Warning: Source and destination are the same: {source_path}")
        return target

    if source_path.is_file():
        shutil.copy2(source_path, target)
    elif source_path.is_dir():
        shutil.copytree(source_path, target, dirs_exist_ok=True)
    else:
        raise FileNotFoundError(f"Model path not found: {source_path}")

    return target

def normalize_home_path(path_str: str) -> str:
    """
    Replace $HOME or ~ with /root (Docker home) for container paths.
    Absolute paths without $HOME or ~ are left untouched.
    """
    if "$HOME" in path_str or path_str.startswith("~"):
        expanded = os.path.expandvars(os.path.expanduser(path_str))
        return expanded.replace(str(Path.home()), "/root", 1)

    return path_str

def shell_quote(path: str) -> str:
    """
    Return a shell-quoted version of a path string for safe inclusion in bash commands.
    """
    return f'"{path}"'

def main():
    # Ensure models directory exists
    MODELS_DIR.mkdir(exist_ok=True)

    # Load the model configuration
    config = load_config()
    symlink_commands = []

    # Start entrypoint.sh with bash init and optional cleanup
    with open(ENTRYPOINT, "w") as f:
        f.write("#!/bin/bash\nset -e\n\n")
        f.write("echo 'Creating symlinks'\n")
    
    # Process each model defined in the config
    for name, model in config.items():
        if name == "default":
            continue

        model_type = model["type"]
        print(f"🔧 Processing '{name}' (type: {model_type})")

        if model_type in {"custom", "protege"}:
            # Get host and optional container paths
            path_str = model["path"]
            host_path = Path(os.path.expandvars(path_str)).expanduser().resolve()
            copied_path = prepare_custom(host_path)

            container_path = model.get("container_path", path_str) 
            container_path = normalize_home_path(container_path)

            # Resolve container-side symlink path
            container_target_path = Path(container_path)
            if not container_target_path.is_absolute():
                container_target_path = (Path("/app") / container_target_path).resolve()

            # Prepare symlink shell commands for Docker startup
            model_file_name = copied_path.name
            quote_model_file_name = shell_quote("/app/models/" + model_file_name)
            symlink_commands.append(
                f"mkdir -p {shell_quote(container_target_path.parent)}"
            )
            symlink_commands.append(
                f"ln -sf {quote_model_file_name} {shell_quote(container_target_path)}"
            )

        elif model_type == "hf":
            # Placeholder: HF models handled via environment variables
            pass

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    # Append symlink commands to entrypoint script
    with open(ENTRYPOINT, "a") as f:
        for cmd in symlink_commands:
            f.write(cmd + "\n")
        f.write("\n")
        f.write("echo '✅ Model symlinks setup complete'\n")
        f.write("# Execute the passed command\n")
        f.write('exec "$@"\n')

    ENTRYPOINT.chmod(0o755)  # Make entrypoint executable
    print("✅ Prepared models and wrote symlink logic to entrypoint.sh")

if __name__ == "__main__":
    main()
