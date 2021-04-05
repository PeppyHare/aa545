from pathlib import Path


def create_folder(path):
    """Create a (possibly nested) folder if it does not already exist."""
    Path(path).mkdir(parents=True, exist_ok=True)
