from pathlib import Path


def create_directory(path: Path) -> None:
    already_exists = path.exists()
    path.mkdir(parents=True, exist_ok=True)
    if not already_exists:
        print(path)


def create_empty_file(path: Path) -> None:
    if not path.exists():
        path.touch()
        print(path)


def main() -> None:
    root = Path.cwd()

    folders = [
        root / "data" / "raw",
        root / "src",
        root / "app",
        root / "outputs" / "checkpoints",
        root / "outputs" / "plots" / "gradcam",
        root / "outputs" / "results",
    ]

    files = [
        root / "src" / "__init__.py",
        root / "src" / "dataset.py",
        root / "src" / "model.py",
        root / "src" / "train.py",
        root / "src" / "evaluate.py",
        root / "src" / "gradcam.py",
        root / "src" / "utils.py",
        root / "app" / "__init__.py",
        root / "app" / "app.py",
        root / "config.yaml",
        root / "requirements.txt",
    ]

    for folder in folders:
        create_directory(folder)

    for file_path in files:
        create_empty_file(file_path)


if __name__ == "__main__":
    main()
