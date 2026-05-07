import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clone huggingface/evaluate into the repository root for local metric loading."
    )
    parser.add_argument(
        "--repo-url",
        default="https://github.com/huggingface/evaluate.git",
        help="Git URL of the evaluate repository.",
    )
    parser.add_argument(
        "--target-dir",
        default=None,
        help="Target directory for the checkout. Defaults to <repo_root>/evaluate.",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Only clone the repository and skip 'pip install -e'.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    target_dir = Path(args.target_dir) if args.target_dir else repo_root / "evaluate"

    if target_dir.exists():
        print(f"evaluate checkout already exists at: {target_dir}")
    else:
        subprocess.run(
            ["git", "clone", args.repo_url, str(target_dir)],
            check=True,
        )

    if not args.skip_install:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", str(target_dir)],
            check=True,
        )

    print("evaluate setup complete.")
    print(f"Metric root: {target_dir}")


if __name__ == "__main__":
    main()
