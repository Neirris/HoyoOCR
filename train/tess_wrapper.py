import argparse
import subprocess
from pathlib import Path, PurePosixPath


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)

    parser.add_argument("--max_iterations", type=int, default=100000)
    parser.add_argument("--learning_rate", type=float, default=0.002)

    return parser.parse_args()


def validate_paths(data_dir):
    base_name = data_dir.split("_")[0]

    datasets_dir = Path("datasets") / data_dir / "tesseract"
    ground_truth_dir = datasets_dir / f"{base_name}-ground-truth"

    if not datasets_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {datasets_dir}")
    if not ground_truth_dir.exists():
        raise FileNotFoundError(f"Ground truth directory not found: {ground_truth_dir}")

    rel_datasets_dir = PurePosixPath("..") / "datasets" / data_dir / "tesseract"
    rel_ground_truth_dir = rel_datasets_dir / f"{base_name}-ground-truth"

    return str(rel_datasets_dir).replace("\\", "/"), str(rel_ground_truth_dir).replace(
        "\\", "/"
    )


def run_make_command(args, datasets_dir, ground_truth_dir):
    make_command = [
        "make",
        "training",
        f"MODEL_NAME={args.model_name}",
        f"DATA_DIR={datasets_dir}",
        f"GROUND_TRUTH_DIR={ground_truth_dir}",
        f"MAX_ITERATIONS={args.max_iterations}",
        f"LEARNING_RATE={args.learning_rate}",
    ]

    try:
        print(f"Running command: {' '.join(make_command)}")
        subprocess.run(make_command, check=True, cwd="tesstrain", shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        raise


def main():
    try:
        args = parse_arguments()
        datasets_dir, ground_truth_dir = validate_paths(args.data_dir)

        run_make_command(args, datasets_dir, ground_truth_dir)

    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
