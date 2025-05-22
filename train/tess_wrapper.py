import argparse
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--max_iterations", type=int, default=300000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.002, )
    parser.add_argument("--target_error_rate", type=float, default=0.05, )
    parser.add_argument("--ratio_train", type=float, default=0.8,)
    parser.add_argument("--plot", action="store_true", default=True, )
    parser.add_argument("--plot_only", action="store_true")

    args = parser.parse_args()

    if not args.plot_only and not args.model_name:
        parser.error("--model_name is required unless --plot_only is used")

    return args


def validate_paths(data_dir, model_name=None):
    base_name = Path(data_dir).name.split("_")[0]
    datasets_dir = (Path("datasets") / data_dir / "tesseract").absolute()
    ground_truth_dir = datasets_dir / f"{base_name}-ground-truth"

    if not datasets_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {datasets_dir}")
    if not ground_truth_dir.exists():
        raise FileNotFoundError(f"Ground truth directory not found: {ground_truth_dir}")

    effective_model_name = model_name if model_name else base_name

    return str(datasets_dir), str(ground_truth_dir), effective_model_name


def plot_cer_fallback(log_file, output_dir, model_name):
    try:
        iterations = []
        cer_values = []
        with open(log_file, 'r') as f:
            for line in f:
                if "At iteration" in line and "BCER train" in line:
                    parts = line.split(',')
                    iter_part = parts[0].strip()
                    iter_num = int(iter_part.split()[2].split('/')[0])
                    
                    bcer_part = [p for p in parts if "BCER train" in p][0]
                    bcer_value = float(bcer_part.split('=')[1].replace('%', '').strip())
                    
                    iterations.append(iter_num)
                    cer_values.append(bcer_value)
        
        if not cer_values:
            print("No CER values found in training.log. Cannot generate fallback plot.")
            return

        plt.figure(figsize=(12, 6))
        plt.plot(iterations, cer_values, 'b-', label='CER (training)')
        
        ax = plt.gca()
        
        locs = ax.get_xticks()
        new_labels = [f'{int(x/1000)}K' if x != 0 else '0' for x in locs]
        ax.set_xticks(locs)
        ax.set_xticklabels(new_labels)
        
        plt.xlabel('Iteration')
        plt.ylabel('Character Error Rate (%)')
        plt.title(f'Training CER for {model_name}')
        plt.legend()
        plt.grid(True)
        
        plt.autoscale(enable=True, axis='both', tight=True)
        
        output_file = output_dir / f'{model_name}.plot_cer.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Fallback plot saved to: {output_file}")
        
    except Exception as e:
        print(f"Error in fallback plot generation: {e}")
        print(f"Error details: {str(e)}")
        if 'iter_num' in locals():
            print(f"Last processed iteration: {iter_num}")
        if 'bcer_value' in locals():
            print(f"Last processed CER value: {bcer_value}")


def run_make_command(args, datasets_dir, ground_truth_dir):
    make_command = [
        "make",
        "training",
        f"MODEL_NAME={args.model_name}",
        f"DATA_DIR={datasets_dir}",
        f"GROUND_TRUTH_DIR={ground_truth_dir}",
        f"LEARNING_RATE={args.learning_rate}",
        f"TARGET_ERROR_RATE={args.target_error_rate}",
        f"RATIO_TRAIN={args.ratio_train}",
    ]

    if args.epochs > 0:
        make_command.append(f"EPOCHS={args.epochs}")
    else:
        make_command.append(f"MAX_ITERATIONS={args.max_iterations}")

    try:
        print(f"Running command: {' '.join(make_command)}")
        subprocess.run(make_command, check=True, cwd="tesstrain", shell=True)

        if args.plot:
            log_file = Path(datasets_dir) / args.model_name / "training.log"
            print(f"Checking for training.log at: {log_file}")
            if not log_file.exists():
                print(f"Warning: Training log file {log_file} not found. Skipping plot generation.")
                return

            eval_log_dir = Path(datasets_dir) / args.model_name / "eval"
            eval_log_dir.mkdir(parents=True, exist_ok=True)

            plot_command = [
                "make",
                "plot",
                f"MODEL_NAME={args.model_name}",
                f"DATA_DIR={datasets_dir}",
                f"GROUND_TRUTH_DIR={ground_truth_dir}",
            ]
            output_dir = Path(datasets_dir) / args.model_name
            print(f"Running plot command: {' '.join(plot_command)}")
            print(f"Expected PNG files: {output_dir / f'{args.model_name}.plot_cer.png'}, {output_dir / f'{args.model_name}.plot_log.png'}")
            try:
                subprocess.run(plot_command, check=True, cwd="tesstrain", shell=True)
            except subprocess.CalledProcessError as e:
                print(f"Warning: Failed to generate plots with make: {e}")
                print("Attempting fallback plot generation...")
                plot_cer_fallback(log_file, output_dir, args.model_name)
                if log_file.exists():
                    with open(log_file, "r") as f:
                        print(f"Training log contents (last 10 lines):\n{'-'*50}\n{''.join(f.readlines()[-10:])}\n{'-'*50}")
                eval_logs = list(eval_log_dir.glob(f"{args.model_name}_*.eval.log")) if eval_log_dir.exists() else []
                if eval_logs:
                    latest_log = max(eval_logs, key=lambda x: x.stat().st_mtime)
                    with open(latest_log, "r") as f:
                        print(f"Latest evaluation log ({latest_log}) contents:\n{'-'*50}\n{f.read()}\n{'-'*50}")
                else:
                    print(f"No evaluation logs found in {eval_log_dir}.")

    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        raise


def run_plot_only(args, datasets_dir, ground_truth_dir, model_name):
    log_file = Path(datasets_dir) / model_name / "training.log"
    print(f"Checking for training.log at: {log_file}")
    if not log_file.exists():
        print(f"Error: Training log file {log_file} not found. Cannot generate plots.")
        return

    eval_log_dir = Path(datasets_dir) / model_name / "eval"
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    plot_command = [
        "make",
        "plot",
        f"MODEL_NAME={model_name}",
        f"DATA_DIR={datasets_dir}",
        f"GROUND_TRUTH_DIR={ground_truth_dir}",
    ]
    output_dir = Path(datasets_dir) / model_name
    print(f"Running plot command: {' '.join(plot_command)}")
    print(f"Expected PNG files: {output_dir / f'{model_name}.plot_cer.png'}, {output_dir / f'{model_name}.plot_log.png'}")
    try:
        subprocess.run(plot_command, check=True, cwd="tesstrain", shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to generate plots with make: {e}")
        print("Attempting fallback plot generation...")
        plot_cer_fallback(log_file, output_dir, model_name)
        if log_file.exists():
            with open(log_file, "r") as f:
                print(f"Training log contents (last 10 lines):\n{'-'*50}\n{''.join(f.readlines()[-10:])}\n{'-'*50}")
        eval_logs = list(eval_log_dir.glob(f"{model_name}_*.eval.log")) if eval_log_dir.exists() else []
        if eval_logs:
            latest_log = max(eval_logs, key=lambda x: x.stat().st_mtime)
            with open(latest_log, "r") as f:
                print(f"Latest evaluation log ({latest_log}) contents:\n{'-'*50}\n{f.read()}\n{'-'*50}")
        else:
            print(f"No evaluation logs found in {eval_log_dir}.")


def main():
    try:
        print(f"Current working directory: {Path.cwd()}")
        args = parse_arguments()
        datasets_dir, ground_truth_dir, model_name = validate_paths(args.data_dir, args.model_name)

        if args.plot_only:
            run_plot_only(args, datasets_dir, ground_truth_dir, model_name)
        else:
            run_make_command(args, datasets_dir, ground_truth_dir)

    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()