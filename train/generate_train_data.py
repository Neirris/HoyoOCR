import os
from datetime import datetime
import yaml
import argparse
from image_generator import DatasetGenerator


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--font", type=str, required=True)
    parser.add_argument("--background_dir", type=str, default="assets/background")
    parser.add_argument("--corpus_path", type=str, default="assets/misc/corpus.txt")
    parser.add_argument("--output_dir", type=str, default="datasets")
    parser.add_argument("--font_sizes", type=int, nargs="+", default=[13, 24, 40])
    parser.add_argument("--image_height", type=int, default=40)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--mode", type=str, default="random")
    return parser.parse_args()


def setup_directories(args):
    font_name = args.font
    font_extensions = [".otf", ".ttf"]

    if not os.path.splitext(font_name)[1]:
        fonts_dir = "assets/fonts"
        for ext in font_extensions:
            font_path = os.path.join(fonts_dir, font_name + ext)
            if os.path.exists(font_path):
                args.font_path = font_path
                break
        else:
            raise FileNotFoundError(f"Font file not found: {font_name}")
    else:
        args.font_path = os.path.join("assets/fonts", font_name)
        if not os.path.exists(args.font_path):
            raise FileNotFoundError(f"Font file not found: {args.font_path}")

    if not os.path.exists(args.background_dir):
        raise FileNotFoundError(
            f"Background directory not found: {args.background_dir}"
        )

    if not os.path.exists(args.corpus_path):
        raise FileNotFoundError(f"Corpus file not found: {args.corpus_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    font_basename = os.path.splitext(os.path.basename(args.font_path))[0]
    base_output = os.path.join(args.output_dir, f"{font_basename}_{timestamp}")

    args.tesseract_dir = os.path.join(
        base_output, "tesseract", f"{font_basename}-ground-truth"
    )
    args.yolo_dir = os.path.join(base_output, "yolo")

    yolo_subdirs = [
        os.path.join(args.yolo_dir, "images", "train"),
        os.path.join(args.yolo_dir, "images", "val"),
        os.path.join(args.yolo_dir, "labels", "train"),
        os.path.join(args.yolo_dir, "labels", "val"),
    ]

    for directory in [args.tesseract_dir] + yolo_subdirs:
        os.makedirs(directory, exist_ok=True)

    yolo_config = {
        "path": os.path.abspath(args.yolo_dir),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "text"},
    }

    with open(os.path.join(args.yolo_dir, "dataset.yaml"), "w") as f:
        yaml.dump(yolo_config, f)

    return args


def main():
    try:
        args = parse_arguments()
        args = setup_directories(args)
        generator = DatasetGenerator(args)
        generator.generate_all()
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
