import argparse
from ultralytics import YOLO
import os


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Обязательные аргументы
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)

    parser.add_argument("--model", type=str, default="assets/misc/yolo11n.pt")

    # Параметры обучения
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--patience", type=int, default=10)
    return parser.parse_args()


def validate_paths(args):
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")

    data_dir = os.path.join("datasets", args.data)
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    dataset_yaml = os.path.join(data_dir, "yolo", "dataset.yaml")
    if not os.path.exists(dataset_yaml):
        raise FileNotFoundError(f"Dataset config not found: {dataset_yaml}")

    save_dir = os.path.join(data_dir, "yolo", "runs")
    os.makedirs(save_dir, exist_ok=True)

    return dataset_yaml, save_dir


def main():
    try:
        args = parse_arguments()
        dataset_yaml, save_dir = validate_paths(args)

        model = YOLO(args.model)

        model.train(
            data=dataset_yaml,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            name=args.name,
            optimizer=args.optimizer,
            lr0=args.lr0,
            lrf=args.lrf,
            warmup_epochs=args.warmup_epochs,
            device=args.device,
            pretrained=args.pretrained,
            patience=args.patience,
            project=save_dir,
        )

    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
