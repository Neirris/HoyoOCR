import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
from tqdm import tqdm
from datetime import datetime
import concurrent.futures
import yaml
import argparse
import image_processing as ImageProcessor


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--font", type=str, required=True)
    parser.add_argument("--background_dir", type=str, default="assets/background")
    parser.add_argument("--corpus_path", type=str, default="assets/misc/corpus.txt")
    parser.add_argument("--output_dir", type=str, default="datasets")
    parser.add_argument("--font_sizes", type=int, nargs="+", default=[13, 24, 40])
    parser.add_argument("--image_height", type=int, default=40)
    parser.add_argument("--val_split", type=float, default=0.2)
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


def apply_random_effects(image):
    image = ImageProcessor.ensure_grayscale(image)
    effect = random.choice(
        [
            ImageProcessor.apply_gaussian_noise,
            ImageProcessor.apply_salt_pepper_noise,
            ImageProcessor.apply_random_blur,
        ]
    )
    image, _ = effect(image)

    other_effect = random.choice(
        [
            ImageProcessor.apply_clahe,
            ImageProcessor.sharpen_image,
            ImageProcessor.denoise_bilateral,
            ImageProcessor.denoise_nlmeans,
        ]
    )
    image, _ = other_effect(image)

    return image


def apply_mask(image, boxes):
    if not boxes:
        return image

    mask = np.ones_like(image) * 255
    for box in boxes:
        x1, y1, x2, y2 = box
        if 0 <= x1 < x2 <= image.shape[1] and 0 <= y1 < y2 <= image.shape[0]:
            mask[y1:y2, x1:x2] = image[y1:y2, x1:x2]
    return mask


class DatasetGenerator:
    def __init__(self, args):
        self.args = args
        self.classes = ["text"]

    def get_random_background(self, width, height):
        bg_file = random.choice(os.listdir(self.args.background_dir))
        bg_path = os.path.join(self.args.background_dir, bg_file)
        bg = cv2.imread(bg_path)
        bg = cv2.resize(bg, (width, height))
        return bg

    def get_char_bbox(self, image, char, threshold=200):
        np_img = np.array(image)
        if len(np_img.shape) == 3:
            np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)

        _, binary = cv2.threshold(np_img, threshold, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return (0, 0, image.width, image.height)

        x_min, y_min = image.width, image.height
        x_max, y_max = 0, 0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)

        padding = 1
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(image.width, x_max + padding)
        y_max = min(image.height, y_max + padding)

        return (x_min, y_min, x_max, y_max)

    def is_monochrome_box(self, image, box, threshold=0.95):
        x1, y1, x2, y2 = box
        box_img = image[y1:y2, x1:x2]

        if box_img.size == 0:
            return True

        if len(box_img.shape) == 3:
            box_img = box_img[:, :, 0]

        white_pixels = np.sum(box_img > 250) / box_img.size
        black_pixels = np.sum(box_img < 5) / box_img.size

        return white_pixels > threshold or black_pixels > threshold

    def create_text_image(self, text, font_size, angle_mode="straight"):
        font = ImageFont.truetype(self.args.font_path, font_size)
        inter_char_space = (
            [random.randint(0, 20) for _ in text]
            if angle_mode != "plain"
            else [0] * len(text)
        )

        bboxes = [font.getbbox(c) for c in text]
        widths = [bbox[2] - bbox[0] for bbox in bboxes]
        heights = [bbox[3] - bbox[1] for bbox in bboxes]

        total_width = sum(widths) + (
            sum(inter_char_space) if angle_mode != "plain" else 0
        )
        max_height = max(heights) if heights else self.args.image_height
        img_height = max(self.args.image_height, max_height)

        canvas = Image.new("L", (total_width + 20, img_height + 20), color=255)
        draw = ImageDraw.Draw(canvas)

        x = 10
        bbox_all = [canvas.width, canvas.height, 0, 0]
        char_boxes = []

        for i, char in enumerate(text):
            char_bbox = bboxes[i]
            _w, h = widths[i], heights[i]

            y_offset = 0
            if angle_mode == "curve_up":
                y_offset = int((i / len(text)) * -10)
            elif angle_mode == "curve_down":
                y_offset = int((i / len(text)) * 10)

            y = (img_height - h) // 2 + y_offset + 10
            draw.text((x, y), char, font=font, fill=0)

            box = (
                x + char_bbox[0],
                y + char_bbox[1],
                x + char_bbox[2],
                y + char_bbox[3],
            )
            char_boxes.append(box)

            bbox_all[0] = min(bbox_all[0], box[0])
            bbox_all[1] = min(bbox_all[1], box[1])
            bbox_all[2] = max(bbox_all[2], box[2])
            bbox_all[3] = max(bbox_all[3], box[3])

            x += widths[i] + inter_char_space[i]

        mask = Image.new("L", canvas.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        for box in char_boxes:
            mask_draw.rectangle(box, fill=255)

        canvas.putalpha(mask)
        cropped = canvas.crop(bbox_all).convert("L")

        shifted_boxes = [
            (x1 - bbox_all[0], y1 - bbox_all[1], x2 - bbox_all[0], y2 - bbox_all[1])
            for (x1, y1, x2, y2) in char_boxes
        ]

        return cropped, shifted_boxes

    def convert_to_yolo_bbox(self, box, img_width, img_height):
        x1, y1, x2, y2 = box
        x_center = ((x1 + x2) / 2) / img_width
        y_center = ((y1 + y2) / 2) / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        return [x_center, y_center, width, height]

    def save_data(self, image, text, boxes, base_name, is_val=False):
        # Tesseract
        img_path = os.path.join(self.args.tesseract_dir, f"{base_name}.png")
        image.save(img_path)

        with open(
            os.path.join(self.args.tesseract_dir, f"{base_name}.gt.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(text)

        if boxes:
            self._create_box_file(
                base_name, text, boxes, self.args.tesseract_dir, image.height
            )

        # YOLO
        img_folder = "val" if is_val else "train"
        label_folder = "val" if is_val else "train"
        img_path = os.path.join(
            self.args.yolo_dir, "images", img_folder, f"{base_name}.png"
        )
        image.save(img_path)

        if boxes:
            label_path = os.path.join(
                self.args.yolo_dir, "labels", label_folder, f"{base_name}.txt"
            )
            yolo_boxes = [
                self.convert_to_yolo_bbox(box, image.width, image.height)
                for box in boxes
            ]

            with open(label_path, "w") as f:
                for box in yolo_boxes:
                    f.write(f"0 {' '.join(map(str, box))}\n")

    def _create_box_file(self, base_name, text, boxes, output_dir, image_height):
        box_path = os.path.join(output_dir, f"{base_name}.box")
        with open(box_path, "w", encoding="utf-8") as f:
            for char, box in zip(text, boxes):
                x1, y1, x2, y2 = box
                f.write(f"{char} {x1} {image_height - y2} {x2} {image_height - y1} 0\n")
            f.write("\t 0 0 0 0 0\n")

    def apply_processing_pipeline(self, image, boxes=None):
        processed = apply_random_effects(np.array(image))
        processed = apply_mask(processed, boxes)

        if boxes and all(self.is_monochrome_box(processed, box) for box in boxes):
            return None

        return Image.fromarray(processed)

    def generate_variants(self, image, text, boxes, base_name, is_val):
        processing_functions = [
            ImageProcessor.adaptive_binarization,
            ImageProcessor.otsu_binarization,
        ]

        for process_func in processing_functions:
            bin_img, suffix = process_func(image.copy())
            bin_img = apply_mask(bin_img, boxes)

            if not boxes or not all(
                self.is_monochrome_box(bin_img, box) for box in boxes
            ):
                self.save_data(
                    Image.fromarray(bin_img),
                    text,
                    boxes,
                    f"{base_name}_{suffix}",
                    is_val,
                )

        if len(text) == 1:
            morph_functions = [
                ImageProcessor.image_erosion,
                ImageProcessor.image_dilation,
            ]

            for morph_func in morph_functions:
                morph_img, morph_name = morph_func(image.copy())

                for process_func in processing_functions:
                    bin_img, suffix = process_func(morph_img)
                    bin_img = apply_mask(bin_img, boxes)

                    if not boxes or not all(
                        self.is_monochrome_box(bin_img, box) for box in boxes
                    ):
                        self.save_data(
                            Image.fromarray(bin_img),
                            text,
                            boxes,
                            f"{base_name}_{morph_name}_{suffix}",
                            is_val,
                        )

    def generate_single_char_rotations(self, char, counter, size):
        font = ImageFont.truetype(self.args.font_path, size)
        bbox = font.getbbox(char)
        width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]

        for angle in range(0, 360, 15):
            img = Image.new("L", (width * 3, height * 3), color=255)
            draw = ImageDraw.Draw(img)

            x, y = (img.width - width) // 2, (img.height - height) // 2
            draw.text((x, y), char, font=font, fill=0)

            rotated = img.rotate(angle, expand=True, fillcolor=255)
            char_bbox = self.get_char_bbox(rotated, char)

            if char_bbox:
                rotated = rotated.crop(char_bbox)
                is_val = random.random() < self.args.val_split
                self.process_and_save(
                    rotated.convert("RGB"),
                    char,
                    [(0, 0, rotated.width, rotated.height)],
                    f"{char}_{counter}_{size}_{angle}",
                    is_val,
                )

    def process_and_save(self, img, text, boxes, base_name, is_val):
        processed = self.apply_processing_pipeline(img, boxes)
        if processed is None:
            return

        self.save_data(processed, text, boxes, base_name, is_val)
        self.generate_variants(np.array(processed), text, boxes, base_name, is_val)

    def generate_image_for_line(self, line, size, mode, counter, pbar):
        img, boxes = self.create_text_image(line, size, angle_mode=mode)
        bg = self.get_random_background(img.width, img.height)
        combined = cv2.addWeighted(
            ImageProcessor.ensure_grayscale(bg), 0.5, np.array(img), 0.5, 0
        )

        is_val = random.random() < self.args.val_split
        self.process_and_save(
            Image.fromarray(combined).convert("RGB"),
            line,
            boxes,
            f"sample_{counter}_{size}_{mode}",
            is_val,
        )

        plain_img, plain_boxes = self.create_text_image(line, size, angle_mode="plain")
        self.process_and_save(
            plain_img.convert("RGB"),
            line,
            plain_boxes,
            f"sample_{counter}_{size}_plain",
            is_val,
        )

        if len(line) == 1:
            self.generate_single_char_rotations(line, counter, size)

        pbar.update(1)

    def generate_all(self):
        with open(self.args.corpus_path, "r", encoding="utf-8") as f:
            lines = [line.strip().lower() for line in f if line.strip()]

        total_images = len(lines) * len(self.args.font_sizes)

        with tqdm(total=total_images, desc="Generating dataset", ncols=100) as pbar:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for counter, line in enumerate(lines):
                    for size in self.args.font_sizes:
                        mode = random.choice(["straight", "curve_up", "curve_down"])
                        futures.append(
                            executor.submit(
                                self.generate_image_for_line,
                                line,
                                size,
                                mode,
                                counter,
                                pbar,
                            )
                        )
                concurrent.futures.wait(futures)


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
