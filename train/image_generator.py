import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import logging
from tqdm import tqdm
import concurrent.futures
import image_processing as ImageProcessor

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

class DatasetGenerator:
    def __init__(self, args):
        self.args = args
        self.classes = ["text"]
        self.rotation_angles = [-90, -60, -30, 0, 30, 60, 90]

    def apply_effects(self, image, mask, mode="random"):
        if len(image.shape) == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        elif len(image.shape) == 2:  # Оттенки серого
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        if mode == "random":
            effect = random.choice([
                ImageProcessor.apply_gaussian_noise,
                ImageProcessor.apply_salt_pepper_noise,
                ImageProcessor.apply_random_blur,
            ])
            image, _ = effect(image)
            
            other_effect = random.choice([
                ImageProcessor.apply_clahe,
                ImageProcessor.sharpen_image,
                ImageProcessor.denoise_bilateral,
                ImageProcessor.denoise_nlmeans,
            ])
            image, effect_name = other_effect(image)
        else:
            effects = [
                ImageProcessor.apply_gaussian_noise,
                ImageProcessor.apply_salt_pepper_noise,
                ImageProcessor.apply_random_blur,
            ]
            other_effects = [
                ImageProcessor.apply_clahe,
                ImageProcessor.sharpen_image,
                ImageProcessor.denoise_bilateral,
                ImageProcessor.denoise_nlmeans,
            ]
            for effect in effects:
                temp_img, _ = effect(image.copy())
                for other_effect in other_effects:
                    image, effect_name = other_effect(temp_img.copy())
        
        return image

    def get_random_background(self, width, height):
        bg_file = random.choice(os.listdir(self.args.background_dir))
        bg_path = os.path.join(self.args.background_dir, bg_file)
        bg = cv2.imread(bg_path)
        bg = cv2.resize(bg, (width, height))
        return bg

    def get_mask_bbox(self, mask, padding=10): 
        mask_np = np.array(mask)
        non_zero = np.where(mask_np > 0)
        if len(non_zero[0]) == 0:
            return (0, 0, mask_np.shape[1], mask_np.shape[0])
        y_min, y_max = non_zero[0].min(), non_zero[0].max() + 1
        x_min, x_max = non_zero[1].min(), non_zero[1].max() + 1
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(mask_np.shape[1], x_max + padding)
        y_max = min(mask_np.shape[0], y_max + padding)
        return (x_min, y_min, x_max, y_max)

    def trim_transparent_pixels(self, image, mask, boxes):
        image_np = np.array(image)
        mask_np = np.array(mask)
        
        if len(image_np.shape) == 2:  # Grayscale
            non_white = np.where(image_np < 255)
        elif len(image_np.shape) == 4:  # RGBA
            non_white = np.where((image_np[:, :, :3] < 255).any(axis=2) | (image_np[:, :, 3] > 0))
        else:  # RGB
            non_white = np.where((image_np < 255).any(axis=2))
        
        if len(non_white[0]) == 0:
            return image, mask, boxes, (0, 0, image_np.shape[1], image_np.shape[0])
        
        y_min, y_max = non_white[0].min(), non_white[0].max() + 1
        x_min, x_max = non_white[1].min(), non_white[1].max() + 1
        
        trimmed_image = image_np[y_min:y_max, x_min:x_max]
        trimmed_mask = mask_np[y_min:y_max, x_min:x_max]
        
        trimmed_boxes = [
            (
                max(0, x1 - x_min),
                max(0, y1 - y_min),
                min(x_max - x_min, x2 - x_min),
                min(y_max - y_min, y2 - y_min)
            ) for x1, y1, x2, y2 in boxes
        ]
        
        trimmed_boxes = [
            box for box in trimmed_boxes
            if box[2] > box[0] and box[3] > box[1]
        ]
        
        if not trimmed_boxes:
            return image, mask, boxes, (0, 0, image_np.shape[1], image_np.shape[0])
        
        return (
            Image.fromarray(trimmed_image),
            Image.fromarray(trimmed_mask),
            trimmed_boxes,
            (x_min, y_min, x_max, y_max)
        )

    def resize_image_and_boxes(self, image, boxes, mask, extra_size=0):
        if not boxes:
            return image, boxes, mask

        mask_np = np.array(mask)
        image_np = np.array(image)

        non_zero = np.where(mask_np > 0)
        if len(non_zero[0]) == 0:
            return image, boxes, mask

        y_min, y_max = non_zero[0].min(), non_zero[0].max()
        x_min, x_max = non_zero[1].min(), non_zero[1].max()

        cropped_image = image_np[y_min:y_max + 1, x_min:x_max + 1]
        cropped_mask = mask_np[y_min:y_max + 1, x_min:x_max + 1]

        shifted_boxes = [
            (
                max(0, x1 - x_min),
                max(0, y1 - y_min),
                min(x_max - x_min, x2 - x_min),
                min(y_max - y_min, y2 - y_min)
            ) for x1, y1, x2, y2 in boxes
        ]

        resized_image_pil = Image.fromarray(cropped_image).convert("RGB")
        resized_mask_pil = Image.fromarray(cropped_mask).convert("L")

        resized_image, resized_mask, scaled_boxes = ImageProcessor.resize_to_target_font_size(
            np.array(resized_image_pil), np.array(resized_mask_pil), shifted_boxes, target_height=40
        )
        resized_image_pil = Image.fromarray(resized_image).convert("RGB")
        resized_mask_pil = Image.fromarray(resized_mask).convert("L")

        trimmed_image, trimmed_mask, trimmed_boxes, _ = self.trim_transparent_pixels(
            resized_image_pil, resized_mask_pil, scaled_boxes
        )

        return trimmed_image, trimmed_boxes, trimmed_mask

    def check_mask_overlap(self, original_mask, processed_image, boxes, threshold=0.45):
        original_mask_np = np.array(original_mask)
        processed_image_np = np.array(processed_image)
        
        if original_mask_np.shape != processed_image_np.shape[:2]:
            return False, processed_image, boxes, original_mask
        
        result_image = np.full_like(processed_image_np, 255)
        
        for box in boxes:
            x1, y1, x2, y2 = box
            mask_roi = original_mask_np[y1:y2, x1:x2]
            processed_roi = processed_image_np[y1:y2, x1:x2]
            
            kernel = np.ones((3, 3), np.uint8)
            cleaned_mask = cv2.morphologyEx(mask_roi, cv2.MORPH_OPEN, kernel, iterations=1)
            cleaned_mask = np.where(cleaned_mask > 127, 255, 0).astype(np.uint8)
            
            _, processed_binary = cv2.threshold(processed_roi, 127, 255, cv2.THRESH_BINARY_INV)
            
            overlap = np.logical_and(cleaned_mask > 0, processed_binary > 0)
            overlap_area = np.sum(overlap)
            original_area = np.sum(cleaned_mask > 0)
            
            if original_area == 0:
                return False, processed_image, boxes, original_mask
            
            overlap_ratio = overlap_area / original_area
            if overlap_ratio < threshold:
                return False, processed_image, boxes, original_mask
            
            result_image[y1:y2, x1:x2] = np.where(cleaned_mask > 0, processed_roi, 255)
        
        trimmed_image, trimmed_mask, trimmed_boxes, _ = self.trim_transparent_pixels(
            Image.fromarray(result_image), original_mask, boxes
        )
        
        if not trimmed_boxes:
            return False, processed_image, boxes, original_mask
        
        return True, trimmed_image, trimmed_boxes, trimmed_mask

    def create_text_image(self, text, font_size, angle_mode="straight", extra_size=0):
        font = ImageFont.truetype(self.args.font_path, font_size)
        inter_char_space = [0] * len(text) if angle_mode == "plain" else [random.randint(0, 20) for _ in text]
        
        bboxes = [font.getbbox(c) for c in text]
        widths = [bbox[2] - bbox[0] + extra_size for bbox in bboxes]
        heights = [bbox[3] - bbox[1] + extra_size for bbox in bboxes]
        
        total_width = sum(widths) + sum(inter_char_space)
        max_height = max(heights) if heights else self.args.image_height
        img_height = max(self.args.image_height, max_height)
        
        canvas = Image.new("L", (total_width + 20, img_height + 20), color=255)
        mask = Image.new("L", canvas.size, 0)
        draw = ImageDraw.Draw(canvas)
        mask_draw = ImageDraw.Draw(mask)
        
        x, bbox_all = 10, [canvas.width, canvas.height, 0, 0]
        char_boxes = []
        
        for i, char in enumerate(text):
            char_bbox = bboxes[i]
            _w, h = widths[i], heights[i]
            
            if angle_mode == "curve":
                pass
            else:
                y = (img_height - h) // 2 + 10
            
            draw.text((x, y), char, font=font, fill=0)
            mask_draw.text((x, y), char, font=font, fill=255)
            
            box = (
                x + char_bbox[0],
                y + char_bbox[1],
                x + char_bbox[2] + extra_size,
                y + char_bbox[3] + extra_size
            )
            char_boxes.append(box)
            
            bbox_all[0] = min(bbox_all[0], box[0])
            bbox_all[1] = min(bbox_all[1], box[1])
            bbox_all[2] = max(bbox_all[2], box[2])
            bbox_all[3] = max(bbox_all[3], box[3])
            
            x += widths[i] + inter_char_space[i]
        
        canvas.putalpha(mask)
        cropped = canvas.crop(bbox_all).convert("L")
        cropped_mask = mask.crop(bbox_all)
        
        shifted_boxes = [
            (x1 - bbox_all[0], y1 - bbox_all[1], x2 - bbox_all[0], y2 - bbox_all[1])
            for (x1, y1, x2, y2) in char_boxes
        ]
        
        return cropped, shifted_boxes, cropped_mask

    def convert_to_yolo_bbox(self, box, img_width, img_height):
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            return None
        x_center = ((x1 + x2) / 2) / img_width
        y_center = ((y1 + y2) / 2) / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        return [x_center, y_center, width, height]

    def save_data(self, image, text, boxes, base_name, is_val=False):
        trimmed_image, trimmed_mask, trimmed_boxes, _ = self.trim_transparent_pixels(image, image, boxes)
        
        if trimmed_image.mode == "RGBA":
            trimmed_image = trimmed_image.convert("L")
        
        img_path = os.path.join(self.args.tesseract_dir, f"{base_name}.png")
        trimmed_image.save(img_path)
        
        with open(os.path.join(self.args.tesseract_dir, f"{base_name}.gt.txt"), "w", encoding="utf-8") as f:
            f.write(text)
        
        if trimmed_boxes:
            for box in trimmed_boxes:
                x1, y1, x2, y2 = box
                if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0 or x2 > trimmed_image.width or y2 > trimmed_image.height:
                    return
            self._create_box_file(base_name, text, trimmed_boxes, self.args.tesseract_dir, trimmed_image.height)
        
        img_folder = "val" if is_val else "train"
        label_folder = "val" if is_val else "train"
        img_path = os.path.join(self.args.yolo_dir, "images", img_folder, f"{base_name}.png")
        trimmed_image.save(img_path)
        
        if trimmed_boxes:
            label_path = os.path.join(self.args.yolo_dir, "labels", label_folder, f"{base_name}.txt")
            yolo_boxes = []
            for box in trimmed_boxes:
                yolo_box = self.convert_to_yolo_bbox(box, trimmed_image.width, trimmed_image.height)
                if yolo_box is not None:
                    yolo_boxes.append(yolo_box)
            if yolo_boxes:
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

    def apply_processing_pipeline(self, image, boxes, mask):
        processed = self.apply_effects(np.array(image), np.array(mask), mode=self.args.mode)
        processed_image = Image.fromarray(processed)
        return processed_image, mask, boxes

    def generate_variants(self, image, text, boxes, mask, base_name, is_val):
        processing_functions = [
            ImageProcessor.adaptive_binarization,
            ImageProcessor.otsu_binarization,
        ]
        
        for process_func in processing_functions:
            bin_img, suffix = process_func(np.array(image))
            bin_img_pil = Image.fromarray(bin_img)
            is_valid, cleaned_img, updated_boxes, updated_mask = self.check_mask_overlap(mask, bin_img_pil, boxes)
            if is_valid:
                self.save_data(cleaned_img, text, updated_boxes, f"{base_name}_{suffix}", is_val)
        
        if len(text) == 1:
            morph_functions = [
                ImageProcessor.image_erosion,
                ImageProcessor.image_dilation,
                lambda img: ImageProcessor.add_text_glow(img, np.array(mask)),
            ]
            for morph_func in morph_functions:
                morph_img, morph_name = morph_func(np.array(image))
                extra_size = 0
                if morph_name.startswith("Dilation_k") or morph_name.startswith("Glow_k"):
                    extra_size = int(morph_name.split("_k")[1])
                
                morph_img_pil = Image.fromarray(morph_img)
                resized_img, resized_boxes, resized_mask = self.resize_image_and_boxes(
                    morph_img_pil, boxes, mask, extra_size=extra_size
                )
                
                for process_func in processing_functions:
                    bin_img, suffix = process_func(np.array(resized_img))
                    bin_img_pil = Image.fromarray(bin_img)
                    is_valid, cleaned_img, updated_boxes, updated_mask = self.check_mask_overlap(resized_mask, bin_img_pil, resized_boxes)
                    if is_valid:
                        self.save_data(
                            cleaned_img,
                            text,
                            updated_boxes,
                            f"{base_name}_{morph_name}_{suffix}",
                            is_val
                        )

    def generate_text_char(self, char, counter, size, image, boxes, mask, extra_size=0):
        box = boxes[0]
        for angle in self.rotation_angles:
            mask_np = np.array(mask)
            kernel = np.ones((3, 3), np.uint8)
            expanded_mask = cv2.dilate(mask_np, kernel, iterations=2)
            expanded_mask_pil = Image.fromarray(expanded_mask).convert("L")
            
            rotated_img = image.rotate(angle, expand=True, fillcolor=255)
            rotated_mask = expanded_mask_pil.rotate(angle, expand=True, fillcolor=0)
            
            rotated_img, rotated_mask, trimmed_boxes, trim_bbox = self.trim_transparent_pixels(
                rotated_img, rotated_mask, [box]
            )
            x_min, y_min, x_max, y_max = trim_bbox
            
            x1, y1, x2, y2 = box
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            rad = np.radians(-angle)
            cos_a, sin_a = np.cos(rad), np.sin(rad)
            
            corners = [
                (x1, y1), (x2, y1), (x2, y2), (x1, y2)
            ]
            rotated_corners = []
            for x, y in corners:
                x_new = center_x + (x - center_x) * cos_a - (y - center_y) * sin_a
                y_new = center_y + (x - center_x) * sin_a + (y - center_y) * cos_a
                rotated_corners.append((x_new, y_new))
            
            x_coords, y_coords = zip(*rotated_corners)
            new_x1 = min(x_coords) - extra_size
            new_y1 = min(y_coords) - extra_size
            new_x2 = max(x_coords) + extra_size
            new_y2 = max(y_coords) + extra_size
            
            expanded_box = (
                max(0, new_x1 - x_min),
                max(0, new_y1 - y_min),
                min(rotated_img.width, new_x2 - x_min),
                min(rotated_img.height, new_y2 - y_min)
            )
            
            if expanded_box[2] <= expanded_box[0] or expanded_box[3] <= expanded_box[1]:
                continue
            
            scaled_img, scaled_boxes, scaled_mask = self.resize_image_and_boxes(
                rotated_img, [expanded_box], rotated_mask, extra_size=extra_size
            )
            
            is_val = random.random() < self.args.val_split
            self.process_and_save(
                scaled_img.convert("RGB"),
                char,
                scaled_boxes,
                scaled_mask,
                f"{char}_{counter}_{size}_{angle}",
                is_val
            )

    def generate_text_line(self, line, size, mode, counter, pbar, extra_size=0):
        try:
            if len(line) == 1:
                img, boxes, mask = self.create_text_image(line, size, "plain", extra_size)
                self.generate_text_char(line, counter, size, img, boxes, mask, extra_size)
            else:
                img, boxes, mask = self.create_text_image(line, size, mode, extra_size)
                
                bg = self.get_random_background(img.width, img.height)
                bg_pil = Image.fromarray(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)).convert("L")
                combined = Image.new("L", img.size, 255)
                combined.paste(bg_pil, (0, 0))
                combined.paste(img, (0, 0), mask)
                
                is_val = random.random() < self.args.val_split
                scaled_img, scaled_boxes, scaled_mask = self.resize_image_and_boxes(
                    combined.convert("RGB"), boxes, mask
                )
                self.process_and_save(
                    scaled_img,
                    line,
                    scaled_boxes,
                    scaled_mask,
                    f"sample_{counter}_{size}_{mode}",
                    is_val
                )
                
                plain_img, plain_boxes, plain_mask = self.create_text_image(line, size, "plain", extra_size)
                scaled_plain_img, scaled_plain_boxes, scaled_plain_mask = self.resize_image_and_boxes(
                    plain_img.convert("RGB"), plain_boxes, plain_mask
                )
                self.process_and_save(
                    scaled_plain_img,
                    line,
                    scaled_plain_boxes,
                    scaled_plain_mask,
                    f"sample_{counter}_{size}_plain",
                    is_val
                )
            
            pbar.update(1)
        except Exception as e:
            logging.error(f"Ошибка при обработке строки '{line}' с размером {size}, режимом {mode}, счётчиком {counter}: {str(e)}")
            raise

    def process_and_save(self, img, text, boxes, mask, base_name, is_val):
        processed_img, processed_mask, processed_boxes = self.apply_processing_pipeline(img, boxes, mask)
        self.generate_variants(processed_img, text, processed_boxes, processed_mask, base_name, is_val)

    def generate_all(self):
        try:
            with open(self.args.corpus_path, "r", encoding="utf-8") as f:
                lines = [line.strip().lower() for line in f if line.strip()]
            
            total_images = len(lines) * len(self.args.font_sizes)
            
            with tqdm(total=total_images, desc="Генерация датасета", ncols=100) as pbar:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = []
                    for counter, line in enumerate(lines):
                        for size in self.args.font_sizes:
                            mode = random.choice(["straight", "plain"])
                            futures.append(
                                executor.submit(
                                    self.generate_text_line,
                                    line,
                                    size,
                                    mode,
                                    counter,
                                    pbar,
                                    extra_size=0
                                )
                            )
                    concurrent.futures.wait(futures)
        except Exception as e:
            logging.error(f"Ошибка в generate_all: {str(e)}")
            raise