import cv2
import numpy as np
from ultralytics import YOLO
from image_processing import trim_box


def prepare_for_yolo(image):
    """Преобразует изображение для YOLO (3 канала)"""
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image

def detect_with_yolo(
    image,
    model_path="abyss.pt",
    conf_threshold=0.7,
    size_tolerance=0.1,
    small_overlap_threshold=0.2,
    white_background=True,
    missing_gap_threshold=1.5,
):
    """
    Детекция объектов YOLO с нормализацией размеров, устранением пересечений по X
    и вставкой пропущенных объектов при больших разрывах (множество MISSING).
    Возвращает:
    - аннотированное изображение
    - массив боксов в формате [x1, y1, x2, y2]
    - массив классов
    - массив скоров уверенности
    """
    model = YOLO(model_path)
    color_image = prepare_for_yolo(image)
    results = model(color_image)

    annotated_image = color_image.copy()
    boxes = []
    scores = []
    classes = []

    for result in results:
        for box in result.boxes:
            conf = float(box.conf)
            if conf < conf_threshold:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls)
            boxes.append([x1, y1, x2, y2])
            scores.append(conf)
            classes.append(cls_id)

    if not boxes:
        return annotated_image, np.array([]), np.array([]), np.array([])

    boxes = np.array(boxes)
    scores = np.array(scores)
    classes = np.array(classes)

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    median_width = np.median(widths)
    median_height = np.median(heights)

    # Нормализация размеров
    adjusted_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        new_x1 = int(center_x - median_width / 2)
        new_y1 = int(center_y - median_height / 2)
        new_x2 = int(center_x + median_width / 2)
        new_y2 = int(center_y + median_height / 2)
        adjusted_boxes.append([new_x1, new_y1, new_x2, new_y2])

    adjusted_boxes = np.array(adjusted_boxes)

    # Устраняем пересечения по горизонтали
    for i in range(len(adjusted_boxes)):
        for j in range(i + 1, len(adjusted_boxes)):
            box1 = adjusted_boxes[i]
            box2 = adjusted_boxes[j]
            ix1 = max(box1[0], box2[0])
            ix2 = min(box1[2], box2[2])
            if ix1 < ix2:
                iy1 = max(box1[1], box2[1])
                iy2 = min(box1[3], box2[3])
                if iy1 < iy2:
                    intersection_x = ix2 - ix1
                    min_width = min(box1[2] - box1[0], box2[2] - box2[0])
                    if intersection_x / min_width < small_overlap_threshold:
                        mid_x = (ix1 + ix2) // 2
                        if box1[0] < box2[0]:
                            adjusted_boxes[i][2] = mid_x
                            adjusted_boxes[j][0] = mid_x
                        else:
                            adjusted_boxes[j][2] = mid_x
                            adjusted_boxes[i][0] = mid_x

    indices = np.argsort(adjusted_boxes[:, 0])
    adjusted_boxes = adjusted_boxes[indices]
    classes = classes[indices]
    scores = scores[indices]

    new_boxes = []
    new_classes = []
    new_scores = []

    for i in range(len(adjusted_boxes) - 1):
        box1 = adjusted_boxes[i]
        box2 = adjusted_boxes[i + 1]

        new_boxes.append(box1)
        new_classes.append(classes[i])
        new_scores.append(scores[i])

        gap = box2[0] - box1[2]

        if gap > missing_gap_threshold * median_width:
            num_missing = int(round(gap / median_width)) - 1

            if num_missing > 0:
                step = gap / (num_missing + 1)

                for m in range(num_missing):
                    center_x = int(box1[2] + (m + 1) * step)
                    center_y = (box1[1] + box1[3]) // 2

                    new_x1 = int(center_x - median_width / 2)
                    new_y1 = int(center_y - median_height / 2)
                    new_x2 = int(center_x + median_width / 2)
                    new_y2 = int(center_y + median_height / 2)

                    new_boxes.append([new_x1, new_y1, new_x2, new_y2])
                    new_classes.append(-1)  # -1 для пропущенных
                    new_scores.append(0.0)

    new_boxes.append(adjusted_boxes[-1])
    new_classes.append(classes[-1])
    new_scores.append(scores[-1])

    new_boxes = np.array(new_boxes)
    new_classes = np.array(new_classes)
    new_scores = np.array(new_scores)

    trimmed_boxes = []
    for box in new_boxes:
        trimmed_box = trim_box(
            color_image if len(color_image.shape) == 3 else image, box, white_background
        )
        trimmed_boxes.append(trimmed_box)

    trimmed_boxes = np.array(trimmed_boxes)

    for box, cls, score in zip(trimmed_boxes, new_classes, new_scores):
        x1, y1, x2, y2 = box
        color = (0, 255, 0) if cls != -1 else (0, 0, 255)
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
        if cls != -1:
            # label = f"{model.names[cls]} {score:.2f}"
            label = f"{score:.2f}"
            cv2.putText(
                annotated_image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )
        else:
            cv2.putText(
                annotated_image,
                "Missing",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )

    return annotated_image, trimmed_boxes, new_classes, new_scores