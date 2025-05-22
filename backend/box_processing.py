import cv2
import numpy as np
import asyncio

async def extract_and_align_symbols(original_image, boxes, padding=3, crop_margin=5):
    """Выравнивание символов"""
    if len(original_image.shape) == 2:
        channels = 1
    else:
        channels = original_image.shape[2]

    max_height = max(y2 - y1 for x1, y1, x2, y2 in boxes)
    total_width = sum(x2 - x1 for x1, y1, x2, y2 in boxes) + (len(boxes) - 1) * padding

    if channels == 1:
        new_image = np.ones((max_height, total_width), dtype=np.uint8) * 255
    else:
        new_image = np.ones((max_height, total_width, 3), dtype=np.uint8) * 255

    current_x_position = 0

    for box in boxes:
        x1, y1, x2, y2 = box
        symbol_region = original_image[y1:y2, x1:x2]

        if channels == 3 and len(symbol_region.shape) == 2:
            symbol_region = await asyncio.to_thread(
                cv2.cvtColor, symbol_region, cv2.COLOR_GRAY2BGR
            )

        symbol_height = y2 - y1
        new_y1 = max_height - symbol_height
        new_y2 = max_height

        new_image[new_y1:new_y2, current_x_position : current_x_position + (x2 - x1)] = symbol_region
        current_x_position += (x2 - x1) + padding

    if channels == 1:
        final_image = np.ones((original_image.shape[0], total_width), dtype=np.uint8) * 255
    else:
        final_image = np.ones((original_image.shape[0], total_width, 3), dtype=np.uint8) * 255

    vertical_center = (final_image.shape[0] - new_image.shape[0]) // 2
    horizontal_center = (final_image.shape[1] - new_image.shape[1]) // 2

    vertical_end = min(vertical_center + new_image.shape[0], final_image.shape[0])
    horizontal_end = min(horizontal_center + new_image.shape[1], final_image.shape[1])

    actual_height = vertical_end - vertical_center
    actual_width = horizontal_end - horizontal_center

    final_image[vertical_center:vertical_end, horizontal_center:horizontal_end] = (
        new_image[:actual_height, :actual_width]
    )

    if channels == 1:
        gray = final_image
    else:
        gray = await asyncio.to_thread(cv2.cvtColor, final_image, cv2.COLOR_BGR2GRAY)

    coords = await asyncio.to_thread(cv2.findNonZero, 255 - gray)
    if coords is not None: 
        x, y, w, h = await asyncio.to_thread(cv2.boundingRect, coords)

        x = max(0, x - crop_margin)
        y = max(0, y - crop_margin)
        w = min(final_image.shape[1] - x, w + 2 * crop_margin)
        h = min(final_image.shape[0] - y, h + 2 * crop_margin)

        cropped_final_image = final_image[y : y + h, x : x + w]
    else:
        cropped_final_image = final_image  

    return cropped_final_image

async def correct_box(image, box, padding=5):
    """Уточнение границ бокса"""
    x1, y1, x2, y2 = box
    cropped = image[y1:y2, x1:x2]

    gray = await asyncio.to_thread(cv2.cvtColor, cropped, cv2.COLOR_BGR2GRAY)
    _, binary = await asyncio.to_thread(
        cv2.threshold, gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = await asyncio.to_thread(
        cv2.findContours, binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return box 

    cnt = max(contours, key=lambda c: cv2.contourArea(c))
    x, y, w, h = await asyncio.to_thread(cv2.boundingRect, cnt)

    new_x1 = max(x1 + x - padding, 0)
    new_y1 = max(y1 + y - padding, 0)
    new_x2 = min(x1 + x + w + padding, image.shape[1])
    new_y2 = min(y1 + y + h + padding, image.shape[0])

    return (new_x1, new_y1, new_x2, new_y2)

async def trim_box(image, box, white_background=True):
    """Асинхронный тримминг изображения внутри рамки"""
    x1, y1, x2, y2 = box

    height, width = image.shape[:2]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))

    if x1 >= x2 or y1 >= y2:
        return box

    cropped = image[y1:y2, x1:x2]

    if cropped.size == 0:
        return box

    try:
        if len(cropped.shape) == 3:
            gray = await asyncio.to_thread(cv2.cvtColor, cropped, cv2.COLOR_BGR2GRAY)
        else:
            gray = cropped.copy()
    except Exception:
        return box

    if white_background:
        mask = gray < 255
    else:
        mask = gray > 0

    coords = np.argwhere(mask)
    if len(coords) == 0:
        return box

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    x_min = max(0, x_min - 1)
    y_min = max(0, y_min - 1)
    x_max = min(gray.shape[1] - 1, x_max + 1)
    y_max = min(gray.shape[0] - 1, y_max + 1)

    new_x1 = x1 + x_min
    new_y1 = y1 + y_min
    new_x2 = x1 + x_max
    new_y2 = y1 + y_max

    if new_x1 >= new_x2 or new_y1 >= new_y2:
        return box

    return [new_x1, new_y1, new_x2, new_y2]