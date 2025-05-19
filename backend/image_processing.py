import os
import cv2
import numpy as np
import random
import asyncio
import aiofiles

async def ensure_grayscale(image):
    """преобразование в оттенки серого"""
    if len(image.shape) == 3:
        return await asyncio.to_thread(cv2.cvtColor, image, cv2.COLOR_BGR2GRAY)
    return image


async def set_image_dpi(image, dpi=300):
    """установка DPI"""
    original_is_grayscale = len(image.shape) == 2
    image_to_save = (
        image if not original_is_grayscale 
        else await asyncio.to_thread(cv2.cvtColor, image, cv2.COLOR_GRAY2BGR)
    )

    async with aiofiles.tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_file:
        temp_path = tmp_file.name

    try:
        await asyncio.to_thread(
            cv2.imwrite,
            temp_path,
            image_to_save,
            [cv2.IMWRITE_TIFF_XDPI, dpi, cv2.IMWRITE_TIFF_YDPI, dpi],
        )
        result = await asyncio.to_thread(cv2.imread, temp_path)
        if original_is_grayscale:
            result = await ensure_grayscale(result)

        return result, f"DPI {dpi}"

    finally:
        await asyncio.to_thread(os.unlink, temp_path)


async def normalize_image(image):
    """нормализация изображения"""
    image = await ensure_grayscale(image)
    norm_img = np.zeros((image.shape[0], image.shape[1]))
    normalized = await asyncio.to_thread(
        cv2.normalize, image, norm_img, 0, 255, cv2.NORM_MINMAX
    )
    return normalized, "Normalization"

async def apply_clahe(image):
    """повышение контрастности"""
    image = await ensure_grayscale(image)
    clahe = await asyncio.to_thread(
        cv2.createCLAHE, clipLimit=3.0, tileGridSize=(8, 8)
    )
    enhanced = await asyncio.to_thread(clahe.apply, image)
    return enhanced, "CLAHE_Contrast"

async def sharpen_image(image, method="unsharp", kernel_size=(5, 5), strength=1.0):
    """повышение резкости"""
    if method == "unsharp":
        blurred = await asyncio.to_thread(
            cv2.GaussianBlur, image, kernel_size, 0
        )
        sharpened = await asyncio.to_thread(
            cv2.addWeighted, image, 1.0 + strength, blurred, -strength, 0
        )
        return sharpened, "Sharpening"

    elif method == "laplacian":
        gray = await ensure_grayscale(image)
        laplacian = await asyncio.to_thread(
            cv2.Laplacian, gray, cv2.CV_64F
        )
        laplacian = laplacian * strength * 0.01
        laplacian = np.clip(laplacian, 0, 255).astype(np.uint8)
        laplacian = await asyncio.to_thread(
            cv2.cvtColor, laplacian, cv2.COLOR_GRAY2BGR
        )
        sharpened = await asyncio.to_thread(cv2.add, image, laplacian)
        return sharpened, "Sharpening"

    elif method == "filter2d":
        kernel = (
            np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
            * strength
        )
        sharpened = await asyncio.to_thread(
            cv2.filter2D, image, -1, kernel
        )
        return sharpened, "Sharpening"


async def denoise_bilateral(image):
    """сглаживание"""
    denoised = await asyncio.to_thread(
        cv2.bilateralFilter, image, 9, 75, 75
    )
    return denoised, "Bilateral_Denoising"

async def denoise_nlmeans(image):
    """удаление шумов"""
    image = await ensure_grayscale(image)
    denoised = await asyncio.to_thread(
        cv2.fastNlMeansDenoising, image, None, 10, 7, 21
    )
    return denoised, "NL_Means_Denoising"




async def adaptive_binarization(image):
    """адаптивная бинаризация"""
    image = await ensure_grayscale(image)
    binary = await asyncio.to_thread(
        cv2.adaptiveThreshold,
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return binary, "Adaptive_Binarization"



async def otsu_binarization(image):
    """бинаризация по методу Оцу"""
    image = await ensure_grayscale(image)
    _, binary = await asyncio.to_thread(
        cv2.threshold, image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary, "Otsu_Binarization"



async def apply_otsu_mask(original_image, binary_image=None):
    """применение маски Оцу"""
    if binary_image is None:
        gray = await ensure_grayscale(original_image)
        _, binary_image = await asyncio.to_thread(
            cv2.threshold, gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    if len(original_image.shape) == 3:
        result = await asyncio.to_thread(
            cv2.cvtColor, original_image, cv2.COLOR_BGR2BGRA
        )
        result[:, :, 3] = await asyncio.to_thread(
            cv2.bitwise_and, result[:, :, 3], binary_image
        )
    else:
        result = await asyncio.to_thread(
            cv2.bitwise_and, original_image, binary_image
        )

    return result, "Otsu_Mask"



async def image_erosion(image, ksize=None):
    """эрозия"""
    if ksize is None:
        ksize = random.randint(1, 4)
    kernel = np.ones((ksize, ksize), np.uint8)
    processed_img = await asyncio.to_thread(
        cv2.erode, image, kernel, iterations=1
    )
    return processed_img, f"Erosion_k{ksize}"



async def image_dilation(image, ksize=None):
    """дилатация"""
    if ksize is None:
        ksize = random.randint(1, 4)
    kernel = np.ones((ksize, ksize), np.uint8)
    processed_img = await asyncio.to_thread(
        cv2.dilate, image, kernel, iterations=1
    )
    return processed_img, f"Dilation_k{ksize}"



async def image_closing(image):
    """морфологическое закрытие"""
    kernel = np.ones((3, 3), np.uint8)
    processed_img = await asyncio.to_thread(
        cv2.erode, image, kernel, iterations=1
    )
    processed_img = await asyncio.to_thread(
        cv2.dilate, processed_img, kernel, iterations=1
    )
    return processed_img, "Morphology_Closing"



async def image_opening(image):
    """морфологическое открытие"""
    kernel = np.ones((3, 3), np.uint8)
    processed_img = await asyncio.to_thread(
        cv2.dilate, image, kernel, iterations=1
    )
    processed_img = await asyncio.to_thread(
        cv2.erode, processed_img, kernel, iterations=1
    )
    return processed_img, "Morphology_Opening"



async def apply_gaussian_noise(image, mean=0, var=10):
    """добавление гауссовского шума"""
    sigma = var**0.5
    gaussian = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy = await asyncio.to_thread(
        cv2.add, image.astype(np.float32), gaussian
    )
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy, "Gaussian_Noise"



async def apply_salt_pepper_noise(image, amount=0.005, salt_vs_pepper=0.5):
    """добавление шума соли и перца"""
    noisy = np.copy(image)
    num_salt = np.ceil(amount * image.size * salt_vs_pepper).astype(int)
    num_pepper = np.ceil(amount * image.size * (1.0 - salt_vs_pepper)).astype(int)
    
    coords_salt = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
    coords_pepper = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
    
    noisy[coords_salt[0], coords_salt[1]] = 255
    noisy[coords_pepper[0], coords_pepper[1]] = 0
    
    return noisy, "Salt_Pepper_Noise"



async def smooth_edges(
    img: np.ndarray,
    blur_kernel_size: int = 5,
    threshold: int = 127,
    edge_blur_radius: int = 3,
    use_adaptive_threshold: bool = False,
    invert_mask: bool = False,
) -> np.ndarray:
    """сглаживание краёв"""
    image = await ensure_grayscale(img)
    blurred = await asyncio.to_thread(
        cv2.GaussianBlur, image, (blur_kernel_size, blur_kernel_size), 0
    )

    if use_adaptive_threshold:
        binary = await asyncio.to_thread(
            cv2.adaptiveThreshold,
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
    else:
        _, binary = await asyncio.to_thread(
            cv2.threshold, blurred, threshold, 255, cv2.THRESH_BINARY_INV
        )

    if not invert_mask:
        binary = await asyncio.to_thread(cv2.bitwise_not, binary)

    mask_blurred = await asyncio.to_thread(
        cv2.GaussianBlur, binary, (edge_blur_radius, edge_blur_radius), 0
    )
    mask = mask_blurred / 255.0

    if len(img.shape) == 3:
        mask = await asyncio.to_thread(cv2.merge, [mask, mask, mask])
    
    smoothed_img = (img * mask).astype(np.uint8)
    return smoothed_img, "Smooth_Edges"



async def apply_random_blur(image):
    """добавление случайного размытия"""
    if random.random() < 0.5:
        blurred = await asyncio.to_thread(
            cv2.GaussianBlur, image, (5, 5), 0
        )
        return blurred, "Gaussian_Blur"
    else:
        blurred = await asyncio.to_thread(
            cv2.medianBlur, image, 5
        )
        return blurred, "Median_Blur"



async def remove_binary_noise(binary_image, noise_level=3):
    """удаление шумов с бинаризованного изображения"""
    inverted = await asyncio.to_thread(cv2.bitwise_not, binary_image)
    kernel = np.ones((noise_level, noise_level), np.uint8)
    opened = await asyncio.to_thread(
        cv2.morphologyEx, inverted, cv2.MORPH_OPEN, kernel, iterations=1
    )
    closed = await asyncio.to_thread(
        cv2.morphologyEx, opened, cv2.MORPH_CLOSE, kernel, iterations=1
    )
    result = await asyncio.to_thread(cv2.bitwise_not, closed)
    return result



async def adjust_image_properties(image, mode="adaptive", isLight=True):
    """цветокоррекция изображения"""
    # Конвертация в HSV и коррекция насыщенности
    hsv = await asyncio.to_thread(cv2.cvtColor, image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.int16) - 100, 0, 255).astype(np.uint8)
    image = await asyncio.to_thread(cv2.cvtColor, hsv, cv2.COLOR_HSV2BGR)

    # Коррекция света в YCrCb
    ycrcb = await asyncio.to_thread(cv2.cvtColor, image, cv2.COLOR_BGR2YCrCb)
    if isLight:
        ycrcb[:, :, 0] = np.clip(ycrcb[:, :, 0].astype(np.int16) - 100, 0, 255).astype(np.uint8)
    else:
        ycrcb[:, :, 0] = np.clip(ycrcb[:, :, 0].astype(np.int16) + 100, 0, 255).astype(np.uint8)
    image = await asyncio.to_thread(cv2.cvtColor, ycrcb, cv2.COLOR_YCrCb2BGR)

    # Гамма-коррекция
    gamma = 0.30
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    image = await asyncio.to_thread(cv2.LUT, image, table)

    # Бинаризация
    if mode == "adaptive":
        image, _ = await adaptive_binarization(image)
    else:
        image, _ = await otsu_binarization(image)
        
    return image




async def rotate_image(image, angle):
    """Асинхронный поворот изображения"""
    (h, w) = image.shape[:2]

    radians = np.deg2rad(angle)
    sin = np.abs(np.sin(radians))
    cos = np.abs(np.cos(radians))

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    center_x, center_y = w // 2, h // 2
    new_center_x, new_center_y = new_w // 2, new_h // 2

    M = await asyncio.to_thread(
        cv2.getRotationMatrix2D, (center_x, center_y), angle, 1.0
    )

    M[0, 2] += new_center_x - center_x
    M[1, 2] += new_center_y - center_y

    rotated = await asyncio.to_thread(
        cv2.warpAffine,
        image, M, (new_w, new_h), 
        flags=cv2.INTER_CUBIC, 
        borderMode=cv2.BORDER_REPLICATE
    )

    return rotated



async def compute_skew_fft(binary_image):
    """вычисление угла наклона через FFT"""
    fft = await asyncio.to_thread(np.fft.fft2, binary_image)
    fft_shift = await asyncio.to_thread(np.fft.fftshift, fft)
    magnitude = 20 * np.log(np.abs(fft_shift))

    _, binary_spectrum = await asyncio.to_thread(
        cv2.threshold, 
        np.uint8(magnitude), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    lines = await asyncio.to_thread(
        cv2.HoughLinesP,
        binary_spectrum, 1, np.pi / 180, 100, 
        minLineLength=100, maxLineGap=10
    )

    if lines is None:
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angles.append(angle)

    return np.median(angles) if angles else 0.0


async def extract_and_align_symbols(original_image, boxes, padding=3, crop_margin=5):
    """выравнивание символов"""
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



async def resize_to_target_font_size(image, target_height=40):
    """изменение размера до целевой высоты шрифта"""
    current_height = image.shape[0]
    scale_factor = target_height / current_height
    new_width = int(image.shape[1] * scale_factor)
    new_height = target_height

    resized_image = await asyncio.to_thread(
        cv2.resize,
        image, (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )
    return resized_image


async def correct_box(image, box, padding=5):
    """уточнение границ бокса"""
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
