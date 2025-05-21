import cv2
import numpy as np
import random

def ensure_grayscale(image):
    """Преобразование в оттенки серого"""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 4:  # RGBA
        return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    return image

def normalize_image(image):
    """Нормализация изображения"""
    image = ensure_grayscale(image)
    norm_img = np.zeros((image.shape[0], image.shape[1]))
    normalized = cv2.normalize(image, norm_img, 0, 255, cv2.NORM_MINMAX)
    return normalized, "Normalization"

def apply_clahe(image):
    """Повышение контрастности"""
    image = ensure_grayscale(image)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    return enhanced, "CLAHE_Contrast"

def sharpen_image(image, method="unsharp", kernel_size=(5, 5), strength=1.0):
    """Повышение резкости"""
    if len(image.shape) == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    if method == "unsharp":
        blurred = cv2.GaussianBlur(image, kernel_size, 0)
        sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
        return sharpened, "Sharpening"
    elif method == "laplacian":
        gray = ensure_grayscale(image)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = laplacian * strength * 0.01
        laplacian = np.clip(laplacian, 0, 255).astype(np.uint8)
        laplacian = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)
        sharpened = cv2.add(image, laplacian)
        return sharpened, "Sharpening"
    elif method == "filter2d":
        kernel = (
            np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
            * strength
        )
        sharpened = cv2.filter2D(image, -1, kernel)
        return sharpened, "Sharpening"

def denoise_bilateral(image):
    """Применение двустороннего сглаживания"""
    if len(image.shape) == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    if len(image.shape) == 2:  # Оттенки серого
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    denoised = cv2.bilateralFilter(image, 9, 75, 75)
    return denoised, "Bilateral_Denoising"

def denoise_nlmeans(image):
    """Применение сглаживания методом нелокальных средних"""
    image = ensure_grayscale(image)
    denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    return denoised, "NL_Means_Denoising"

def adaptive_binarization(image):
    """Адаптивная бинаризация"""
    image = ensure_grayscale(image)
    binary = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return binary, "Adaptive_Binarization"

def otsu_binarization(image):
    """Бинаризация по методу Оцу"""
    image = ensure_grayscale(image)
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary, "Otsu_Binarization"

def apply_otsu_mask(original_image, binary_image=None):
    """Применение маски Оцу"""
    if len(original_image.shape) == 4:  # RGBA
        original_image = cv2.cvtColor(original_image, cv2.COLOR_RGBA2BGR)
    if binary_image is None:
        gray = ensure_grayscale(original_image)
        _, binary_image = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    if len(original_image.shape) == 3:
        result = cv2.cvtColor(original_image, cv2.COLOR_BGR2BGRA)
        result[:, :, 3] = cv2.bitwise_and(result[:, :, 3], binary_image)
    else:
        result = cv2.bitwise_and(original_image, binary_image)
    return result, "Otsu_Mask"

def image_erosion(image, ksize=None):
    """Эрозия (сужение)"""
    if ksize is None:
        ksize = random.randint(1, 3)
    kernel = np.ones((ksize, ksize), np.uint8)
    processed_img = cv2.erode(image, kernel, iterations=1)
    return processed_img, f"Erosion_k{ksize}"

def image_dilation(image, ksize=None):
    """Дилатация (расширение)"""
    if ksize is None:
        ksize = random.randint(1, 2)
    kernel = np.ones((ksize, ksize), np.uint8)
    processed_img = cv2.dilate(image, kernel, iterations=1)
    return processed_img, f"Dilation_k{ksize}"

def image_closing(image):
    """Морфологическое закрытие (удаление мелких отверстий)"""
    kernel = np.ones((3, 3), np.uint8)
    processed_img = cv2.erode(image, kernel, iterations=1)
    processed_img = cv2.dilate(processed_img, kernel, iterations=1)
    return processed_img, "Morphology_Closing"

def image_opening(image):
    """Морфологическое открытие (удаление мелких объектов)"""
    kernel = np.ones((3, 3), np.uint8)
    processed_img = cv2.dilate(image, kernel, iterations=1)
    processed_img = cv2.erode(processed_img, kernel, iterations=1)
    return processed_img, "Morphology_Opening"

def apply_gaussian_noise(image, mean=0, var=10):
    """Добавление гауссовского шума"""
    if len(image.shape) == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    sigma = var**0.5
    gaussian = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy = cv2.add(image.astype(np.float32), gaussian)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy, "Gaussian_Noise"

def apply_salt_pepper_noise(image, amount=0.005, salt_vs_pepper=0.5):
    """Добавление шума соли и перца"""
    if len(image.shape) == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    noisy = np.copy(image)
    num_salt = np.ceil(amount * image.size * salt_vs_pepper).astype(int)
    num_pepper = np.ceil(amount * image.size * (1.0 - salt_vs_pepper)).astype(int)
    coords_salt = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
    coords_pepper = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
    noisy[coords_salt[0], coords_salt[1]] = 255
    noisy[coords_pepper[0], coords_pepper[1]] = 0
    return noisy, "Salt_Pepper_Noise"

def smooth_edges(
    img: np.ndarray,
    blur_kernel_size: int = 5,
    threshold: int = 127,
    edge_blur_radius: int = 3,
    use_adaptive_threshold: bool = False,
    invert_mask: bool = False,
) -> np.ndarray:
    """Сглаживание краёв изображения"""
    image = ensure_grayscale(img)
    blurred = cv2.GaussianBlur(image, (blur_kernel_size, blur_kernel_size), 0)
    if use_adaptive_threshold:
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
    else:
        _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)
    if not invert_mask:
        binary = cv2.bitwise_not(binary)
    mask_blurred = cv2.GaussianBlur(binary, (edge_blur_radius, edge_blur_radius), 0)
    mask = mask_blurred / 255.0
    if len(img.shape) == 3:
        mask = cv2.merge([mask, mask, mask])
    elif len(img.shape) == 4:
        mask = cv2.merge([mask, mask, mask, mask])
    smoothed_img = (img * mask).astype(np.uint8)
    return smoothed_img, "Smooth_Edges"

def apply_random_blur(image):
    """Добавление случайного размытия"""
    if len(image.shape) == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    if random.random() < 0.5:
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        return blurred, "Gaussian_Blur"
    else:
        blurred = cv2.medianBlur(image, 5)
        return blurred, "Median_Blur"

def remove_binary_noise(binary_image, noise_level=3):
    """Удаление шумов с бинаризованного изображения"""
    inverted = cv2.bitwise_not(binary_image)
    kernel = np.ones((noise_level, noise_level), np.uint8)
    opened = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
    result = cv2.bitwise_not(closed)
    return result

def adjust_image_properties(image, mode="adaptive", isLight=True):
    """Цветокоррекция изображения"""
    if len(image.shape) == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.int16) - 100, 0, 255).astype(np.uint8)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    if isLight:
        ycrcb[:, :, 0] = np.clip(ycrcb[:, :, 0].astype(np.int16) - 100, 0, 255).astype(np.uint8)
    else:
        ycrcb[:, :, 0] = np.clip(ycrcb[:, :, 0].astype(np.int16) + 100, 0, 255).astype(np.uint8)
    image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    gamma = 0.30
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    image = cv2.LUT(image, table)
    if mode == "adaptive":
        image = adaptive_binarization(image)
    else:
        image = otsu_binarization(image)
    return image

def add_text_glow(image, text_mask, glow_size=None):
    """Добавляет внешнее свечение к тексту на изображении по заданной маске"""
    if glow_size is None:
        glow_size = random.randint(3, 10)
    
    if len(image.shape) == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    elif len(image.shape) == 2:  # Оттенки серого
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    output_image = image.copy()
    
    if text_mask.shape[:2] != image.shape[:2]:
        text_mask = cv2.resize(text_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    kernel = np.ones((glow_size, glow_size), np.uint8)
    glow_mask = cv2.dilate(text_mask, kernel, iterations=1)
    
    glow_layer = np.zeros_like(image, dtype=np.uint8) 
    y, x = np.where(glow_mask == 255)
    glow_layer[y, x, :] = [255, 255, 255] 
    
    glow_layer = cv2.GaussianBlur(glow_layer, (glow_size * 2 + 1, glow_size * 2 + 1), 0)
    
    result = output_image.copy()
    mask = glow_mask > 0
    result[mask] = cv2.addWeighted(output_image[mask], 0.5, glow_layer[mask], 0.5, 0.0)
    
    text_mask_binary = text_mask > 0
    result[text_mask_binary] = image[text_mask_binary]
    
    return result, f"Glow_k{glow_size}"

def resize_to_target_font_size(image, mask, boxes, target_height=40):
    """Масштабирование до целевой высоты шрифта"""
    current_height = image.shape[0]
    if current_height == 0:
        return image, mask, boxes  
    
    scale_factor = target_height / current_height
    new_width = int(image.shape[1] * scale_factor)
    new_height = target_height
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    scaled_boxes = [
        (
            int(x1 * scale_factor) if x1 is not None else 0,
            int(y1 * scale_factor) if y1 is not None else 0,
            int(x2 * scale_factor) if x2 is not None else 0,
            int(y2 * scale_factor) if y2 is not None else 0
        ) for x1, y1, x2, y2 in boxes
    ]
    return resized_image, resized_mask, scaled_boxes