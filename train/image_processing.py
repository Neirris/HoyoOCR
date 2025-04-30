import os
import tempfile
import cv2
import numpy as np
import random


def ensure_grayscale(image):
    """Преобразование в оттенки серого"""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def set_image_dpi(image, dpi=300):
    """Установка DPI"""
    original_is_grayscale = len(image.shape) == 2
    image_to_save = (
        image if not original_is_grayscale else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    )

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_file:
        temp_path = tmp_file.name

    try:
        cv2.imwrite(
            temp_path,
            image_to_save,
            [cv2.IMWRITE_TIFF_XDPI, dpi, cv2.IMWRITE_TIFF_YDPI, dpi],
        )
        result = cv2.imread(temp_path)
        if original_is_grayscale:
            result = ensure_grayscale(result)

        return result, f"DPI {dpi}"

    finally:
        os.unlink(temp_path)


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
    """Cглаживание"""
    denoised = cv2.bilateralFilter(image, 9, 75, 75)
    return denoised, "Bilateral_Denoising"


def denoise_nlmeans(image):
    """Удаление шумов"""
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
    """Маска Оцу"""
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
        ksize = random.randint(1, 4)
    kernel = np.ones((ksize, ksize), np.uint8)
    processed_img = cv2.erode(image, kernel, iterations=1)
    return processed_img, f"Erosion_k{ksize}"


def image_dilation(image, ksize=None):
    """Дилатация (расширение)"""
    if ksize is None:
        ksize = random.randint(1, 4)
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
    """Гауссовский шум"""
    sigma = var**0.5
    gaussian = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy = cv2.add(image.astype(np.float32), gaussian)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy, "Gaussian_Noise"


def apply_salt_pepper_noise(image, amount=0.005, salt_vs_pepper=0.5):
    """Шум соли и перца"""
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
    smoothed_img = (img * mask).astype(np.uint8)

    return smoothed_img, "Smooth_Edges"


def apply_random_blur(image):
    """Добавление случайного размытия"""
    if random.random() < 0.5:
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        return blurred, "Gaussia_Blur"
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
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Коррекция насыщенности
    hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.int16) - 100, 0, 255).astype(np.uint8)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Коррекция света
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    if isLight:
        ycrcb[:, :, 0] = np.clip(ycrcb[:, :, 0].astype(np.int16) - 100, 0, 255).astype(
            np.uint8
        )
    else:
        ycrcb[:, :, 0] = np.clip(ycrcb[:, :, 0].astype(np.int16) + 100, 0, 255).astype(
            np.uint8
        )
    image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    # Коррекция теней
    gamma = 0.30
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    image = cv2.LUT(image, table)

    if mode == "adaptive":
        image = adaptive_binarization(image)
    else:
        image = otsu_binarization(image)
    return image
