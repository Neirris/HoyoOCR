import cv2
import numpy as np
import random
import asyncio

async def ensure_grayscale(image):
    """Преобразование в оттенки серого"""
    if len(image.shape) == 3:
        return await asyncio.to_thread(cv2.cvtColor, image, cv2.COLOR_BGR2GRAY)
    return image


async def normalize_image(image):
    """Нормализация изображения"""
    image = await ensure_grayscale(image)
    norm_img = np.zeros((image.shape[0], image.shape[1]))
    normalized = await asyncio.to_thread(
        cv2.normalize, image, norm_img, 0, 255, cv2.NORM_MINMAX
    )
    return normalized, "Normalization"

async def apply_clahe(image):
    """Повышение контрастности"""
    image = await ensure_grayscale(image)
    clahe = await asyncio.to_thread(
        cv2.createCLAHE, clipLimit=3.0, tileGridSize=(8, 8)
    )
    enhanced = await asyncio.to_thread(clahe.apply, image)
    return enhanced, "CLAHE_Contrast"

async def sharpen_image(image, method="unsharp", kernel_size=(5, 5), strength=1.0):
    """Повышение резкости"""
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
    """Сглаживание"""
    denoised = await asyncio.to_thread(
        cv2.bilateralFilter, image, 9, 75, 75
    )
    return denoised, "Bilateral_Denoising"

async def denoise_nlmeans(image):
    """Удаление шумов"""
    image = await ensure_grayscale(image)
    denoised = await asyncio.to_thread(
        cv2.fastNlMeansDenoising, image, None, 10, 7, 21
    )
    return denoised, "NL_Means_Denoising"




async def adaptive_binarization(image):
    """Адаптивная бинаризация"""
    image = await ensure_grayscale(image)
    binary = await asyncio.to_thread(
        cv2.adaptiveThreshold,
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return binary, "Adaptive_Binarization"



async def otsu_binarization(image):
    """Бинаризация по методу Оцу"""
    image = await ensure_grayscale(image)
    _, binary = await asyncio.to_thread(
        cv2.threshold, image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return binary, "Otsu_Binarization"



async def apply_otsu_mask(original_image, binary_image=None):
    """Применение маски Оцу"""
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
    """Эрозия"""
    if ksize is None:
        ksize = random.randint(1, 4)
    kernel = np.ones((ksize, ksize), np.uint8)
    processed_img = await asyncio.to_thread(
        cv2.erode, image, kernel, iterations=1
    )
    return processed_img, f"Erosion_k{ksize}"



async def image_dilation(image, ksize=None):
    """Дилатация"""
    if ksize is None:
        ksize = random.randint(1, 4)
    kernel = np.ones((ksize, ksize), np.uint8)
    processed_img = await asyncio.to_thread(
        cv2.dilate, image, kernel, iterations=1
    )
    return processed_img, f"Dilation_k{ksize}"



async def image_closing(image):
    """Морфологическое закрытие"""
    kernel = np.ones((3, 3), np.uint8)
    processed_img = await asyncio.to_thread(
        cv2.erode, image, kernel, iterations=1
    )
    processed_img = await asyncio.to_thread(
        cv2.dilate, processed_img, kernel, iterations=1
    )
    return processed_img, "Morphology_Closing"



async def image_opening(image):
    """Морфологическое открытие"""
    kernel = np.ones((3, 3), np.uint8)
    processed_img = await asyncio.to_thread(
        cv2.dilate, image, kernel, iterations=1
    )
    processed_img = await asyncio.to_thread(
        cv2.erode, processed_img, kernel, iterations=1
    )
    return processed_img, "Morphology_Opening"



async def apply_gaussian_noise(image, mean=0, var=10):
    """Добавление гауссовского шума"""
    sigma = var**0.5
    gaussian = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy = await asyncio.to_thread(
        cv2.add, image.astype(np.float32), gaussian
    )
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy, "Gaussian_Noise"



async def apply_salt_pepper_noise(image, amount=0.005, salt_vs_pepper=0.5):
    """Добавление шума соли и перца"""
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
    """Сглаживание краёв"""
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
    """Добавление случайного размытия"""
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
    """Удаление шумов с бинаризованного изображения"""
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
    """Цветокоррекция изображения"""
    if len(image.shape) == 2:
        image = await asyncio.to_thread(cv2.cvtColor, image, cv2.COLOR_GRAY2BGR)

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


async def resize_to_target_font_size(image, target_height=40):
    """Изменение размера до целевой высоты шрифта"""
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
