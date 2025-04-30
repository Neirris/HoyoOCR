import image_processing as ImageProcessor

all_pipelines = {
    "General_Otsu": [
        (ImageProcessor.set_image_dpi, 300),
        (ImageProcessor.ensure_grayscale,),
        (ImageProcessor.normalize_image,),
        (ImageProcessor.otsu_binarization,),
    ],
    "Otsu_Adaptive_1": [
        (ImageProcessor.set_image_dpi, 300),
        (ImageProcessor.ensure_grayscale,),
        (ImageProcessor.normalize_image,),
        (ImageProcessor.otsu_binarization,),
        (ImageProcessor.apply_otsu_mask,),
        (ImageProcessor.adaptive_binarization,),
        (ImageProcessor.remove_binary_noise, 1),
    ],
    "Otsu_Adaptive_2": [
        (ImageProcessor.set_image_dpi, 300),
        (ImageProcessor.ensure_grayscale,),
        (ImageProcessor.normalize_image,),
        (ImageProcessor.otsu_binarization,),
        (ImageProcessor.apply_otsu_mask,),
        (ImageProcessor.adaptive_binarization,),
        (ImageProcessor.remove_binary_noise, 3),
    ],
    "General_Adaptive_1": [
        (ImageProcessor.set_image_dpi, 300),
        (ImageProcessor.ensure_grayscale,),
        (ImageProcessor.normalize_image,),
        (ImageProcessor.adaptive_binarization,),
    ],
    "General_Adaptive_2": [
        (ImageProcessor.set_image_dpi, 300),
        (ImageProcessor.ensure_grayscale,),
        (ImageProcessor.normalize_image,),
        (ImageProcessor.adaptive_binarization,),
        (ImageProcessor.remove_binary_noise, 3),
    ],
    "Denoising": [
        (ImageProcessor.set_image_dpi, 300),
        (ImageProcessor.ensure_grayscale,),
        (ImageProcessor.normalize_image,),
        (ImageProcessor.denoise_nlmeans,),
        (ImageProcessor.adaptive_binarization,),
    ],
    "Denoising_smooth": [
        (ImageProcessor.set_image_dpi, 300),
        (ImageProcessor.ensure_grayscale,),
        (ImageProcessor.normalize_image,),
        (ImageProcessor.denoise_nlmeans,),
        (ImageProcessor.adaptive_binarization,),
        (ImageProcessor.image_closing,),
        (ImageProcessor.smooth_edges,),
    ],
    "Contrast": [
        (ImageProcessor.set_image_dpi, 300),
        (ImageProcessor.ensure_grayscale,),
        (ImageProcessor.normalize_image,),
        (ImageProcessor.apply_clahe,),
        (ImageProcessor.denoise_nlmeans,),
        (ImageProcessor.adaptive_binarization,),
    ],
    "Unblur": [
        (ImageProcessor.set_image_dpi, 300),
        (ImageProcessor.ensure_grayscale,),
        (ImageProcessor.normalize_image,),
        (ImageProcessor.denoise_bilateral,),
        (ImageProcessor.sharpen_image, "unsharp"),
        (ImageProcessor.denoise_nlmeans,),
        (ImageProcessor.adaptive_binarization,),
    ],
    "Properties_Adaptive": [
        (ImageProcessor.adjust_image_properties,),
    ],
    "Properties_Adaptive_Dark": [
        (ImageProcessor.adjust_image_properties, "adaptive", False),
    ],
    "Properties_Otsu": [
        (ImageProcessor.adjust_image_properties, "otsu"),
    ],
    "Properties_Otsu_Dark": [
        (ImageProcessor.adjust_image_properties, "otsu", False),
    ],
}
