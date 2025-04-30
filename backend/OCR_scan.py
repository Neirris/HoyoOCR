import cv2
import image_processing as ImageProcessor
from pipelines import all_pipelines
from yolo_scan import detect_with_yolo
from tess_scan import run_tesseract_pipeline, compare_pipeline_results
from image_processing import (
    rotate_image,
    compute_skew_fft,
    extract_and_align_symbols,
    resize_to_target_font_size,
)


def run_pipeline(image, pipeline_steps, pipeline_name):
    current_image = image.copy()
    intermediate_results = {}

    for step in pipeline_steps:
        func = step[0]
        args = step[1:] if len(step) > 1 else ()

        try:
            result = func(current_image, *args)

            if isinstance(result, tuple):
                processed_image, step_name = result
            else:
                processed_image, _step_name = result, func.__name__

            if func.__name__ in ["normalize_image", "otsu_binarization"]:
                intermediate_results[func.__name__] = processed_image

            if func.__name__ == "apply_otsu_mask":
                if (
                    "normalize_image" in intermediate_results
                    and "otsu_binarization" in intermediate_results
                ):
                    processed_image, _ = ImageProcessor.apply_otsu_mask(
                        intermediate_results["normalize_image"],
                        intermediate_results["otsu_binarization"],
                    )

            current_image = processed_image

        except Exception as e:
            print(f"Error: {func.__name__}: {str(e)}")
            continue

    return {"image": current_image}


def main():
    original_image = cv2.imread("in magno.png")
    lang = "abyss"

    if original_image is None:
        return

    pipeline_results = []

    for name, pipeline in all_pipelines.items():
        result = run_pipeline(original_image, pipeline, name)
        binary_image = result["image"]

        skew_angle = compute_skew_fft(binary_image)
        deskewed = rotate_image(binary_image, -skew_angle)

        yolo_image, boxes, classes, scores = detect_with_yolo(deskewed)

        if len(boxes) > 0:
            new_image = extract_and_align_symbols(deskewed, boxes)
            resized_image = resize_to_target_font_size(new_image, target_height=200)
            result_details, result_image = run_tesseract_pipeline(
                resized_image, name, lang, min_confidence=70
            )
        else:
            resized_image = resize_to_target_font_size(deskewed, target_height=200)
            result_details, result_image = run_tesseract_pipeline(
                resized_image, name, lang, min_confidence=70
            )

        if result_details:
            pipeline_results.append((result_details, result_image))

    final_result, final_image = compare_pipeline_results(pipeline_results)
    return final_result


if __name__ == "__main__":
    main()
