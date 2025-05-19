import image_processing as ImageProcessor
from pipelines import all_pipelines
from yolo_scan import detect_with_yolo
from tess_scan import run_tesseract_pipeline_async, compare_pipeline_results_async
from image_processing import (
    rotate_image,
    compute_skew_fft,
    extract_and_align_symbols,
    resize_to_target_font_size,
)
import asyncio


async def run_pipeline(image, pipeline_steps, pipeline_name):
    current_image = image.copy()
    intermediate_results = {}

    for step in pipeline_steps:
        func = step[0]
        args = step[1:] if len(step) > 1 else ()

        try:
            if callable(func):
                result = (
                    await func(current_image, *args)
                    if func.__name__
                    in [
                        "rotate_image",
                        "compute_skew_fft",
                        "extract_and_align_symbols",
                        "resize_to_target_font_size",
                    ]
                    else func(current_image, *args)
                )

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


async def async_ocr_scan(original_image, lang="abyss"):
    if original_image is None:
        return None

    async def process_pipeline(name, pipeline):
        """Асинхронная обработка одного пайплайна с сохранением YOLO боксов"""
        result = await run_pipeline(original_image, pipeline, name)
        binary_image = result["image"]

        skew_angle = await compute_skew_fft(binary_image)
        deskewed = await rotate_image(binary_image, -skew_angle)

        yolo_image, boxes, classes, scores = await detect_with_yolo(deskewed, model_path=f'{lang}.pt')

        if len(boxes) > 0:
            new_image = await extract_and_align_symbols(deskewed, boxes)
        else:
            new_image = deskewed

        resized_image = await resize_to_target_font_size(new_image, target_height=200)
        
        ocr_result = await run_tesseract_pipeline_async(
            resized_image, name, lang, min_confidence=70
        )
        
        return {
            "result": ocr_result,
            "image": resized_image,
            "name": name,
            "yolo_boxes": boxes.tolist() if boxes is not None else [],
            "yolo_classes": classes.tolist() if classes is not None else [],
            "yolo_scores": scores.tolist() if scores is not None else []
        }

    tasks = [
        process_pipeline(name, pipeline) 
        for name, pipeline in all_pipelines.items()
    ]
    pipeline_outputs = await asyncio.gather(*tasks)
    
    valid_results = [out for out in pipeline_outputs if out["result"] is not None]
    
    if not valid_results:
        return None

    images = [out["image"] for out in valid_results]
    names = [out["name"] for out in valid_results]
    results = [out["result"] for out in valid_results]

    final_result, _ = await compare_pipeline_results_async(images, names)
    
    best_pipeline_name = final_result.get("pipelines", [""])[0]
    yolo_data = next(
        (out for out in valid_results if out["name"] == best_pipeline_name), 
        None
    )

    if yolo_data and final_result:
        final_result.update({
            "yolo_boxes": yolo_data["yolo_boxes"],
            "yolo_classes": yolo_data["yolo_classes"],
            "yolo_scores": yolo_data["yolo_scores"]
        })

    return final_result