import image_processing as ImageProcessor
from pipelines import all_pipelines
from yolo_scan import detect_with_yolo
from tess_scan import run_tesseract_pipeline_async, compare_pipeline_results_async
from box_processing import extract_and_align_symbols
import asyncio

async def run_pipeline(image, pipeline_steps, pipeline_name):
    current_image = image.copy()
    intermediate_results = {}

    for step in pipeline_steps:
        func = step[0]
        args = step[1:] if len(step) > 1 else ()

        try:
            result = await func(current_image, *args)
            
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
                    processed_image, _ = await ImageProcessor.apply_otsu_mask(
                        intermediate_results["normalize_image"],
                        intermediate_results["otsu_binarization"],
                    )

            current_image = processed_image

        except Exception as e:
            print(f"Error in {func.__name__} for pipeline {pipeline_name}: {str(e)}")
            continue

    if current_image is None or current_image.size == 0:
        print(f"Warning: Pipeline {pipeline_name} produced an invalid image.")
        return {"image": None}

    return {"image": current_image}

async def async_ocr_scan(original_image, lang="abyss"):
    if original_image is None or original_image.size == 0:
        print("Error: Input image is invalid or empty.")
        return None

    async def process_pipeline(image, pipeline, name):
        result = await run_pipeline(image, pipeline, name)
        binary_image = result["image"]

        if binary_image is None:
            print(f"Warning: Pipeline {name} returned invalid image. Skipping.")
            return {
                "result": None,
                "image": None,
                "name": name,
                "yolo_boxes": [],
                "yolo_classes": [],
                "yolo_scores": []
            }
            
        yolo_image, boxes, classes, scores = await detect_with_yolo(binary_image, lang=lang, pipeline=name)

        if len(boxes) > 0:
            new_image = await extract_and_align_symbols(binary_image, boxes)
        else:
            new_image = binary_image

        resized_image = await ImageProcessor.resize_to_target_font_size(new_image, target_height=200)
        
        
        ocr_result = await run_tesseract_pipeline_async(
            resized_image, name, lang, min_confidence=70
        )
        
        return {
            "result": ocr_result,
            "image": resized_image,
            "name": name,
            "yolo_boxes": boxes.tolist() if boxes is not None else [],
            "yolo_classes": classes.tolist() if classes is not None else [],
            "yolo_scores": scores.tolist() if scores is not None else [],
            "skew_angle": 0.0 
        }

    tasks = []
    for name, pipeline in all_pipelines.items():
        tasks.append(process_pipeline(original_image, pipeline, f"{name}_original"))

    pipeline_outputs = await asyncio.gather(*tasks)
    
    valid_results = [out for out in pipeline_outputs if out["result"] is not None]
    
    if not valid_results:
        return None

    images = [out["image"] for out in valid_results]
    names = [out["name"] for out in valid_results]

    final_result, _ = await compare_pipeline_results_async(images, names)
    
    if not final_result:
        return None

    best_pipeline_name = final_result.get("pipelines", [""])[0]
    yolo_data = next(
        (out for out in valid_results if out["name"] == best_pipeline_name), 
        None
    )

    if yolo_data and final_result:
        final_result.update({
            "yolo_boxes": yolo_data["yolo_boxes"],
            "yolo_classes": yolo_data["yolo_classes"],
            "yolo_scores": yolo_data["yolo_scores"],
            "skew_angle": yolo_data["skew_angle"]
        })

    return final_result