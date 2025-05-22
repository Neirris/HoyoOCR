import pytesseract
import cv2
import asyncio
import os
from collections import defaultdict

async def run_tesseract_pipeline_async(image, pipeline_name, lang="abyss", min_confidence=70):
    psm_modes = {
        6: "Assume a single uniform block of text",
        7: "Treat the image as a single text line",
        8: "Treat the image as a single word",
        10: "Treat the image as a single character",
        11: "Sparse text. Find as much text as possible in no particular order.",
        13: "Raw line. Treat the image as a single text line, bypassing hacks that are Tesseract-specific",
    }

    oem_modes = {3: "OEM 3 (Default)"}

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    tessdata_path = os.path.join("assets", "models", lang)
    if not os.path.exists(os.path.join(tessdata_path, f"{lang}.traineddata")):
        print(f"Error: Tesseract language file not found at {tessdata_path}/{lang}.traineddata")
        return None

    text_stats = defaultdict(list)

    for oem_key in oem_modes:
        for psm_key in psm_modes:
            config = f"--oem {oem_key} --psm {psm_key} -l {lang} --tessdata-dir {tessdata_path}"
            try:
                data = await asyncio.to_thread(
                    pytesseract.image_to_data,
                    image,
                    config=config,
                    output_type=pytesseract.Output.DICT,
                )

                full_text = []
                conf_values = []

                for i in range(len(data["text"])):
                    word = data["text"][i].strip()
                    try:
                        conf = float(data["conf"][i])
                    except ValueError:
                        continue

                    if word and conf >= min_confidence:
                        full_text.append(word)
                        conf_values.append(conf)

                if full_text:
                    joined = " ".join(full_text)
                    avg_conf = sum(conf_values) / len(conf_values)
                    text_stats[joined].append({
                        "avg_conf": avg_conf,
                        "count": 1,
                        "oem": oem_key,
                        "psm": psm_key,
                        "confs": conf_values,
                    })

            except Exception as e:
                print(f"Error OCR in {pipeline_name} (OEM={oem_key}, PSM={psm_key}): {e}")

    if not text_stats:
        print(f"[Tesseract] no results: {pipeline_name}")
        return None

    best_text, best_score, best_details = None, -1, None

    for text, stats_list in text_stats.items():
        total_count = sum(s["count"] for s in stats_list)
        total_avg_conf = (
            sum(s["avg_conf"] * s["count"] for s in stats_list) / total_count
        )
        score = total_avg_conf * total_count

        if score > best_score:
            best_score = score
            best_text = text
            best_details = {
                "text": text,
                "avg_conf": total_avg_conf,
                "count": total_count,
                "score": score,
                "pipeline": pipeline_name,
            }

    if best_text:
        for stats in text_stats[best_text]:
            config = f"--oem {stats['oem']} --psm {stats['psm']} -l {lang} --tessdata-dir {tessdata_path}"
            data = await asyncio.to_thread(
                pytesseract.image_to_data,
                image,
                config=config,
                output_type=pytesseract.Output.DICT,
            )
            full_text = " ".join(
                [data["text"][i].strip() for i in range(len(data["text"])) if data["text"][i].strip()]
            )
            if full_text == best_text:
                box_data = []
                for i in range(len(data["text"])):
                    word = data["text"][i].strip()
                    try:
                        conf = float(data["conf"][i])
                    except ValueError:
                        continue

                    if word and conf >= min_confidence:
                        x, y, w, h = (
                            data["left"][i],
                            data["top"][i],
                            data["width"][i],
                            data["height"][i],
                        )
                        box_data.append({
                            "text": word,
                            "confidence": conf,
                            "bbox": [x, y, w, h],
                        })

                return {
                    "text": best_text,
                    "avg_conf": best_details["avg_conf"],
                    "boxes": box_data,
                    "pipeline": pipeline_name,
                }

    return None

async def compare_pipeline_results_async(images, pipeline_names, lang="abyss", min_confidence=70):
    tasks = [
        run_tesseract_pipeline_async(image, name, lang=lang, min_confidence=min_confidence)
        for image, name in zip(images, pipeline_names)
    ]
    results = await asyncio.gather(*tasks)

    final_text_stats = defaultdict(list)

    for result in results:
        if not result:
            continue
        details = result
        final_text_stats[details["text"]].append({
            "avg_conf": details["avg_conf"],
            "count": len(details["boxes"]),
            "pipeline": details["pipeline"],
            "score": details["avg_conf"] * len(details["boxes"]),
            "boxes": details["boxes"],
        })

    final_scores = []
    for text, stats_list in final_text_stats.items():
        total_count = sum(s["count"] for s in stats_list)
        total_avg_conf = (
            sum(s["avg_conf"] * s["count"] for s in stats_list) / total_count
        )
        total_score = total_avg_conf * total_count
        pipelines = [s["pipeline"] for s in stats_list]

        final_scores.append({
            "text": text,
            "avg_conf": total_avg_conf,
            "count": total_count,
            "score": total_score,
            "pipelines": pipelines,
            "boxes": stats_list[0]["boxes"],
        })

    final_scores.sort(key=lambda x: x["score"], reverse=True)

    if final_scores:
        best = final_scores[0]
        return best, results

    return None, results