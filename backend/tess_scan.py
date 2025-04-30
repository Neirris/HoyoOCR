import pytesseract
import cv2
from collections import defaultdict


def run_tesseract_pipeline(image, pipeline_name, lang="abyss", min_confidence=70):
    """Запускает Tesseract OCR и возвращает лучший результат для данного пайплайна"""

    psm_modes = {
        6: "Assume a single uniform block of text",
        7: "Treat the image as a single text line",
        8: "Treat the image as a single word",
        10: "Treat the image as a single character",
        11: "Sparse text. Find as much text as possible in no particular order.",
        13: "Raw line. Treat the image as a single text line, bypassing hacks that are Tesseract-specific",
    }

    oem_modes = {
        3: "OEM 3 (Default)",
    }

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    text_stats = defaultdict(list)

    for oem_key, oem_name in oem_modes.items():
        for psm_key, psm_name in psm_modes.items():
            config = f"--oem {oem_key} --psm {psm_key} -l {lang}"

            try:
                data = pytesseract.image_to_data(
                    image, config=config, output_type=pytesseract.Output.DICT
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
                    avg_conf = sum(conf_values) / len(conf_values) if conf_values else 0
                    text_stats[joined].append(
                        {
                            "avg_conf": avg_conf,
                            "count": 1,
                            "oem": oem_key,
                            "psm": psm_key,
                            "confs": conf_values,
                        }
                    )

            except Exception as e:
                print(
                    f"Error OCR in {pipeline_name} (OEM={oem_key}, PSM={psm_key}): {str(e)}"
                )

    if not text_stats:
        print(f"[Tesseract] no results: {pipeline_name}")
        return None, None

    # Находим лучший результат для этого пайплайна
    best_text = None
    best_score = -1
    best_details = None
    best_img_with_boxes = None

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
            config = f"--oem {stats['oem']} --psm {stats['psm']} -l {lang}"
            data = pytesseract.image_to_data(
                image, config=config, output_type=pytesseract.Output.DICT
            )
            full_text = " ".join(
                [
                    data["text"][i].strip()
                    for i in range(len(data["text"]))
                    if data["text"][i].strip()
                ]
            )
            if full_text == best_text:
                img_with_boxes = image.copy()

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
                        cv2.rectangle(
                            img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2
                        )
                        cv2.putText(
                            img_with_boxes,
                            word,
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            1,
                            cv2.LINE_AA,
                        )

                best_img_with_boxes = img_with_boxes
                break

    return best_details, best_img_with_boxes


def compare_pipeline_results(results):
    """Сравнивает результаты всех пайплайнов и выбирает лучший"""
    if not results:
        return None

    # Собираем все уникальные тексты и их статистику
    final_text_stats = defaultdict(list)

    for result in results:
        if result[0] is None:  # Пропускаем неудачные пайплайны
            continue
        result_details = result[0]
        final_text_stats[result_details["text"]].append(
            {
                "avg_conf": result_details["avg_conf"],
                "count": result_details["count"],
                "pipeline": result_details["pipeline"],
                "score": result_details["score"],
            }
        )

    # Вычисляем общую оценку для каждого уникального текста
    final_scores = []
    for text, stats_list in final_text_stats.items():
        total_count = sum(s["count"] for s in stats_list)
        total_avg_conf = (
            sum(s["avg_conf"] * s["count"] for s in stats_list) / total_count
        )
        total_score = total_avg_conf * total_count

        pipelines = [s["pipeline"] for s in stats_list]

        final_scores.append(
            {
                "text": text,
                "avg_conf": total_avg_conf,
                "count": total_count,
                "score": total_score,
                "pipelines": pipelines,
            }
        )

    # Сортируем результаты по оценке
    final_scores.sort(key=lambda x: x["score"], reverse=True)

    # Возвращаем лучший результат
    if final_scores:
        best_result = final_scores[0]

        # Находим изображение с bounding boxes из лучшего пайплайна
        best_pipeline_name = best_result["pipelines"][0]
        for result in results:
            if result[0] and result[0]["pipeline"] == best_pipeline_name:
                return result[0], result[1]

    return None, None