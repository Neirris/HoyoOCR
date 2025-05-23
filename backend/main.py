from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from OCR_scan import async_ocr_scan
import uvicorn
import traceback
import os

app = FastAPI()

cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")


app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/translate")
async def translate_image(file: UploadFile = File(...), source_lang: str = Query(...)):
    try:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            return JSONResponse(
                status_code=400, content={"detail": "Failed to decode image"}
            )

        result = await async_ocr_scan(image, lang=source_lang)

        if result:
            return {
                "text": result.get("text", ""),
                "confidence": result.get("avg_conf", 0),
                "source": result.get("pipeline", ""),
                "boxes": result.get("boxes", []),
                "yolo_boxes": result.get("yolo_boxes", []),
                "yolo_classes": result.get("yolo_classes", []),
                "yolo_scores": result.get("yolo_scores", []),
            }

        return {
            "text": "",
            "confidence": 0,
            "source": "",
            "boxes": [],
            "yolo_boxes": [],
            "yolo_classes": [],
            "yolo_scores": [],
        }

    except Exception as e:
        print(f"[SERVER ERROR] Exception in /translate: {str(e)}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "detail": f"Server error: {str(e)}",
                "boxes": [],
                "yolo_boxes": [],
                "yolo_classes": [],
                "yolo_scores": [],
            },
        )


@app.get("/test")
async def test_func():
    print("test")
    return {"text": "", "confidence": 0, "source": ""}


@app.get("/")
async def root():
    return {"message": "OCR API is running"}


if __name__ == "__main__":
    print("Server is running at http://localhost:8000")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
    )  # reload=True
