from typing import Annotated

from fastapi import Depends
from starlette.requests import Request

from ml.api_layer.ocr_model_driver import OCRModel



"OCR Model"
def get_ocr_model(request: Request):
    return request.app.state.ocr_model

OCRDep = Annotated[OCRModel, Depends(get_ocr_model)]
