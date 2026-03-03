import secrets
from typing import Annotated

from fastapi import Depends, HTTPException
from starlette.requests import Request

from ml.api_server.ocr_model_driver import OCRModel
from ml.config import env

"OCR Model"
def get_ocr_model(request: Request):
    return request.app.state.ocr_model

OCRDep = Annotated[OCRModel, Depends(get_ocr_model)]


"Auth DI"
def get_auth_service(request: Request):
    auth_header = request.headers.get("X-Auth-Service")
    if not auth_header or not secrets.compare_digest(auth_header, env.auth_service_secret):
        raise HTTPException(403, detail="Доступ запрещён")
