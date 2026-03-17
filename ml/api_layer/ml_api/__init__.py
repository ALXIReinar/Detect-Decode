from fastapi import APIRouter
from .inference_api import router as inference_router

main_router = APIRouter(prefix="/api/v1")

main_router.include_router(inference_router)


@main_router.get("/healthcheck")
async def healthcheck():
    return {
        "status": True,
        'service': 'ml-server',
        'detector_version': {
            'model': 'yolov8n',
            'version': '0.1'
        },
        'word_decooder': {
            'model': 'crnn',
            'version': '0.2'
        }
    }
