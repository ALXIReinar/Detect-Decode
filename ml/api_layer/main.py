from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from ml.api_layer.ml_api import main_router
from ml.api_layer.ml_api.middleware import ASGILoggingMiddleware
from ml.api_layer.ocr_model_driver import OCRModel
from ml.config import env, broker, word_decoder_weights_path, detector_weights_path

from ml.api_layer.ml_api import s3_consumer # Для иниц консумера при старте брокера


@asynccontextmanager
async def lifespan(web_app):
    """"""
    "Загрузка модели"
    web_app.state.ocr_model = OCRModel(detector_weights_path, word_decoder_weights_path, use_beam_search=True, max_det=env.max_det)

    "Соединение с Кафкой"
    await broker.start()
    try:
        yield
    finally:
        await broker.stop()

app = FastAPI(
    lifespan=lifespan,
    docs_url='/api/docs',
    openapi_url='/api/openapi.json',
    response_model=env.post_processing_responses,
    response_model_exclude_unset=env.post_processing_responses,
)
app.include_router(main_router)

"Миддлвари"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://127.0.0.1", "http://127.0.0.1:8000", "http://localhost:8000", env.domain],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)
# app.add_middleware(ASGIAuthServiceMiddleware)
app.add_middleware(ASGILoggingMiddleware)


if __name__ == '__main__':
    uvicorn.run('ml.api_layer.main:app', host='0.0.0.0', port=8000, workers=env.uvi_workers, log_config=None)