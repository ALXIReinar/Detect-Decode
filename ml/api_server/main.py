import asyncio
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from ml.api_server.ml_api import main_router
from ml.api_server.ocr_model_driver import OCRModel
from ml.config import env, broker


@asynccontextmanager
async def lifespan(web_app):
    """"""
    "Загрузка модели"
    web_app.state.ocr_model = OCRModel(env.detector_weights_path, env.word_decoder_weights_path)

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://127.0.0.1", "http://127.0.0.1:8000", "http://localhost:8000", env.domain],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)


if __name__ == '__main__':
    uvicorn.run('ml.api_server:app', host='0.0.0.0', port=8000, workers=env.uvi_workers)