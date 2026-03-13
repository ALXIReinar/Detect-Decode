from contextlib import asynccontextmanager

import uvicorn
from aiohttp import ClientSession
from asyncpg import create_pool
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from web.api import main_router
from web.api.middleware import ASGILoggingMiddleware
from web.config import env, pool_settings, ml_aiohttp_settings


@asynccontextmanager
async def lifespan(web_app):
    """"""
    "Соединение с БД"
    web_app.state.pg_pool = await create_pool(**pool_settings)

    "Соединение с ML Server"
    web_app.state.ml_aiohttp = ClientSession(**ml_aiohttp_settings)
    "Отдельная сессия для скачивания файлов с Тг"
    web_app.state.any_aiohttp = ClientSession()
    try:
        yield
    finally:
        await web_app.state.pg_pool.close()
        await web_app.state.ml_aiohttp.close()
        await web_app.state.any_aiohttp.close()

app = FastAPI(
    docs_url='/api/docs',
    openapi_url='/api/openapi.json',
    lifespan=lifespan,
    response_model=env.post_processing_responses,
    response_model_exclude_unset=env.post_processing_responses
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
app.add_middleware(ASGILoggingMiddleware)


if __name__ == '__main__':
    uvicorn.run('web.main:app', host="0.0.0.0", port=8000, log_config=None, workers=env.uvi_workers)