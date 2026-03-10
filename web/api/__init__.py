from fastapi import APIRouter

from web.api.tg_routing import tg_router

main_router = APIRouter(prefix="/api/v1")

main_router.include_router(tg_router)