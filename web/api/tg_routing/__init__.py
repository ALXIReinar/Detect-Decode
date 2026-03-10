from fastapi import APIRouter
from web.api.tg_routing.users_api import router as users_router
from web.api.tg_routing.ml_service.ml_service_api import router as inference_api_router

tg_router = APIRouter(prefix="/telegram")

tg_router.include_router(users_router)
tg_router.include_router(inference_api_router)