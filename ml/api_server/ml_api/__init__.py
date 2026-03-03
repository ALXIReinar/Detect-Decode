from fastapi import APIRouter

main_router = APIRouter(prefix="/api/v1")


@main_router.get("/healthcheck")
async def healthcheck():
    return {"status": True, 'service': 'ml-server'}
