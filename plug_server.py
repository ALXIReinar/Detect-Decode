from typing import Annotated

from starlette.requests import Request

import secrets

from fastapi import FastAPI, HTTPException, Body

app = FastAPI()

@app.get("/health")
async def health(request: Request):
    secrets_header = request.headers.get("X-Auth-Service", '')
    if not secrets.compare_digest(secrets_header, "A*9aTs88&2hFHFWPsTw6dtA(77a*7989AIS"):
        raise HTTPException(status_code=403, detail="Service.exe")
    return {"status": True}


@app.post("/api/v1/users/add")
async def add_user(
        tg_id: Annotated[int, Body()],
        is_created: Annotated[bool, Body()],
):
    print(tg_id)
    print(is_created)
    return {'success': True, 'message': 'Пользователь из ТГ сохранён'}