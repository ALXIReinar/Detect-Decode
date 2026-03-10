from fastapi import APIRouter, HTTPException
from starlette.requests import Request

from web.data.postgres import PgSqlDep
from web.schemas.users_schema import TgUserSchema
from web.utils.logger_config import log_event

router = APIRouter(prefix="/users")


@router.post("/add")
async def add_user(body: TgUserSchema, request: Request, db: PgSqlDep):
    user_id, insert_flag = await db.users.add_tg_user(body.tg_id, body.first_name, body.last_name, body.is_created)
    if insert_flag:
        log_event('Пользователь Тг уже есть', )
        raise HTTPException(status_code=409, detail="Пользователь Телеграм с таким id уже существует")
    log_event(f'добавлен новый пользователь | user_id: \033[31m{user_id}\033[0m', request=request)
    return {"success": True, "message": "Пользователь добавлен"}