from pydantic import BaseModel


class TgUserSchema(BaseModel):
    tg_id: int
    first_name: str
    last_name: str
    is_created: bool