from pydantic import BaseModel, Field, field_validator
from pydantic_core.core_schema import ValidationInfo


class TgImgSchema(BaseModel):
    tg_id: int
    tg_file_path_list: list[str] = Field(..., alias='file_path_list')
    file_ids: list[str]

    @field_validator('file_ids', mode='after')
    @classmethod
    def file_paths_file_ids_equals(cls, v, info: ValidationInfo):
        tg_f_path_len = len(v)
        f_ids_len = len(info.data['tg_file_path_list'])

        "Количество file_id д.б. == количество file_path"
        if tg_f_path_len != f_ids_len :
            return ValueError(f'Количество tg_file_path_list({tg_f_path_len}) должно быть равно file_ids({f_ids_len})')

        return v

class ImgOcrRateSchema(BaseModel):
    img_id: int
    rate: int = Field(ge=1, le=5) # Оценка от 1 до 5