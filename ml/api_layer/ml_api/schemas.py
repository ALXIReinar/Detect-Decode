from pydantic import BaseModel, field_validator
from pydantic_core.core_schema import ValidationInfo


class ImgMetadataSchema(BaseModel):
    img_ids: list[int]

class S3SendSchema(ImgMetadataSchema):
    preds_results: list[str]

    @field_validator('preds_results', mode='after')
    @classmethod
    def validate_value(cls, v, info: ValidationInfo):
        if len(v) != len(info.data['img_ids']):
            raise ValueError('Количество img_ids должно быть столько же, сколько и preds_results')
        return v


