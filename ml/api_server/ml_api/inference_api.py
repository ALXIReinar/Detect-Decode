from fastapi import APIRouter, UploadFile
from fastapi.params import Depends

from ml.api_server.di_injections import get_auth_service, OCRDep

router = APIRouter(prefix="/inference/ocr", tags=["🔮Inference"])


@router.post('/en', dependencies=[Depends(get_auth_service)])
async def img2text(img: UploadFile, model: OCRDep):
    # 1. Инференс, упаковка вырезок слов с изображения и самого изображения в архив
    res = model.forward_pass(...)
    # 2. Брокер отправляет в фон задачу на выгрузку архива в С3(В очередь)
    # 3. Консумер ловит событие, отправляет в облако архив
    return res