import io

import cv2
import torch
import numpy as np
import albumentations as A
from PIL import Image
from albumentations.pytorch import ToTensorV2
from cvat_sdk.api_client import ApiClient, Configuration
from tqdm import tqdm

from ml.config import WORKDIR, env
from ml.crop_refiner.models import Extent2CoreRefiner
from ml.logger_config import log_event

# --- НАСТРОЙКИ ---
cvat_url = 'http://localhost:8080'
task_id = 7
model_path = WORKDIR / 'ml' / 'crop_refiner' / 'model_weights' / 'cvat_weights' / 'crop_refiner11.pt'

transforms = A.Compose([
    A.LongestMaxSize(max_size=128),
    A.PadIfNeeded(min_height=128, min_width=128, border_mode=cv2.BORDER_CONSTANT, fill=(255, 255, 255)),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


def preprocess_crop(img_rgb):
    return transforms(image=img_rgb)['image'].unsqueeze(0).to(env.device)

# Загружаем модель
model = Extent2CoreRefiner().to(env.device)
model_weights = torch.load(model_path, map_location=env.device, weights_only=False)['model_state_dict']
model.load_state_dict(model_weights)
model.eval()


with ApiClient(Configuration(host=cvat_url, username=env.cvat_admin_username, password=env.cvat_admin_passw)) as client:
    meta = client.tasks_api.retrieve_data(task_id, type="frame")

    # Создаем маппинг {frame_id: filename}
    id_to_filename = {i: frame.name for i, frame in enumerate(meta.frames)}

    # 2. Получаем аннотации
    annotations = client.tasks_api.retrieve_annotations(task_id)

    # Определяем, какие файлы мы хотим обработать (например, только конкретного автора)
    # Можно фильтровать по подстроке в пути: "author_001" или по списку
    TARGET_SUBSTRING = "14"  # Пример фильтра

    shapes_by_frame = {}
    for shape in annotations.shapes:
        if shape.type.value == 'rectangle':
            frame_name = id_to_filename[shape.frame]

            # ФИЛЬТРАЦИЯ: обрабатываем только если имя файла подходит
            if TARGET_SUBSTRING not in frame_name:
                shapes_by_frame.setdefault(shape.frame, []).append(shape)

    print(f"Найдено {len(shapes_by_frame)} подходящих кадров.")

    for frame_id, shapes in shapes_by_frame.items():
        # ИСПРАВЛЕННЫЙ МЕТОД: в 2.x используем retrieve_data
        # type="frame" и quality="original" (или "compressed")
        frame_data, _ = client.tasks_api.retrieve_data(
            id=task_id,
            type="frame",
            number=frame_id,
            quality="original"
        )

        # Читаем байты (retrieve_data возвращает BufferedReader)
        image = Image.open(io.BytesIO(frame_data.read()))
        image = np.array(image.convert('RGB'))
        h_img, w_img = image.shape[:2]

        for shape in shapes:
            # shape.points в SDK — это список [x1, y1, x2, y2]
            x1, y1, x2, y2 = shape.points

            crop = image[int(max(0, y1)):int(min(h_img, y2)), int(max(0, x1)):int(min(w_img, x2))]
            if crop.size == 0: continue

            h_ext, w_ext = crop.shape[:2]

            # 3. Инференс
            input_tensor = preprocess_crop(crop)
            with torch.no_grad():
                pred = model(input_tensor).cpu().numpy()[0]
                nx, ny, nw, nh = pred

            # 4. Пересчет координат
            abs_cx = x1 + (nx * w_ext)
            abs_cy = y1 + (ny * h_ext)
            abs_w = nw * w_ext
            abs_h = nh * h_ext

            # Обновляем точки в объекте
            shape.points = [
                abs_cx - abs_w / 2,
                abs_cy - abs_h / 2,
                abs_cx + abs_w / 2,
                abs_cy + abs_h / 2
            ]

    # 5. Пушим обновление. В v2.x используем PATCH или PUT
    # Нам нужно передать объект LabeledData обратно
    client.tasks_api.update_annotations(task_id, annotations)

    log_event("Готово! Пора в CVAT", level='WARNING')