import tempfile
from pathlib import Path

import cv2
import torch
import numpy as np
import albumentations as A
from PIL import Image
from albumentations.pytorch import ToTensorV2
from cvat_sdk import make_client
from tqdm import tqdm

from ml.config import WORKDIR, env
from ml.crop_refiner.models import Extent2CoreMobileNetRefiner, Extent2CoreResnetRefiner
from ml.logger_config import log_event

# --- НАСТРОЙКИ ---
cvat_url = 'http://localhost:8080'
job_id = 2  # Используем job_id вместо task_id
model_path = WORKDIR / 'ml' / 'crop_refiner' / 'model_weights' / 'cvat_weights' / 'crop_refiner14.pt'

transforms = A.Compose([
    A.Resize(height=128, width=512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


def preprocess_crop(img_rgb):
    return transforms(image=img_rgb)['image'].unsqueeze(0).to(env.device)

# Загружаем модель
log_event("Загрузка модели...")
model = Extent2CoreResnetRefiner().to(env.device)
model_weights = torch.load(model_path, map_location=env.device, weights_only=False)['model_state_dict']
model.load_state_dict(model_weights)
model.eval()


# Используем High-level API (cvat_sdk 2.x)
with make_client(cvat_url, credentials=(env.cvat_admin_username, env.cvat_admin_passw)) as client:
    # Получаем job объект вместо task
    job = client.jobs.retrieve(job_id)
    log_event(f"Job: {job.id}, Task: {job.task_id}, Frames: {job.start_frame}-{job.stop_frame}")
    
    # Получаем метаданные для маппинга frame_id -> filename
    meta = job.get_meta()
    id_to_filename = {i: frame.name for i, frame in enumerate(meta.frames)}
    
    # Получаем аннотации job'а
    annotations = job.get_annotations()
    log_event(f"Всего shapes: {len(annotations.shapes)}")
    
    # Фильтр по названию файла
    target_substring = "HWR200/simplified/"  # Пример фильтра

    shapes_by_frame = {}
    filtered_shapes_count = 0
    for shape in annotations.shapes:
        if shape.type.value == 'rectangle':
            frame_name = id_to_filename[shape.frame]

            # ФИЛЬТРАЦИЯ: обрабатываем только если имя файла подходит
            if target_substring in frame_name:
                shapes_by_frame.setdefault(shape.frame, []).append(shape)
                filtered_shapes_count += 1
    
    log_event(f"Найдено {len(shapes_by_frame)} подходящих кадров после фильтрации")
    log_event(f"Будет обработано {filtered_shapes_count} shapes из {len(annotations.shapes)} всего")
    
    # Создаём временную директорию для скачивания фреймов
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Скачиваем нужные фреймы
        frame_ids = list(shapes_by_frame.keys())
        log_event(f"Скачивание {len(frame_ids)} фреймов...")
        job.download_frames(
            frame_ids=frame_ids,
            outdir=temp_path,
            quality="original"
        )
        
        # Проверяем, что реально скачалось
        downloaded_files = list(temp_path.glob("*"))
        log_event(f"Скачано файлов: {len(downloaded_files)}")
        if downloaded_files:
            log_event(f"Примеры скачанных файлов: {[f.name for f in downloaded_files[:3]]}")
        
        # Создаём маппинг frame_id -> скачанный файл
        # download_frames сохраняет файлы как frame_<id>.<ext>
        frame_id_to_file = {}
        for file_path in downloaded_files:
            # Пытаемся извлечь frame_id из имени файла
            # Формат может быть: frame_000123.jpg или просто 123.jpg
            stem = file_path.stem
            if stem.startswith('frame_'):
                try:
                    fid = int(stem.split('_')[1])
                    frame_id_to_file[fid] = file_path
                except (ValueError, IndexError):
                    pass
            else:
                # Пробуем просто как число
                try:
                    fid = int(stem)
                    frame_id_to_file[fid] = file_path
                except ValueError:
                    pass
        
        log_event(f"Распознано frame_id для {len(frame_id_to_file)} файлов")
        
        # Обрабатываем каждый фрейм
        for frame_id, shapes in tqdm(shapes_by_frame.items(), desc="Processing frames"):
            # Ищем скачанный файл по frame_id
            if frame_id <= 1:
                continue

            if frame_id not in frame_id_to_file:
                frame_filename = id_to_filename[frame_id]
                log_event(f"Файл для frame_id={frame_id} ({frame_filename}) не найден в скачанных", level='WARNING')
                continue
            
            frame_path = frame_id_to_file[frame_id]
            
            image = Image.open(frame_path).convert('RGB')
            image = np.array(image)
            h_img, w_img = image.shape[:2]
            
            for shape in shapes:
                # shape.points в SDK — это список [x1, y1, x2, y2]
                x1, y1, x2, y2 = shape.points
                
                # log_event(f"Frame {frame_id}, Shape ID {shape.id}: До обработки points = {shape.points}")
                
                crop = image[int(max(0, y1)):int(min(h_img, y2)), int(max(0, x1)):int(min(w_img, x2))]
                if crop.size == 0:
                    continue
                
                h_ext, w_ext = crop.shape[:2]
                
                # Инференс
                input_tensor = preprocess_crop(crop)
                with torch.no_grad():
                    pred = model(input_tensor).cpu().numpy()[0]
                    nx, ny, nw, nh = pred
                
                # log_event(f"Предсказание модели: cx={nx:.4f}, cy={ny:.4f}, w={nw:.4f}, h={nh:.4f}")
                
                # Пересчет координат
                abs_cx = x1 + (nx * w_ext)
                abs_cy = y1 + (ny * h_ext)
                abs_w = nw * w_ext
                abs_h = nh * h_ext
                
                # Обновляем точки в объекте
                float_points = list(map(float, [
                    abs_cx - abs_w / 2,
                    abs_cy - abs_h / 2,
                    abs_cx + abs_w / 2,
                    abs_cy + abs_h / 2
                ]))
                
                # log_event(f"После обработки points = {float_points}")
                # log_event(f"Изменение: Δx1={float_points[0]-x1:.2f}, Δy1={float_points[1]-y1:.2f}, Δx2={float_points[2]-x2:.2f}, Δy2={float_points[3]-y2:.2f}")
                
                shape.points = float_points
    
    # Обновляем аннотации на сервере (заменяем все аннотации)
    log_event("Обновление аннотаций на сервере...")
    log_event(f"Всего shapes для отправки: {len(annotations.shapes)}")
    log_event(f"Всего tags для отправки: {len(annotations.tags)}")
    log_event(f"Всего tracks для отправки: {len(annotations.tracks)}")
    
    # Конвертируем LabeledData в LabeledDataRequest
    from cvat_sdk import models
    
    # Преобразуем shapes в LabeledShapeRequest
    shapes_request = []
    for shape in annotations.shapes:
        shapes_request.append(
            models.LabeledShapeRequest(
                type=shape.type,
                frame=shape.frame,
                label_id=shape.label_id,
                points=shape.points,
                attributes=shape.attributes if hasattr(shape, 'attributes') else [],
                occluded=shape.occluded if hasattr(shape, 'occluded') else False,
                outside=shape.outside if hasattr(shape, 'outside') else False,
                z_order=shape.z_order if hasattr(shape, 'z_order') else 0,
                rotation=shape.rotation if hasattr(shape, 'rotation') else 0.0,
            )
        )
    
    # Преобразуем tags в LabeledImageRequest
    tags_request = []
    for tag in annotations.tags:
        tags_request.append(
            models.LabeledImageRequest(
                frame=tag.frame,
                label_id=tag.label_id,
                attributes=tag.attributes if hasattr(tag, 'attributes') else []
            )
        )
    
    # Преобразуем tracks в LabeledTrackRequest
    tracks_request = []
    for track in annotations.tracks:
        track_shapes = []
        for shape in track.shapes:
            track_shapes.append(
                models.TrackedShapeRequest(
                    type=shape.type,
                    frame=shape.frame,
                    points=shape.points,
                    attributes=shape.attributes if hasattr(shape, 'attributes') else [],
                    occluded=shape.occluded if hasattr(shape, 'occluded') else False,
                    outside=shape.outside if hasattr(shape, 'outside') else False,
                    z_order=shape.z_order if hasattr(shape, 'z_order') else 0,
                    rotation=shape.rotation if hasattr(shape, 'rotation') else 0.0,
                )
            )
        tracks_request.append(
            models.LabeledTrackRequest(
                label_id=track.label_id,
                frame=track.frame,
                shapes=track_shapes,
                attributes=track.attributes if hasattr(track, 'attributes') else []
            )
        )
    
    # Создаём LabeledDataRequest
    labeled_data_request = models.LabeledDataRequest(
        shapes=shapes_request,
        tags=tags_request,
        tracks=tracks_request
    )
    
    log_event(f"Отправка {len(shapes_request)} shapes на сервер...")
    
    try:
        job.set_annotations(labeled_data_request)
        log_event("✓ Аннотации успешно обновлены на сервере!", level='INFO')
    except Exception as e:
        log_event(f"✗ Ошибка при обновлении аннотаций: {e}", level='ERROR')
        raise
    
    log_event("Готово! Обновите страницу в CVAT (Ctrl+R или F5)", level='INFO')