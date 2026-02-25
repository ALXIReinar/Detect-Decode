# Использование test_stage.py

## 🚀 Быстрый старт:

### Базовый запуск:
```bash
python ml/test_stage.py --weights ml/model_weights/detector/production_weights/model_detector47.pth
```

### С короткими флагами:
```bash
python ml/test_stage.py -w ml/model_weights/detector/production_weights/model_detector47.pth
```

## 📝 Все параметры:

### Обязательные:
```bash
--weights, -w    Путь к файлу весов модели (.pth)
```

### Опциональные:
```bash
--batch-size, -b    Размер батча (default: 4)
--img-size, -i      Размер изображения (default: 1280)
--workers, -j       Количество workers (default: 6)
--conf-thres        Порог confidence для NMS (default: 0.25)
--iou-thres         Порог IoU для NMS (default: 0.45)
```

## 💡 Примеры использования:

### 1. Базовое тестирование:
```bash
python ml/test_stage.py -w ml/model_weights/detector/20240125-143022/model_detector47.pth
```

### 2. С кастомным batch size:
```bash
python ml/test_stage.py -w path/to/weights.pth -b 8
```

### 3. С другим разрешением:
```bash
python ml/test_stage.py -w path/to/weights.pth -i 640
```

### 4. Полная кастомизация:
```bash
python ml/test_stage.py \
    --weights ml/model_weights/detector/best_model.pth \
    --batch-size 2 \
    --img-size 1280 \
    --workers 4 \
    --conf-thres 0.3 \
    --iou-thres 0.5
```

### 5. По SSH на сервере:
```bash
# Абсолютный путь
python ml/test_stage.py -w /home/user/project/ml/model_weights/detector/model.pth

# Относительный путь (от WORKDIR)
python ml/test_stage.py -w ml/model_weights/detector/model.pth
```

## 🔍 Справка:

```bash
python ml/test_stage.py --help
```

Выведет:
```
usage: test_stage.py [-h] --weights WEIGHTS [--batch-size BATCH_SIZE]
                     [--img-size IMG_SIZE] [--workers WORKERS]
                     [--conf-thres CONF_THRES] [--iou-thres IOU_THRES]

Тестирование модели детектора OCR

optional arguments:
  -h, --help            show this help message and exit
  --weights WEIGHTS, -w WEIGHTS
                        Путь к файлу весов модели (.pth) (default: None)
  --batch-size BATCH_SIZE, -b BATCH_SIZE
                        Размер батча для тестирования (default: 4)
  --img-size IMG_SIZE, -i IMG_SIZE
                        Размер входного изображения (640, 960, 1280, etc.)
                        (default: 1280)
  --workers WORKERS, -j WORKERS
                        Количество workers для DataLoader (default: 6)
  --conf-thres CONF_THRES
                        Порог confidence для NMS (default: 0.25)
  --iou-thres IOU_THRES
                        Порог IoU для NMS (default: 0.45)
```

## 📊 Вывод:

Скрипт выведет:
```
================================================================================
ТЕСТИРОВАНИЕ ЗАВЕРШЕНО
================================================================================
Веса: model_detector47.pth
Test Loss: 2.2941
  - Box Loss: 0.4298
  - Cls Loss: 0.3883
  - DFL Loss: 0.9025
mAP@0.5: 0.9470 (94.70%)
mAP@0.5:0.95: 0.8826 (88.26%)
================================================================================
```

## 🎯 Советы:

### Для быстрого тестирования:
```bash
python ml/test_stage.py -w path/to/weights.pth -b 8 -j 8
```

### Для экономии памяти:
```bash
python ml/test_stage.py -w path/to/weights.pth -b 2 -j 2
```

### Для более строгого NMS:
```bash
python ml/test_stage.py -w path/to/weights.pth --conf-thres 0.5 --iou-thres 0.3
```
