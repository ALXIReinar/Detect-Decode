FROM python:3.12-slim-bookworm as builder

COPY requirements.txt ./requirements.txt
COPY local_wheels/*.whl ./

# local_wheels - локальные файлы библиотек. Ввиду "ограничений", они не скачиваются по сети
# Сначала ставим torch, иначе ultralytics подхватит тяжёлую cuda версию
RUN pip3 install --no-cache-dir --user \
    torch-2.9.1+cpu-cp312-cp312-manylinux_2_28_x86_64.whl \
    torchvision-0.24.1+cpu-cp312-cp312-manylinux_2_28_x86_64.whl

RUN pip3 install --no-cache-dir --user -r requirements.txt


FROM python:3.12-slim-bookworm

# ставим недостающие пакеты для некоторых библиотек (torch, opencv)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r appuser && useradd -r -m -g appuser appuser

WORKDIR /app

COPY --from=builder /root/.local /home/appuser/.local
COPY --chown=appuser:appuser . /app/ml/

ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONPATH=/app

USER appuser

EXPOSE 8000

CMD ["python", "-m", "ml.api_layer.main"]