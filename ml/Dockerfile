FROM python:3.12-slim as builder

COPY requirements.txt .
RUN pip3 install --no-cache-dir --user -r requirements.txt


FROM python:3.12-slim

RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app
COPY --from=builder /root/.local /home/appuser/.local
COPY --chown=appuser:appuser . /app
COPY --chown=appuser:appuser ./secrets/certs/* /usr/local/share/ca-certificates

ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONPATH=/app

USER appuser

EXPOSE 8000

CMD ["python", "ml/api_layer/main.py"]