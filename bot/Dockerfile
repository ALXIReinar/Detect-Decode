FROM python:3.12-slim as builder

COPY requirements.txt ./requirements.txt


RUN pip3 install --no-cache-dir --user -r requirements.txt


FROM python:3.12-slim


RUN groupadd -r appuser && useradd -r -m -g  appuser appuser -d /home/appuser

WORKDIR /app

COPY --from=builder /root/.local /home/appuser/.local
COPY --chown=appuser:appuser . /app/bot/

RUN chown -R appuser:appuser /home/appuser /app

ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONPATH=/app

USER appuser

CMD ["python", "-m", "bot.core.main"]