from functools import wraps
from redis.asyncio import Redis
from aiogram.types import Message

from bot.core.utils.anything import RedisKeys


def rate_limit(max_requests: int = 25, window_seconds: int = 60):
    """
    Декоратор для ограничения пользовательских запросов на APIService. Ограничение происходит по tg_id
    Использовать на бот-хэндлерах с aio_http соединением
    ВАЖНО! Необходимо указывать аргумент redis в бот-хендлере с этим декоратором

    Пример использования:
        @rate_limit(max_requests=5, window_seconds=60)
        async def my_handler(message: Message, ...):
            ...
    """

    def decorator(handler):
        @wraps(handler)
        async def wrapper(message: Message, redis: Redis, *args, **kwargs):
            key = RedisKeys.rate_limit(message.from_user.id)

            "Редис транзакция"
            pipe = await redis.pipeline()

            pipe.incr(key)
            pipe.expire(key, window_seconds)

            results = await pipe.execute()

            current_count = results[0] # ответ от инкремента
            if current_count > max_requests:
                await message.answer(f"⏳ Слишком много запросов. Подождите некоторое время")
                return

            return await handler(message, redis, *args, **kwargs)
        return wrapper
    return decorator
