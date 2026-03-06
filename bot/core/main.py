import asyncio

from aiogram import Dispatcher
from aiogram.filters import Command
from aiohttp import ClientSession
from redis.asyncio import Redis

from bot.config import bot, api_base_url, env, redis_settings
from bot.core.utils.aio_http2api_server import ApiServerConn

from bot.core.handlers.start import helping, on_startup, start_handler

dp = Dispatcher()


async def main():
    """"""
    "AioHttp"
    aio_http_session = ClientSession(
        base_url=api_base_url,
        headers={'X-Auth-Service': env.auth_api_service_secret}
    )
    "Redis"
    redis_conn = Redis(**redis_settings)

    "Команды"
    dp.message.register(start_handler, Command('start'))
    dp.message.register(helping, Command('help'))

    dp.startup.register(on_startup)
    try:
        await dp.start_polling(
            bot,
            allowed_updates=dp.resolve_used_update_types(),
            aio_http=ApiServerConn(aio_http_session),
            redis=redis_conn
        )
    finally:
        await aio_http_session.close()
        await redis_conn.close()



if __name__ == '__main__':
    asyncio.run(main())