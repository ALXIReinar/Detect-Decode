import asyncio

from aiogram import Dispatcher
from aiogram.filters import Command

from bot.config import bot
from bot.core.handlers.start import helping, on_startup, start_handler

dp = Dispatcher()


async def main():
    """"""
    "Команды"
    dp.message.register(start_handler, Command('/start'))
    dp.message.register(helping, Command('/help'))

    dp.startup.register(on_startup)
    await dp.start_polling(bot)


if __name__ == '__main__':
    asyncio.run(main())