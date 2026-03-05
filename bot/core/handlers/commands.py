from aiogram.types import BotCommand, BotCommandScopeDefault
from aiogram import Bot


async def set_commands(bot: Bot):
    commands = [
        BotCommand(command='/start', description='Запуск бота'),
        BotCommand(command='/profile', description='статистика и уровень подписки'),
        BotCommand(command='/history', description='История изображений'),
        BotCommand(command='/help', description='Помощь')
    ]

    await bot.set_my_commands(commands, BotCommandScopeDefault())