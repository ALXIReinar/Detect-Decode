from aiogram.types import ReplyKeyboardRemove, Message

from bot.config import bot, env
from bot.core.handlers.commands import set_commands
from bot.core.utils.aio_http2api_server import ApiServerConn
from bot.core.utils.anything import MessageTemplates
from bot.logger_config import log_event


async def on_startup():
    log_event('Бот запущен', level='WARNING')
    await bot.send_message(env.admin_id, 'Бот запущен!', reply_markup=ReplyKeyboardRemove())

async def start_handler(message: Message, aio_http: ApiServerConn):
    """Запрос на сохранение пользователя + Приветствие"""
    f_name = message.from_user.first_name

    await aio_http.save_user(message.from_user.id)

    await message.answer(MessageTemplates.start_msg.format(f_name))
    await set_commands(bot)


async def helping(message: Message):
    await message.answer(MessageTemplates.help_msg)