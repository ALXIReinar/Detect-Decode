from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery

from bot.core.utils.aio_http2api_server import ApiServerConn
from bot.core.utils.keyboards import inference_feedback_show_user_rate


async def callback_factory(call: CallbackQuery, state: FSMContext, aio_http: ApiServerConn):
    call_data = call.data

    if call_data.startswith('inference-rate'):
        *_, img_id, msg_id, rate = call_data.split('_')

        await aio_http.rate_img_inference(img_id, rate) # строки, т.к Pydantic справится
        await call.message.edit_reply_markup(reply_markup=inference_feedback_show_user_rate(rate))
        await call.answer('Спасибо за Ваш ответ! <3')

    elif call_data.startswith('history-next'):
        # нажатие на стрелочку ">>>", перелистнуть страницу
        ...

    elif call_data.startswith('history-prev'):
        # нажатие на стрелочку "<<<", перелистнуть страницу
        ...

    elif call_data.startswith('history'):
        # по задумке, "отобразить сообщение с фоткой, текстом, оценкой пользователя(если есть), время"
        ...

    await call.answer()