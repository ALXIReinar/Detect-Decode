from dataclasses import dataclass
from typing import Literal


class RedisKeys:
    @staticmethod
    def media_lock(tg_id: str | int) -> str:
        return f'media_lock:{tg_id}'

    @staticmethod
    def media_group(media_group_id: str | int) -> str:
        return f'media_group:{media_group_id}'

    @staticmethod
    def rate_limit(tg_id: str | int) -> str:
        return f'rate_limit:{tg_id}'


def post_processing_text(text: str, lang: Literal['ru', 'en', 'ru-en']) -> str:
    """
    После точек делает буквы заглавными.
    Добавляет знаки препинания по предлогам
    """
    "Заглавные после точки"
    sentences = text.split('. ')
    sentences = [s.capitalize() for s in sentences]
    text = '. '.join(sentences)

    def postprocess_russian(raw_text: str) -> str:
        """"""
        "Запятые перед союзами"
        conjunctions = {'что', 'чтобы', 'если', 'когда', 'потому', 'хотя', 'как'}
        for word in conjunctions:
            raw_text = raw_text.replace(f' {word}', f', {word}')

        return raw_text

    def postprocess_english(raw_text: str) -> str:
        """"""
        # какие-то обработки для английского текста
        return raw_text

    def hybrid(raw_text: str) -> str:
        ru_text = postprocess_russian(raw_text)
        ru_en_text = postprocess_english(ru_text)
        return ru_en_text

    proc_type = {
        'ru': postprocess_russian,
        'en': postprocess_english,
        'ru-en': hybrid,
    }
    return proc_type[lang](text)


@dataclass
class MessageTemplates:
    start_msg: str = '''
Добро пожаловать, {}!

Этот бот может распознавать текст с картинок!
Отправьте боту фото, выберите на каком языке изображён текст и дождитесь обработки изображения

/profile - Покажет статистику и статус подписки
/help - Помощь
/history - Выдаст меню с историей изображений
'''

    help_msg: str = '''
Помощь.ехе

СКОРО!

Если изображение слишком тяжёлое и оно в .png, переведите его в .jpg формат через онлайн-конвертер или через этого бота /png2jpeg
Также Вы можете переводить фотографии из HEIC разрешения прямо у нас:) /heic2jpeg 
        '''