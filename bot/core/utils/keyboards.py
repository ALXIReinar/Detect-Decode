import random

from aiogram.utils.keyboard import InlineKeyboardBuilder

from bot.config import env


def inference_feedback(img_id: int):
    kb = InlineKeyboardBuilder()
    kb.button(text=f'Оцените результат от 1 до 5 {random.choice(env.inference_feedback_emoji)}', callback_data='None')
    kb.button(text='1', callback_data=f'inference-rate_{img_id}_1')
    kb.button(text='2', callback_data=f'inference-rate_{img_id}_2')
    kb.button(text='3', callback_data=f'inference-rate_{img_id}_3')
    kb.button(text='4', callback_data=f'inference-rate_{img_id}_4')
    kb.button(text='5', callback_data=f'inference-rate_{img_id}_5')

    kb.adjust(1, 5)
    return kb.as_markup()

def inference_feedback_show_user_rate(rate: int | str):
    kb = InlineKeyboardBuilder()
    kb.button(text=f'Ваша оценка: {rate}', callback_data='None')
    return kb.as_markup()