from aiogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
)


def cancel_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="/cancel")]],
        resize_keyboard=True,
        input_field_placeholder="Введите ответ или /cancel",
    )


def remove_keyboard() -> ReplyKeyboardRemove:
    return ReplyKeyboardRemove()


def consent_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Согласен", callback_data="consent:agree")],
            [InlineKeyboardButton(text="Не согласен", callback_data="consent:decline")],
        ]
    )


def confirm_delete_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Да, удалить мои данные", callback_data="delete:confirm")],
            [InlineKeyboardButton(text="Отмена", callback_data="delete:cancel")],
        ]
    )


def quiz_keyboard(options: list[str]) -> InlineKeyboardMarkup:
    """Варианты ответа на вопрос раздела: один клик вместо ввода текста."""
    rows = [
        [InlineKeyboardButton(text=option[:60], callback_data=f"quiz:{index}")]
        for index, option in enumerate(options[:4])
    ]
    return InlineKeyboardMarkup(inline_keyboard=rows)
