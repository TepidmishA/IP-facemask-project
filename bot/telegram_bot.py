import asyncio
import os
from tempfile import NamedTemporaryFile

import requests
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    ApplicationBuilder,
    CallbackContext,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    filters,
)

import dotenv
for env_path in ['.env', '../.env', '../../.env']:
    if os.path.exists(env_path):
        dotenv.load_dotenv(env_path)
        break

DEFAULT_MODEL = "dl"
SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000/predict")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")


def build_keyboard():
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("DL", callback_data="dl"),
                InlineKeyboardButton("Classical", callback_data="classical"),
                InlineKeyboardButton("Third", callback_data="third"),
            ]
        ]
    )


async def start(update: Update, context: CallbackContext):
    context.user_data["model"] = DEFAULT_MODEL
    text = (
        "Привет! Отправьте фото, а я скажу, в маске человек или нет.\n"
        "Команда /model <dl|classical|third> — выбрать модель.\n"
        "По умолчанию используется DL (transfer learning)."
    )
    await update.message.reply_text(text, reply_markup=build_keyboard())


async def help_cmd(update: Update, context: CallbackContext):
    await update.message.reply_text(
        "Отправьте фотографию лица. Можно выбрать модель:\n"
        "/model dl — трансферное обучение\n"
        "/model classical — HOG + логрег\n"
        "/model third — ResNet18 embeddings + логрег"
    )


async def set_model(update: Update, context: CallbackContext):
    if not context.args:
        await update.message.reply_text("Укажите модель: dl, classical или third.")
        return
    model = context.args[0].strip().lower()
    if model not in {"dl", "classical", "third"}:
        await update.message.reply_text("Допустимые значения: dl, classical, third.")
        return
    context.user_data["model"] = model
    await update.message.reply_text(f"Модель установлена: {model}")


async def button_handler(update: Update, context: CallbackContext):
    query = update.callback_query
    await query.answer()
    model = query.data
    context.user_data["model"] = model
    await query.edit_message_text(text=f"Модель установлена: {model}")


def call_server(image_path: str, model: str):
    with open(image_path, "rb") as f:
        resp = requests.post(SERVER_URL, params={"model": model}, files={"image": f}, timeout=30)
    resp.raise_for_status()
    return resp.json()


async def handle_photo(update: Update, context: CallbackContext):
    model = context.user_data.get("model", DEFAULT_MODEL)
    if not TELEGRAM_TOKEN:
        await update.message.reply_text("Переменная TELEGRAM_TOKEN не установлена.")
        return
    photo = update.message.photo[-1]
    file = await photo.get_file()
    # On Windows, NamedTemporaryFile keeps the handle open; close before writing.
    tmp = NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        await file.download_to_drive(tmp_path)
        result = await asyncio.get_event_loop().run_in_executor(None, call_server, tmp_path, model)
        prob = result.get("probability", 0) * 100
        prediction = result.get("prediction", "unknown")
        if prediction in {"masked", "with_mask"}:
            text = f"Человек в маске ✅ (confidence: {prob:.1f}%)"
        else:
            text = f"Человек без маски ❌ (confidence: {prob:.1f}%)"
        await update.message.reply_text(text)
    except Exception as exc:
        await update.message.reply_text(f"Ошибка: {exc}")
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def main():
    if not TELEGRAM_TOKEN:
        raise RuntimeError("Set TELEGRAM_TOKEN environment variable.")
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("model", set_model))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_photo))
    app.add_handler(CallbackQueryHandler(button_handler))
    print("Bot started. Press Ctrl+C to stop.")
    app.run_polling()


if __name__ == "__main__":
    main()

