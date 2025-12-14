import asyncio
import logging
import os
from pathlib import Path
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
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger("bot")


def build_keyboard(selected: str = DEFAULT_MODEL):
    def label(model_code: str, title: str):
        if selected == "compare" and model_code == "compare":
            return f"✅ {title}"
        if selected == model_code:
            return f"✅ {title}"
        return title

    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(label("dl", "DL (MobileNetV2)"), callback_data="set:dl"),
                InlineKeyboardButton(label("classical", "Classical (HOG+LR)"), callback_data="set:classical"),
            ],
            [
                InlineKeyboardButton(label("third", "ResNet18+LR"), callback_data="set:third"),
                InlineKeyboardButton(label("compare", "Сравнить модели"), callback_data="compare"),
            ],
            [
                InlineKeyboardButton("Описание моделей", callback_data="info"),
            ],
        ]
    )


async def start(update: Update, context: CallbackContext):
    context.user_data["model"] = DEFAULT_MODEL
    text = (
        "Привет! Отправь фото лица, я скажу, есть ли маска.\n"
        "Выбирай модель кнопками или командой /model <dl|classical|third>.\n"
        "По умолчанию — DL (transfer learning).\n"
        "Кнопка «Все 3» прогонит фото через все модели."
    )
    await update.message.reply_text(text, reply_markup=build_keyboard(DEFAULT_MODEL))


async def help_cmd(update: Update, context: CallbackContext):
    await update.message.reply_text(
        "Отправь фото лица. Выбор модели:\n"
        "/model dl — MobileNetV2 transfer (по умолчанию)\n"
        "/model classical — HOG + логистическая регрессия\n"
        "/model third — ResNet18 эмбеддинги + логрег\n"
        "Или жми кнопки. «Все 3» — сравнение."
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
    pretty = {
        "dl": "DL (MobileNetV2)",
        "classical": "Classical (HOG+LR)",
        "third": "ResNet18+LR",
    }.get(model, model)
    await update.message.reply_text(f"Модель установлена: {pretty}", reply_markup=build_keyboard(model))


async def button_handler(update: Update, context: CallbackContext):
    query = update.callback_query
    await query.answer()
    data = query.data
    if data.startswith("set:"):
        model = data.split(":", 1)[1]
        context.user_data["model"] = model
        pretty = {
            "dl": "DL (MobileNetV2)",
            "classical": "Classical (HOG+LR)",
            "third": "ResNet18+LR",
        }.get(model, model)
        await query.edit_message_text(text=f"Модель установлена: {pretty}", reply_markup=build_keyboard(model))
    elif data == "compare":
        context.user_data["model"] = "compare"
        await query.edit_message_text(
            text="Режим сравнения: следующее изображение будет обработано всеми моделями, результаты вернутся списком.",
            reply_markup=build_keyboard("compare"),
        )
    elif data == "info":
        await query.edit_message_text(
            text=(
                "Общий препроцессинг:\n"
                "  • Выделение контуров лица\n"
                "  • Resize / Normalize\n"
                "\n"
                "DL (MobileNetV2):\n"
                "  • Признаки: Embedding MobileNetV2 (ImageNet)\n"
                "  • Нормализация: Batch Normalization (BN)\n"
                "  • Классификатор: Linear + Sigmoid (binary)\n"
                "\n"
                "Classical (HOG+LR):\n"
                "  • Признаки: HOG\n"
                "  • Нормализация: StandardScaler\n"
                "  • Классификатор: Logistic Regression\n"
                "\n"
                "ResNet18+LR:\n"
                "  • Признаки: Embedding ResNet18 (ImageNet)\n"
                "  • Нормализация: StandardScaler\n"
                "  • Классификатор: Logistic Regression"
            ),
            reply_markup=build_keyboard(context.user_data.get("model", DEFAULT_MODEL)),
        )


def call_server(image_path: str, model: str):
    url = SERVER_URL
    # If someone set https://localhost... but the server is HTTP, downgrade to http to avoid SSL errors.
    if url.startswith("https://localhost") or url.startswith("https://127.0.0.1"):
        url = "http://" + url.split("://", 1)[1]
    LOGGER.info("POST %s model=%s file=%s", url, model, image_path)
    with open(image_path, "rb") as f:
        # Disable proxies and SSL verify for local calls to avoid WRONG_VERSION_NUMBER issues.
        resp = requests.post(
            url,
            params={"model": model},
            files={"image": f},
            timeout=30,
            proxies={"http": None, "https": None},
            verify=False
        )
    LOGGER.info("Response status=%s text=%s", resp.status_code, resp.text[:500])
    resp.raise_for_status()
    return resp.json()


def compare_all_models(image_path: str):
    results = {}
    errors = {}
    for model in ["dl", "classical", "third"]:
        try:
            results[model] = call_server(image_path, model)
        except Exception as exc:
            LOGGER.exception("Compare failed for model=%s", model)
            errors[model] = str(exc)
    return results, errors


def format_result(result: dict) -> str:
    p_mask = float(result.get("probability_masked", result.get("probability", 0))) * 100
    p_no = float(result.get("probability_not_masked", max(0.0, 1.0 - result.get("probability_masked", 0)))) * 100
    pred = result.get("prediction", "unknown")
    verdict = "Маска" if pred in {"masked", "with_mask"} else "Без маски"
    return f"{verdict} | маска {p_mask:.1f}% · без маски {p_no:.1f}%"


def extract_error(exc: Exception) -> str:
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        try:
            data = exc.response.json()
            if isinstance(data, dict) and "detail" in data:
                return str(data["detail"])
        except Exception:
            pass
        return f"{exc.response.status_code} {exc.response.text}"
    return str(exc)


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
        LOGGER.info("Downloaded photo to %s", tmp_path)
        if model == "compare":
            results, errors = await asyncio.get_event_loop().run_in_executor(None, compare_all_models, tmp_path)
            lines = []
            for m in ["dl", "classical", "third"]:
                if m in results:
                    tag = {
                        "dl": "  DL",
                        "classical": "  Classical",
                        "third": "  ResNet18+LR",
                    }.get(m, m)
                    lines.append(f"{tag}: {format_result(results[m])}")
                else:
                    lines.append(f"{m}: error — {errors.get(m)}")
            await update.message.reply_text(
                "📊 Результаты сравнения:\n" + "\n".join(lines),
                reply_markup=build_keyboard("compare"),
            )
        else:
            try:
                result = await asyncio.get_event_loop().run_in_executor(None, call_server, tmp_path, model)
                text = format_result(result)
                await update.message.reply_text(text, reply_markup=build_keyboard(model))
            except Exception as exc:
                msg = extract_error(exc)
                LOGGER.exception("Failed single-model request: %s", msg)
                await update.message.reply_text(f"Ошибка: {msg}", reply_markup=build_keyboard(model))
    except Exception as exc:
        msg = extract_error(exc)
        LOGGER.exception("Failed to process photo: %s", msg)
        await update.message.reply_text(f"Ошибка: {msg}", reply_markup=build_keyboard(model))
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


async def handle_document(update: Update, context: CallbackContext):
    model = context.user_data.get("model", DEFAULT_MODEL)
    if not TELEGRAM_TOKEN:
        await update.message.reply_text("Переменная TELEGRAM_TOKEN не установлена.")
        return
    document = update.message.document
    if document is None:
        return
    mime = document.mime_type or ""
    name = document.file_name or ""
    if not (mime.startswith("image/") or name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))):
        await update.message.reply_text("Отправьте изображение (png/jpg).")
        return
    file = await document.get_file()
    tmp = NamedTemporaryFile(suffix=Path(name).suffix or ".jpg", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        await file.download_to_drive(tmp_path)
        LOGGER.info("Downloaded document to %s", tmp_path)
        if model == "compare":
            results, errors = await asyncio.get_event_loop().run_in_executor(None, compare_all_models, tmp_path)
            lines = []
            for m in ["dl", "classical", "third"]:
                if m in results:
                    tag = {
                        "dl": "DL",
                        "classical": "Classical",
                        "third": "ResNet18+LR",
                    }.get(m, m)
                    lines.append(f"{tag}: {format_result(results[m])}")
                else:
                    lines.append(f"{m}: error — {errors.get(m)}")
            await update.message.reply_text(
                "Результаты сравнения:\n" + "\n".join(lines),
                reply_markup=build_keyboard("compare"),
            )
        else:
            try:
                result = await asyncio.get_event_loop().run_in_executor(None, call_server, tmp_path, model)
                text = format_result(result)
                await update.message.reply_text(text, reply_markup=build_keyboard(model))
            except Exception as exc:
                msg = extract_error(exc)
                LOGGER.exception("Failed single-model request: %s", msg)
                await update.message.reply_text(f"Ошибка: {msg}", reply_markup=build_keyboard(model))
    except Exception as exc:
        msg = extract_error(exc)
        LOGGER.exception("Failed to process document: %s", msg)
        await update.message.reply_text(f"Ошибка: {msg}", reply_markup=build_keyboard(model))
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
    app.add_handler(MessageHandler(filters.Document.IMAGE, handle_document))
    app.add_handler(CallbackQueryHandler(button_handler))
    print("Bot started. Press Ctrl+C to stop.")
    app.run_polling()


if __name__ == "__main__":
    main()

