import nest_asyncio
import os
import logging
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ConversationHandler,
    ContextTypes,
    filters,
)
from PIL import Image
import io
from src.models.image_editor import DiffusionImageEditor
from src.models.asr_model import ASRModelWrapper

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

editor = DiffusionImageEditor(
    model_type="instruct_pix2pix",
    model_name="timbrooks/instruct-pix2pix",
    device="cuda",
    num_inference_steps=20,
    strength=0.75,
    image_guidance_scale=1.5,
    guidance_scale=7.5,
    max_side=1024,
)
asr = ASRModelWrapper(model_type="tone", device="cuda")

WAITING_IMAGE = 0
WAITING_AUDIO = 1


def process_image_with_audio(image_path: str, audio_path: str) -> str:
    # заглушка - просто копируем картинку, чуть позже обернем в аутпут модели 
    output_path = image_path.replace("_input", "_output")
    if output_path == image_path:
        output_path = image_path.rsplit(".", 1)[0] + "_edited." + image_path.rsplit(".", 1)[1]
    
    with Image.open(image_path) as img:
        print("Старт распознавания из пути: ", audio_path)
        audio_text = asr.transcribe(audio_path)
        print(audio_text)
        print("Старт редактирования")
        result_img = editor.edit(image=img, instruction=audio_text)
        print("Конец редактирования")
        result_img.save(output_path)
    
    return output_path


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        "Привет! Я бот для редактирования изображений по голосовым инструкциям.\n\n Отправь мне картинку, которую хочешь отредактировать."
    )
    return WAITING_IMAGE


async def receive_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.effective_user
    
    photo = update.message.photo[-1]
    file = await photo.get_file()
    
    # Сохраняем изображение
    os.makedirs("temp", exist_ok=True)
    image_path = f"temp/{user.id}_input.jpg"
    await file.download_to_drive(image_path)
    
    # Сохраняем путь в контексте
    context.user_data["image_path"] = image_path
    
    await update.message.reply_text(
        "Картинка получена!\n\n Теперь отправь голосовое сообщение или аудиофайл с инструкциями, что нужно изменить на картинке."
    )
    return WAITING_AUDIO


async def receive_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.effective_user
    
    if update.message.voice:
        audio = update.message.voice
        audio_ext = "ogg"
    elif update.message.audio:
        audio = update.message.audio
        audio_ext = "mp3"
    else:
        await update.message.reply_text(
            "Пожалуйста, отправь голосовое сообщение или аудиофайл."
        )
        return WAITING_AUDIO
    
    file = await audio.get_file()
    audio_path = f"temp/{user.id}_audio.{audio_ext}"
    await file.download_to_drive(audio_path)

    image_path = context.user_data.get("image_path")
    
    if not image_path or not os.path.exists(image_path):
        await update.message.reply_text(
            " Изображение не найдено. Пожалуйста, начни сначала с /start"
        )
        return ConversationHandler.END
    
    await update.message.reply_text("Обрабатываю изображение...")
    
    try:
        # Обрабатываем изображение
        output_path = process_image_with_audio(image_path, audio_path)
        
        # Отправляем результат
        with open(output_path, "rb") as photo_file:
            await update.message.reply_photo(
                photo=photo_file,
                caption=" Готово! Вот отредактированное изображение."
            )
        
        # Очищаем временные файлы
        for path in [image_path, audio_path, output_path]:
            if os.path.exists(path):
                os.remove(path)
        
    except Exception as e:
        logger.error(f"Ошибка обработки: {e}")
        await update.message.reply_text(
            f"Произошла ошибка при обработке: {e}"
        )
    
    # Очищаем данные пользователя
    context.user_data.clear()
    
    await update.message.reply_text(
        "🔄 Хочешь отредактировать ещё одну картинку? Отправь /start"
    )
    return ConversationHandler.END


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    # Очищаем временные файлы
    image_path = context.user_data.get("image_path")
    if image_path and os.path.exists(image_path):
        os.remove(image_path)
    
    context.user_data.clear()
    
    await update.message.reply_text(
        "Операция отменена. Отправь /start чтобы начать заново."
    )
    return ConversationHandler.END


async def unexpected_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка неожиданных сообщений."""
    await update.message.reply_text(
        "Не понимаю. Отправь /start чтобы начать редактирование картинки."
    )


def main() -> None:

    TELEGRAM_BOT_TOKEN ='***'
    
    token = TELEGRAM_BOT_TOKEN
    application = Application.builder().token(token).build()
    
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            WAITING_IMAGE: [
                MessageHandler(filters.PHOTO, receive_image),
            ],
            WAITING_AUDIO: [
                MessageHandler(filters.VOICE | filters.AUDIO, receive_audio),
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )
    
    application.add_handler(conv_handler)
    application.add_handler(MessageHandler(filters.ALL, unexpected_message))
    
    print("Бот запущен")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
