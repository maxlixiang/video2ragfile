import asyncio
import logging
import os
import re
import uuid
from urllib.parse import urlparse

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import BadRequest, NetworkError, RetryAfter, TimedOut
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from ai_services import generate_expert_knowledge_card, process_groq_transcription
from config import ALLOWED_USERS, SHARED_DIR, TELEGRAM_BASE_URL, TELEGRAM_LOCAL_MODE, TOKEN
from downloader import sanitize_filename_component, sync_download


logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)

URL_PATTERN = re.compile(r'https?://[^\s<>"\'\u3000\uff0c\u3002\uff01\uff1f\uff1b\uff1a\u3001\uff08\uff09\u3010\u3011\u300a\u300b]+', re.IGNORECASE)
URL_TRAILING_NOISE = '.,!?;:)]}>\'"\uff0c\u3002\uff01\uff1f\uff1b\uff1a\u3001\uff09\u3011\u300b'
DOUYIN_HOSTS = {"douyin.com", "www.douyin.com", "v.douyin.com", "iesdouyin.com"}
TELEGRAM_SEND_TIMEOUTS = {
    "connect_timeout": 120,
    "read_timeout": 600,
    "write_timeout": 600,
    "pool_timeout": 120,
}


def extract_target_url(text: str) -> str | None:
    if not text:
        return None

    for match in URL_PATTERN.finditer(text):
        candidate = match.group(0).rstrip(URL_TRAILING_NOISE)
        parsed = urlparse(candidate)
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            return candidate
    return None


def is_douyin_url(url: str) -> bool:
    hostname = (urlparse(url).hostname or "").lower()
    return hostname in DOUYIN_HOSTS


def format_file_size(size_bytes: int) -> str:
    return f"{size_bytes / (1024 * 1024):.2f} MB"


def build_send_failure_message(error: Exception) -> str:
    error_text = str(error)
    if isinstance(error, TimedOut) or "ConnectTimeout" in error_text or "TimedOut" in error_text:
        return "发送失败：Telegram 上传超时，请稍后重试"
    if isinstance(error, BadRequest):
        if "Request Entity Too Large" in error_text or "file is too big" in error_text.lower():
            return "发送失败：文件过大，无法发送到 Telegram"
        return f"发送失败：Telegram 返回异常 ({error_text})"
    if isinstance(error, RetryAfter):
        return f"发送失败：Telegram 限流，请稍后重试 ({error_text})"
    if isinstance(error, NetworkError):
        return f"发送失败：Telegram 网络异常 ({error_text})"
    if isinstance(error, FileNotFoundError):
        return "发送失败：文件不存在或路径错误"
    return f"发送失败：未知 Telegram 异常 ({error.__class__.__name__})"


def extract_section_content(body: str, section_title: str) -> str:
    pattern = rf"##\s*{re.escape(section_title)}\s*\n(.*?)(?=\n##\s+|\Z)"
    match = re.search(pattern, body, flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def normalize_tag_token(token: str) -> str:
    normalized = token.strip().lower()
    normalized = re.sub(r"^[\-\*\d\.\)\(、，,;；:\s]+", "", normalized)
    normalized = re.sub(r"[`'\"“”‘’]+", "", normalized)
    normalized = re.sub(r"\s+", "_", normalized)
    normalized = re.sub(r"[^0-9a-z_\-\u4e00-\u9fff]+", "", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_-")
    return normalized


def build_tags(metadata: dict, body: str) -> str:
    keywords_text = extract_section_content(body, "适用检索关键词")
    raw_candidates = re.split(r"[\n,，、;/；|]+", keywords_text)
    tags: list[str] = []

    for candidate in raw_candidates:
        normalized = normalize_tag_token(candidate)
        if normalized and normalized not in tags:
            tags.append(normalized)

    if not tags:
        fallback_candidates = re.split(r"[_\s]+", metadata.get("topic", ""))
        for candidate in fallback_candidates[:8]:
            normalized = normalize_tag_token(candidate)
            if normalized and normalized not in tags:
                tags.append(normalized)

    return ", ".join(tags)


def build_knowledge_card_text(metadata: dict, body: str) -> str:
    tags_line = build_tags(metadata, body)
    return (
        f"# title: {metadata['title']}\n"
        f"# expert: {metadata['expert']}\n"
        f"# date: {metadata['upload_date']}\n"
        f"# source_type: llm_summary_of_youtube_transcript\n"
        f"# original_url: {metadata['original_url']}\n"
        f"# domain: {metadata['domain']}\n"
        f"# tags: {tags_line}\n\n"
        f"{body.strip()}\n"
    )


def build_knowledge_card_filename(metadata: dict) -> str:
    date_part = metadata["upload_date"]
    topic_part = sanitize_filename_component(metadata["topic"], f"video_{metadata['video_id']}", max_length=60)
    expert_raw = (metadata.get("expert") or "").strip()
    expert_part = sanitize_filename_component(expert_raw, "", max_length=40)
    if expert_part and expert_part != "unknown_expert":
        return f"{date_part}_{expert_part}_video_{topic_part}.txt"
    return f"{date_part}_video_{topic_part}.txt"


async def send_media_file(context, status_msg, final_file: str, is_audio: bool, original_msg_id: int):
    send_method_name = "send_audio" if is_audio else "send_video"
    file_exists = os.path.exists(final_file)
    file_size = os.path.getsize(final_file) if file_exists else -1
    logging.info(
        "Telegram 发送开始 file_path=%s file_exists=%s file_size_bytes=%s file_size_mb=%s send_method=%s retry_attempted=%s",
        final_file,
        file_exists,
        file_size,
        format_file_size(file_size) if file_size >= 0 else "N/A",
        send_method_name,
        False,
    )

    if not file_exists:
        raise FileNotFoundError(final_file)

    try:
        with open(final_file, "rb") as media_file:
            if is_audio:
                await context.bot.send_audio(
                    chat_id=status_msg.chat_id,
                    audio=media_file,
                    reply_to_message_id=original_msg_id,
                    **TELEGRAM_SEND_TIMEOUTS,
                )
            else:
                await context.bot.send_video(
                    chat_id=status_msg.chat_id,
                    video=media_file,
                    supports_streaming=True,
                    reply_to_message_id=original_msg_id,
                    **TELEGRAM_SEND_TIMEOUTS,
                )
        logging.info(
            "Telegram 发送成功 file_path=%s file_size_bytes=%s send_method=%s",
            final_file,
            file_size,
            send_method_name,
        )
    except Exception as exc:
        logging.error(
            "Telegram 发送失败 file_path=%s file_size_bytes=%s send_method=%s exception_type=%s exception=%s retry_attempted=%s",
            final_file,
            file_size,
            send_method_name,
            exc.__class__.__name__,
            exc,
            False,
        )
        raise


async def receive_link(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    if user_id not in ALLOWED_USERS:
        await update.message.reply_text(f"🛑 无权限。您的 User ID 是：`{user_id}`")
        return

    text = update.message.text or ""
    url = extract_target_url(text)

    if not url:
        await update.message.reply_text(
            "未识别到有效链接，请发送标准链接或包含抖音链接的分享文本。",
            reply_to_message_id=update.message.message_id,
        )
        return

    url_id = str(uuid.uuid4())[:8]
    context.user_data[url_id] = url

    keyboard = [
        [InlineKeyboardButton("🎬 下载高清视频", callback_data=f"video|{url_id}")],
        [InlineKeyboardButton("🎵 提取纯音频", callback_data=f"audio|{url_id}")],
        [InlineKeyboardButton("🧠 生成专家知识卡片", callback_data=f"transcript|{url_id}")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "请选择您需要的处理方式：👇",
        reply_markup=reply_markup,
        reply_to_message_id=update.message.message_id,
    )


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    data = query.data
    action, url_id = data.split("|")
    url = context.user_data.get(url_id)

    if not url:
        await query.edit_message_text("❌ 链接已过期或失效。")
        return

    is_audio = action in ["audio", "transcript"]
    need_transcript = action == "transcript"
    mode_name = "纯音频 (MP3)" if is_audio else "高清视频"
    if need_transcript:
        mode_name = "专家知识卡片"

    status_msg = query.message
    await status_msg.edit_text(f"开始为您解析{mode_name}... 🔎")

    os.makedirs(SHARED_DIR, exist_ok=True)
    filepath_base = os.path.join(SHARED_DIR, f"media_{uuid.uuid4().hex}")
    final_file = ""
    txt_filepath = ""
    context.user_data["last_update"] = 0
    loop = asyncio.get_event_loop()

    try:
        download_result = await asyncio.to_thread(sync_download, url, filepath_base, status_msg, loop, context, is_audio)
        final_file = download_result["file_path"]
        metadata = download_result["metadata"]

        if not os.path.exists(final_file):
            await status_msg.edit_text("下载失败 ❌")
            return

        await status_msg.edit_text("媒体获取完成！正在发送至 Telegram... 🚀")
        original_msg_id = status_msg.reply_to_message.message_id if status_msg.reply_to_message else status_msg.message_id

        try:
            await send_media_file(context, status_msg, final_file, is_audio, original_msg_id)
        except Exception as send_error:
            await status_msg.edit_text(build_send_failure_message(send_error))
            return

        if need_transcript:
            await status_msg.edit_text("🎵 正在进行 Whisper 听写，请稍候... 🧠")
            transcript_text = await process_groq_transcription(final_file)

            await status_msg.edit_text("🧠 听写完成，正在生成 RAG 专家知识卡片... ⏳")
            card_body = await generate_expert_knowledge_card(
                metadata=metadata,
                expert=metadata["expert"],
                domain=metadata["domain"],
                transcript_text=transcript_text,
            )

            card_content = build_knowledge_card_text(metadata, card_body)
            clean_filename = build_knowledge_card_filename(metadata)
            txt_filepath = filepath_base + ".txt"

            with open(txt_filepath, "w", encoding="utf-8") as file_obj:
                file_obj.write(card_content)

            txt_file_exists = os.path.exists(txt_filepath)
            txt_file_size = os.path.getsize(txt_filepath) if txt_file_exists else -1
            logging.info(
                "Telegram 知识卡片发送开始 file_path=%s file_exists=%s file_size_bytes=%s file_size_mb=%s send_method=%s retry_attempted=%s",
                txt_filepath,
                txt_file_exists,
                txt_file_size,
                format_file_size(txt_file_size) if txt_file_size >= 0 else "N/A",
                "send_document",
                False,
            )
            try:
                with open(txt_filepath, "rb") as document_file:
                    await context.bot.send_document(
                        chat_id=status_msg.chat_id,
                        document=document_file,
                        filename=clean_filename,
                        caption="📚 专家知识卡片已生成，请查收附件。",
                        reply_to_message_id=original_msg_id,
                        **TELEGRAM_SEND_TIMEOUTS,
                    )
                logging.info(
                    "Telegram 知识卡片发送成功 file_path=%s file_size_bytes=%s send_method=%s",
                    txt_filepath,
                    txt_file_size,
                    "send_document",
                )
            except Exception as doc_error:
                logging.error(
                    "Telegram 知识卡片发送失败 file_path=%s file_size_bytes=%s send_method=%s exception_type=%s exception=%s retry_attempted=%s",
                    txt_filepath,
                    txt_file_size,
                    "send_document",
                    doc_error.__class__.__name__,
                    doc_error,
                    False,
                )
                await status_msg.edit_text(build_send_failure_message(doc_error))
                return

        await status_msg.delete()

    except Exception as exc:
        await status_msg.edit_text("发生了一些意外错误 😵")
        logging.error("全局错误: %s", exc)
    finally:
        if final_file and os.path.exists(final_file):
            os.remove(final_file)
        if txt_filepath and os.path.exists(txt_filepath):
            os.remove(txt_filepath)


if __name__ == "__main__":
    builder = ApplicationBuilder().token(TOKEN)
    if TELEGRAM_BASE_URL:
        builder = builder.base_url(TELEGRAM_BASE_URL)
    if TELEGRAM_LOCAL_MODE:
        builder = builder.local_mode(True)
    app = builder.build()
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), receive_link))
    app.add_handler(CallbackQueryHandler(button_callback))
    print("🤖 模块化知识卡片机器人已启动...")
    app.run_polling()
