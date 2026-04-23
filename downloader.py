import asyncio
import logging
import os
import re
import subprocess
import time
from datetime import datetime
from urllib.parse import urlparse

import yt_dlp

from douyin_downloader import download_file, fetch_douyin_video_meta, load_douyin_cookie_header
from twitter_downloader import download_file as download_twitter_file
from twitter_downloader import fetch_twitter_video_meta, is_twitter_url


COOKIE_FILE = "/app/cookies.txt"
DOUYIN_HOSTS = {"douyin.com", "www.douyin.com", "v.douyin.com", "iesdouyin.com"}
logger = logging.getLogger(__name__)


def today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def progress_hook(d, status_msg, loop, context):
    if d["status"] == "downloading":
        last_update_time = context.user_data.get("last_update", 0)
        current_time = time.time()

        if current_time - last_update_time > 2.5:
            percent = d.get("_percent_str", "0%")
            speed = d.get("_speed_str", "N/A")
            eta = d.get("_eta_str", "N/A")

            try:
                p = float(percent.replace("%", "").strip()) / 10
                bar = "■" * int(p) + "□" * (10 - int(p))
            except Exception:
                bar = "■■■■■□□□□□"

            progress_text = f"正在极速下载中... ⏳\n\n进度: [{bar}] {percent}\n速度: {speed}\n剩余时间: {eta}"
            asyncio.run_coroutine_threadsafe(status_msg.edit_text(progress_text), loop)
            context.user_data["last_update"] = current_time


def is_douyin_url(url: str) -> bool:
    hostname = (urlparse(url).hostname or "").lower()
    return hostname in DOUYIN_HOSTS


def normalize_upload_date(value: str | None) -> str:
    raw = (value or "").strip()
    if not raw:
        return today_str()

    if re.fullmatch(r"\d{8}", raw):
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
        return raw

    return today_str()


def sanitize_filename_component(value: str, fallback: str, max_length: int = 48) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|]+', " ", (value or "").strip())
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("._ ")
    if not cleaned:
        cleaned = fallback
    return cleaned[:max_length].strip("._ ") or fallback


def build_topic_from_title(title: str, video_id: str) -> str:
    cleaned = re.sub(r"[^\w\u4e00-\u9fff\s-]", " ", (title or "").strip(), flags=re.UNICODE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return f"video_{video_id}"

    words = cleaned.split(" ")
    topic = "_".join(words[:8])
    return sanitize_filename_component(topic.lower(), f"video_{video_id}", max_length=60)


def build_metadata(
    *,
    title: str | None,
    upload_date: str | None,
    original_url: str,
    domain: str,
    expert: str | None,
    video_id: str | None,
) -> dict:
    raw_video_id = (video_id or "").strip() or "unknown"
    safe_video_id = sanitize_filename_component(raw_video_id, "unknown", max_length=40)
    safe_title = (title or "").strip() or f"video_{safe_video_id}"
    safe_expert = (expert or "").strip() or "unknown_expert"
    return {
        "title": safe_title,
        "upload_date": normalize_upload_date(upload_date),
        "original_url": original_url,
        "domain": domain,
        "expert": safe_expert,
        "video_id": safe_video_id,
        "topic": build_topic_from_title(safe_title, safe_video_id),
    }


def extract_audio_to_mp3(input_video_path: str, output_mp3_path: str) -> str:
    process = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            input_video_path,
            "-vn",
            "-acodec",
            "libmp3lame",
            "-b:a",
            "64k",
            output_mp3_path,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if process.returncode != 0:
        logger.error("FFmpeg 转换失败 stderr_preview=%s", process.stderr[:500])
        raise RuntimeError("媒体音频提取失败")
    return output_mp3_path


def download_douyin_media(url: str, filepath_base: str, is_audio: bool) -> dict:
    cookie_header, cookie_names = load_douyin_cookie_header(COOKIE_FILE)
    logger.info(
        "抖音 Cookie 读取结果 raw_url=%s cookie_loaded=%s cookie_names=%s",
        url,
        bool(cookie_names),
        cookie_names,
    )

    media_meta = fetch_douyin_video_meta(url, cookie_header)
    metadata = build_metadata(
        title=media_meta.get("title"),
        upload_date=media_meta.get("upload_date"),
        original_url=url,
        domain="douyin.com",
        expert=media_meta.get("expert"),
        video_id=media_meta.get("aweme_id"),
    )
    mp4_path = filepath_base + ".mp4"
    download_file(media_meta["video_url"], mp4_path, media_meta["headers"])

    if not is_audio:
        return {"file_path": mp4_path, "metadata": metadata}

    mp3_path = filepath_base + ".mp3"
    try:
        extract_audio_to_mp3(mp4_path, mp3_path)
    finally:
        if os.path.exists(mp4_path):
            os.remove(mp4_path)
    return {"file_path": mp3_path, "metadata": metadata}


def download_twitter_media(url: str, filepath_base: str, is_audio: bool) -> dict:
    media_meta = fetch_twitter_video_meta(url)
    metadata = build_metadata(
        title=media_meta.get("title"),
        upload_date=media_meta.get("upload_date"),
        original_url=url,
        domain="x.com",
        expert=media_meta.get("expert"),
        video_id=media_meta.get("tweet_id"),
    )
    mp4_path = filepath_base + ".mp4"
    download_twitter_file(media_meta["video_url"], mp4_path, media_meta["headers"])

    if not is_audio:
        return {"file_path": mp4_path, "metadata": metadata}

    mp3_path = filepath_base + ".mp3"
    try:
        extract_audio_to_mp3(mp4_path, mp3_path)
    finally:
        if os.path.exists(mp4_path):
            os.remove(mp4_path)
    return {"file_path": mp3_path, "metadata": metadata}


def extract_generic_metadata(url: str) -> dict:
    domain = (urlparse(url).hostname or "unknown").lower()
    try:
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True}) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:
        logger.warning("通用元数据提取失败 url=%s error=%s", url, exc)
        info = {}

    return build_metadata(
        title=info.get("title"),
        upload_date=info.get("upload_date") or info.get("release_date"),
        original_url=info.get("webpage_url") or url,
        domain=domain,
        expert=info.get("uploader") or info.get("channel") or info.get("creator") or info.get("author"),
        video_id=info.get("id"),
    )


def sync_download(url: str, filepath_base: str, status_msg, loop, context, is_audio: bool) -> dict:
    if is_douyin_url(url):
        return download_douyin_media(url, filepath_base, is_audio)
    if is_twitter_url(url):
        return download_twitter_media(url, filepath_base, is_audio)

    metadata = extract_generic_metadata(url)

    if is_audio:
        ydl_opts = {
            "outtmpl": filepath_base,
            "format": "bestaudio/best",
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "64"}],
            "quiet": True,
            "no_warnings": True,
            "progress_hooks": [lambda d: progress_hook(d, status_msg, loop, context)],
        }
        final_file = filepath_base + ".mp3"
    else:
        ydl_opts = {
            "outtmpl": filepath_base + ".mp4",
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "quiet": True,
            "no_warnings": True,
            "progress_hooks": [lambda d: progress_hook(d, status_msg, loop, context)],
        }
        final_file = filepath_base + ".mp4"

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return {"file_path": final_file, "metadata": metadata}
