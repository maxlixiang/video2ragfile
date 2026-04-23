# Video2RagFile

Video2RagFile is a Telegram bot that downloads videos from multiple platforms, extracts audio, transcribes speech, and generates RAG-friendly expert knowledge cards as `.txt` files.

## What It Does

- Accepts a video URL from Telegram chat
- Supports three actions:
  - Download video
  - Extract audio
  - Generate expert knowledge card
- Supports:
  - YouTube
  - Bilibili
  - Douyin
  - X / Twitter
  - Other platforms supported by `yt-dlp`
- Extracts metadata with fallback logic
- Generates structured knowledge cards with business-domain classification and searchable tags

## Output Format

The generated `.txt` file contains a structured expert card like this:

```text
# title: ...
# expert: ...
# date: ...
# source_type: llm_summary_of_youtube_transcript
# original_url: ...
# domain: markets
# tags: silver, precious_metals, commodity, supply_deficit

## Core Facts

## Expert Views

## Reasoning Framework

## Market / Geopolitical Impact

## Watch Points

## Retrieval Keywords

## Uncertainty and Caveats
```

## Filename Rules

Preferred format:

```text
date_expert_video_topic.txt
```

Example:

```text
2026-04-23_zhangsan_video_iran_hormuz_ceasefire.txt
```

Fallback when expert information is unavailable:

```text
date_video_topic.txt
```

Example:

```text
2026-04-23_video_iran_hormuz_ceasefire.txt
```

## Domain Classification

The `domain` field is a business domain, not the source website hostname.

Current values are limited to:

- `geopolitics`
- `markets`
- `tech`
- `general`

## Tags

The `tags` field is automatically generated from the `Retrieval Keywords` section of the card body.

If the LLM does not produce usable keywords, the program falls back to the cleaned topic/title tokens.

## Project Structure

```text
.
+-- main.py
+-- downloader.py
+-- ai_services.py
+-- config.py
+-- douyin_downloader.py
+-- twitter_downloader.py
+-- douyin_a_bogus.py
+-- douyin_a_bogus.js
+-- docker-compose.yml
+-- Dockerfile
+-- requirements.txt
+-- cookies.txt
```

## Core Flow

1. User sends a URL to the Telegram bot.
2. The bot extracts the target URL.
3. The bot shows action buttons.
4. The program routes download logic by platform.
5. If needed, audio is extracted to MP3.
6. Whisper transcription is generated.
7. The LLM converts the transcript into an expert knowledge card.
8. The final `.txt` file is sent back to Telegram.

## Platform Strategy

### Generic Platforms

Handled with `yt-dlp`:

- YouTube
- Bilibili
- most other supported platforms

### Douyin

Handled with custom logic:

- resolve share URL
- extract `aweme_id`
- load cookies from `cookies.txt`
- generate `a_bogus`
- request detail API
- download MP4 directly

### X / Twitter

Handled with custom logic:

- extract `tweet_id`
- build `fxtwitter` MP4 URL
- download MP4 directly
- fetch metadata separately when possible

## Requirements

### Python

- `python-telegram-bot`
- `yt-dlp`
- `openai`
- `python-dotenv`
- `requests`

Install with:

```bash
pip install -r requirements.txt
```

### System Dependencies

- `ffmpeg`
- `nodejs`
- `npm`

Douyin also requires:

- `jsdom`

## Environment Variables

Create a `.env` file:

```env
BOT_TOKEN=your_telegram_bot_token
ALLOWED_USERS=123456789
GROQ_API_KEY=your_groq_key
DEEPSEEK_API_KEY=your_deepseek_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
TELEGRAM_BASE_URL=
TELEGRAM_LOCAL_MODE=false
```

### Notes

- `ALLOWED_USERS` can contain multiple Telegram user IDs separated by commas.
- For local Windows Docker testing, keep:
  - `TELEGRAM_BASE_URL=`
  - `TELEGRAM_LOCAL_MODE=false`
- For VPS deployment with local Telegram Bot API, use:
  - `TELEGRAM_BASE_URL=http://127.0.0.1:8081/bot`
  - `TELEGRAM_LOCAL_MODE=true`

## cookies.txt

`cookies.txt` is required for Douyin download support.

If you do not need Douyin immediately, you can still provide an empty placeholder file so Docker volume mounting succeeds.

## Run Locally

```bash
python main.py
```

## Docker Deployment

### Build and Run

```bash
docker compose build
docker compose up -d
```

### View Logs

```bash
docker compose logs -f --tail 100
```

## Windows Local Testing

Recommended `.env` values:

```env
TELEGRAM_BASE_URL=
TELEGRAM_LOCAL_MODE=false
```

Then run:

```powershell
docker compose build
docker compose up
```

Suggested test flow:

1. Send a YouTube or Bilibili URL to the bot.
2. Click the knowledge-card generation button.
3. Verify:
   - audio is returned
   - a `.txt` card is returned
   - `domain` is business-oriented
   - `tags` is not empty

## VPS Deployment

Current `docker-compose.yml` is intended for VPS usage and assumes:

- host networking
- local Telegram Bot API storage mounted at `/var/lib/telegram-bot-api`

Current compose file:

```yaml
services:
  tg_media_bot:
    build: .
    container_name: video_downloader_bot
    restart: unless-stopped
    network_mode: "host"
    volumes:
      - /var/lib/telegram-bot-api:/var/lib/telegram-bot-api
      - ./cookies.txt:/app/cookies.txt:ro
    env_file:
      - .env
```

For VPS `.env`, typically use:

```env
TELEGRAM_BASE_URL=http://127.0.0.1:8081/bot
TELEGRAM_LOCAL_MODE=true
```

## Common Issues

### `git pull` says local changes would be overwritten

Usually this means you edited `docker-compose.yml` on the server.

Safe update flow:

```bash
git stash push -m "local deploy changes"
git pull --ff-only
git stash pop
```

### Telegram connection error

If you see connection failures at startup:

- check `BOT_TOKEN`
- check `TELEGRAM_BASE_URL`
- check whether local `telegram-bot-api` is running
- check whether Docker networking matches your `.env`

### Douyin download failure

Check:

- `cookies.txt`
- `nodejs`
- `npm`
- `jsdom`

### Knowledge card generation is slow

The slowest step is usually the LLM summarization stage, not Whisper transcription.

## Development Notes

- `main.py`: Telegram bot entrypoint and file sending logic
- `downloader.py`: routing, metadata normalization, domain inference
- `ai_services.py`: transcription and knowledge-card generation
- `douyin_downloader.py`: Douyin-specific downloader
- `twitter_downloader.py`: X/Twitter-specific downloader

## License

Add a license if you plan to make the project public for wider reuse.
