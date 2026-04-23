import logging

from openai import AsyncOpenAI

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, GROQ_API_KEY, GROQ_BASE_URL


logger = logging.getLogger(__name__)

groq_client = AsyncOpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
deepseek_client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

TRANSCRIPT_PROMPT_LIMIT = 18000


async def process_groq_transcription(file_path: str) -> str:
    """第一引擎：Whisper 极速语音转文字"""
    try:
        with open(file_path, "rb") as audio_file:
            transcript = await groq_client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=audio_file,
                response_format="text",
            )
        return transcript
    except Exception as exc:
        return f"❌ Whisper 听写失败: {exc}"


def _truncate_transcript(text: str, limit: int = TRANSCRIPT_PROMPT_LIMIT) -> tuple[str, bool]:
    normalized = (text or "").strip()
    if len(normalized) <= limit:
        return normalized, False
    return normalized[:limit], True


async def generate_expert_knowledge_card(
    metadata: dict,
    expert: str,
    domain: str,
    transcript_text: str,
) -> str:
    """使用 DeepSeek 将视频文稿改写成适合 RAG 检索的专家知识卡片。"""
    if not DEEPSEEK_API_KEY:
        return "## 本期核心事实\n\n❌ 未配置 DEEPSEEK_API_KEY，无法生成知识卡片。\n\n## 专家主要观点\n\n## 专家的判断框架\n\n## 对当前国际局势/市场的影响\n\n## 后续观察点\n\n## 适用检索关键词\n\n## 不确定性与保留意见"

    if len((transcript_text or "").strip()) < 10 or "❌" in transcript_text:
        return transcript_text

    truncated_transcript, was_truncated = _truncate_transcript(transcript_text)
    transcript_notice = (
        "注意：输入字幕过长，程序已经做了截断，请在输出中明确体现“字幕已截断”，并避免对未提供内容做推断。"
        if was_truncated
        else "输入字幕为完整或未触发截断的版本。"
    )

    system_prompt = (
        "你是一名RAG知识整理编辑，负责把视频文稿整理成“专家知识卡片”。"
        "必须遵守以下规则："
        "1. 输出语言必须与输入字幕语言保持一致；"
        "2. 不要写成逐字稿，不要复述口头禅；"
        "3. 不得编造视频中没有的信息；"
        "4. 明确区分事实、观点、推测和待验证内容；"
        "5. 尽量提炼专家的分析框架、判断逻辑、观察维度；"
        "6. “适用检索关键词”要兼顾向量检索和关键词检索，尽量输出检索友好的短标签，优先使用小写英文标签或 snake_case，多词标签用下划线连接；"
        "7. 严格只输出各 section 内容，不要输出 Markdown 代码块，不要重复头部字段。"
    )

    user_prompt = f"""
视频元数据：
title: {metadata["title"]}
expert: {expert}
date: {metadata["upload_date"]}
original_url: {metadata["original_url"]}
domain: {domain}

补充说明：
{transcript_notice}

请基于下面的字幕内容，只生成以下 section 的正文内容：

## 本期核心事实

## 专家主要观点

## 专家的判断框架

## 对当前国际局势/市场的影响

## 后续观察点

## 适用检索关键词
这一节请尽量输出一行逗号分隔的标签，例如：silver, precious_metals, commodity, supply_deficit

## 不确定性与保留意见

字幕如下：
{truncated_transcript}
""".strip()

    try:
        response = await deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.error("DeepSeek 知识卡片生成失败: %s", exc)
        return f"## 本期核心事实\n\n❌ 知识卡片生成失败: {exc}\n\n## 专家主要观点\n\n## 专家的判断框架\n\n## 对当前国际局势/市场的影响\n\n## 后续观察点\n\n## 适用检索关键词\n\n## 不确定性与保留意见"
