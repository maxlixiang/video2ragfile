import logging
from urllib.parse import urlparse

from openai import AsyncOpenAI

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, GROQ_API_KEY, GROQ_BASE_URL


logger = logging.getLogger(__name__)

groq_client = AsyncOpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
deepseek_client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

TRANSCRIPT_PROMPT_LIMIT = 18000
ALLOWED_DOMAINS = {"geopolitics", "markets", "tech", "general"}
ALLOWED_SOURCE_PLATFORMS = {"youtube", "article", "podcast", "speech", "manual"}
ALLOWED_LANGUAGES = {"zh", "en"}


async def process_groq_transcription(file_path: str) -> str:
    """使用 Whisper 将音频转为文本。"""
    try:
        with open(file_path, "rb") as audio_file:
            transcript = await groq_client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=audio_file,
                response_format="text",
            )
        return transcript
    except Exception as exc:
        return f"❌ Whisper 转写失败: {exc}"


def _truncate_transcript(text: str, limit: int = TRANSCRIPT_PROMPT_LIMIT) -> tuple[str, bool]:
    normalized = (text or "").strip()
    if len(normalized) <= limit:
        return normalized, False
    return normalized[:limit], True


def _extract_hostname(url: str) -> str:
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        return (parsed.netloc or "").lower()
    except Exception:
        return ""


def _normalize_business_domain(domain: str, metadata: dict) -> str:
    candidate = (domain or "").strip().lower()
    if candidate in ALLOWED_DOMAINS:
        return candidate

    hostname = _extract_hostname(metadata.get("original_url", ""))
    source_hint = " ".join(
        str(value or "").lower()
        for value in (
            metadata.get("title", ""),
            metadata.get("source_type", ""),
            candidate,
            hostname,
        )
    )

    if any(token in source_hint for token in ("market", "macro", "trading", "gold", "silver", "oil", "stock", "bond", "fx", "crypto")):
        return "markets"
    if any(token in source_hint for token in ("geopolitic", "war", "ukraine", "russia", "china", "europe", "military", "diplom")):
        return "geopolitics"
    if any(token in source_hint for token in ("tech", "ai", "chip", "semiconductor", "software", "cloud", "robot")):
        return "tech"
    return "general"


def _infer_source_platform(metadata: dict) -> str:
    source_type = str(metadata.get("source_type", "")).strip().lower()
    hostname = _extract_hostname(metadata.get("original_url", ""))

    if "youtube.com" in hostname or "youtu.be" in hostname:
        return "youtube"
    if "podcast" in source_type or "spotify" in hostname or "apple" in hostname:
        return "podcast"
    if "speech" in source_type or "conference" in source_type or "talk" in source_type:
        return "speech"
    if "manual" in source_type or "documentation" in source_type or "docs" in hostname:
        return "manual"
    if source_type in ALLOWED_SOURCE_PLATFORMS:
        return source_type
    return "article"


def _infer_language(metadata: dict, transcript_text: str) -> str:
    for key in ("language", "lang"):
        candidate = str(metadata.get(key, "")).strip().lower()
        if candidate.startswith("zh"):
            return "zh"
        if candidate.startswith("en"):
            return "en"

    sample = (transcript_text or "")[:1200]
    if any("\u4e00" <= ch <= "\u9fff" for ch in sample):
        return "zh"
    return "en"


def _build_keyword_fallback(metadata: dict, domain: str, source_platform: str) -> str:
    title = str(metadata.get("title", "")).lower()
    tokens: list[str] = []
    if domain:
        tokens.append(domain)
    if source_platform:
        tokens.append(source_platform)

    keyword_map = {
        "silver": "silver",
        "gold": "gold",
        "oil": "oil",
        "gas": "natural_gas",
        "ukraine": "ukraine",
        "russia": "russia",
        "china": "china",
        "europe": "europe",
        "ai": "artificial_intelligence",
        "chip": "semiconductor",
        "semiconductor": "semiconductor",
        "google": "google",
        "tesla": "tesla",
        "nvidia": "nvidia",
        "inflation": "inflation",
        "tariff": "tariffs",
    }
    for raw, normalized in keyword_map.items():
        if raw in title and normalized not in tokens:
            tokens.append(normalized)

    deduped = []
    for token in tokens:
        cleaned = token.strip().lower().replace("-", "_").replace(" ", "_")
        if cleaned and cleaned not in deduped:
            deduped.append(cleaned)
    return ", ".join(deduped[:8])


def _fallback_card(error_message: str = "") -> str:
    headline = f"❌ {error_message}" if error_message else ""
    return (
        "## 主题归一化\n\n"
        "topic_key:\n"
        "topic_family:\n"
        "source_platform:\n"
        "language:\n\n"
        "## 核心观点对象\n\n"
        "- [thesis]\n\n"
        "## 本期核心事实\n\n"
        f"{headline}\n\n"
        "## 专家主要观点\n\n"
        "## 专家的判断框架\n\n"
        "## 对当前国际局势/市场的影响\n\n"
        "## 后续观察点\n\n"
        "## 适用检索关键词\n\n"
        "## 不确定性与保留意见"
    ).strip()


def _ensure_keyword_section(card_text: str, fallback_keywords: str) -> str:
    if not fallback_keywords:
        return card_text

    marker = "## 适用检索关键词"
    if marker not in card_text:
        return card_text

    head, tail = card_text.split(marker, 1)
    stripped_tail = tail.lstrip("\n")
    if stripped_tail.startswith("## ") or not stripped_tail.strip():
        return f"{head}{marker}\n\n{fallback_keywords}\n\n{stripped_tail}".rstrip()
    return card_text


async def generate_expert_knowledge_card(
    metadata: dict,
    expert: str,
    domain: str,
    transcript_text: str,
) -> str:
    """使用 DeepSeek 将视频文稿整理成兼容知识演化层的专家知识卡片。"""
    if not DEEPSEEK_API_KEY:
        return _fallback_card("未配置 DEEPSEEK_API_KEY，无法生成知识卡片。")

    if len((transcript_text or "").strip()) < 10 or "❌" in transcript_text:
        return transcript_text

    normalized_domain = _normalize_business_domain(domain, metadata)
    source_platform = _infer_source_platform(metadata)
    language = _infer_language(metadata, transcript_text)
    fallback_keywords = _build_keyword_fallback(metadata, normalized_domain, source_platform)

    truncated_transcript, was_truncated = _truncate_transcript(transcript_text)
    transcript_notice = (
        "注意：输入字幕过长，程序已做截断。输出中需要明确哪些内容来自已给出的字幕，避免对未提供部分做推断。"
        if was_truncated
        else "输入字幕为完整版本或未触发截断。"
    )

    system_prompt = """
你是一名专家知识卡片编辑，负责把输入 transcript 整理成适合 RAG 检索，并兼容 knowledge_evolution_layer v1 的 v2 知识卡片正文。

必须严格遵守以下规则：
1. 只输出各 section 的正文内容，不要输出 Markdown 代码块，不要重复头部字段。
2. 输出语言默认与 transcript 保持一致；但 topic_key、topic_family、tags、适用检索关键词优先使用英文。
3. 不得编造 transcript 中没有的信息；必须明确区分事实、观点、推测、待验证内容。
4. 这不是整篇摘要，而是可比较、可复用、可跨时间演化的观点单元整理。
5. topic_key 要尽量稳定，使用英文 snake_case，聚焦最核心主题，不要随着措辞轻易变化。
6. topic_family 要使用较粗粒度的英文归类，例如 precious_metals、europe_geopolitics、ai_infrastructure、energy_security。
7. source_platform 只能写 youtube、article、podcast、speech、manual 之一。
8. language 尽量写 zh 或 en。
9. 核心观点对象必须拆成 3 到 6 条，每条只表达一个可比较观点，格式必须是：
- [view_type] 观点内容
其中 view_type 只能是 methodology、thesis、tactical_view、event_call。
10. “本期核心事实”只写 transcript 中明确提到的事实或背景信息，避免夹带观点判断。
11. “专家主要观点”写作者/嘉宾表达的核心判断。
12. “专家的判断框架”优先抽取其分析维度、因果链条、比较框架、观察指标。
13. “对当前国际局势/市场的影响”只写由专家观点推导出的影响，不要把未证实猜测写成事实。
14. “后续观察点”要写后续验证该观点最关键的指标、事件、数据或政策变量。
15. “适用检索关键词”必须只输出一行英文标签，全部小写，逗号分隔，多词用下划线连接；不要输出中文自然语言，不要写解释，不要换行列表。
16. tags/关键词优先选择稳定、可聚合、可检索的实体和主题标签，避免空泛词。
17. card_version 固定为 v2。虽然不要重复头部字段，但正文中的“主题归一化” section 必须与头部字段兼容。
""".strip()

    user_prompt = f"""
请基于以下元数据和 transcript，生成 v2 专家知识卡片正文。

头部兼容目标如下，请在正文 section 中提供与这些字段一致的信息，但不要直接重复输出整段头部：
# title: {metadata.get("title", "")}
# expert: {expert}
# date: {metadata.get("upload_date", "")}
# source_type: {metadata.get("source_type", "")}
# original_url: {metadata.get("original_url", "")}
# domain: {normalized_domain}
# source_platform: {source_platform}
# language: {language}
# topic_key:
# topic_family:
# tags:
# card_version: v2

补充说明：{transcript_notice}

请严格按以下 section 顺序输出正文内容：

## 主题归一化
至少包含以下四行：
topic_key:
topic_family:
source_platform: {source_platform}
language: {language}

要求：
- topic_key 使用稳定的英文 snake_case，例如 silver_supply_deficit、eu_ukraine_financial_support、google_ai_chip_competition。
- topic_family 使用较粗粒度的英文主题族，例如 precious_metals、europe_geopolitics、ai_infrastructure、energy_security。
- source_platform 和 language 应与给定头部兼容。

## 核心观点对象
要求：
- 输出 3 到 6 条列表。
- 每条使用格式：- [view_type] 观点内容
- view_type 只能是 methodology、thesis、tactical_view、event_call。
- 每条尽量一句话，保证可比较、可复用、可跨时间比较。
- methodology 表示方法论或判断框架。
- thesis 表示中长期核心观点。
- tactical_view 表示阶段性判断。
- event_call 表示事件性判断或事件风险判断。

## 本期核心事实
要求：
- 只写事实、数据、背景、明示事件。
- 不要把观点和推测混进去。

## 专家主要观点
要求：
- 聚焦专家核心判断，不要写成长篇综述。

## 专家的判断框架
要求：
- 提炼专家如何得出判断，重点写观察维度、因果链条、验证方式。

## 对当前国际局势/市场的影响

## 后续观察点
要求：
- 写可以用于后续 reinforced、updated、reversed 判断的关键观测点。

## 适用检索关键词
要求：
- 只输出一行标签。
- 优先英文。
- 全部小写。
- 多词用下划线连接。
- 逗号分隔。
- 不要输出中文自然语言关键词串。
- 示例：silver, precious_metals, silver_supply_deficit, silver_inventory, silver_etf, physical_tightness, squeeze_risk

## 不确定性与保留意见
要求：
- 明确哪些是待验证内容、哪些依赖不完整信息、哪些因为字幕截断而无法确认。

transcript 如下：
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
        content = (response.choices[0].message.content or "").strip()
        return _ensure_keyword_section(content, fallback_keywords)
    except Exception as exc:
        logger.error("DeepSeek 知识卡片生成失败: %s", exc)
        return _fallback_card(f"知识卡片生成失败: {exc}")
