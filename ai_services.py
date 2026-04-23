import logging
import re
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
EXPECTED_CARD_SECTIONS = [
    "## 主题归一化",
    "## 核心观点对象",
    "## 本期核心事实",
    "## 专家主要观点",
    "## 专家的判断框架",
    "## 对当前国际局势/市场的影响",
    "## 后续观察点",
    "## 适用检索关键词",
    "## 不确定性与保留意见",
]


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


def _slugify_token(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", (text or "").strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def _keyword_tokens_from_text(text: str) -> list[str]:
    if not text:
        return []
    tokens: list[str] = []
    for part in text.split(","):
        cleaned = _slugify_token(part)
        if cleaned and cleaned not in tokens:
            tokens.append(cleaned)
    return tokens


def _derive_topic_family(metadata: dict, domain: str, fallback_keywords: str) -> str:
    source_text = " ".join(
        [
            str(metadata.get("title", "")).lower(),
            str(metadata.get("source_type", "")).lower(),
            domain.lower(),
            fallback_keywords.lower(),
        ]
    )

    if any(token in source_text for token in ("silver", "gold", "bullion", "precious_metals")):
        return "precious_metals"
    if any(token in source_text for token in ("commodity", "commodities", "copper", "agriculture")):
        return "commodities"
    if any(token in source_text for token in ("ukraine", "russia", "europe", "eu", "nato")):
        return "europe_geopolitics"
    if any(token in source_text for token in ("ai", "chip", "semiconductor", "data_center", "cloud", "gpu")):
        return "ai_infrastructure"
    if any(token in source_text for token in ("oil", "gas", "lng", "hormuz", "energy")):
        return "energy_security"
    if domain == "markets":
        return "macro_markets"
    if domain == "geopolitics":
        return "global_geopolitics"
    if domain == "tech":
        return "technology_strategy"
    return "general_analysis"


def _derive_topic_key(metadata: dict, domain: str, fallback_keywords: str, topic_family: str) -> str:
    title = str(metadata.get("title", "")).lower()
    title_tokens = re.findall(r"[a-z0-9]+", title)
    stopwords = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "into",
        "about",
        "after",
        "before",
        "video",
        "podcast",
        "interview",
        "update",
        "market",
        "markets",
        "expert",
        "talk",
        "news",
        "why",
        "what",
        "how",
    }

    ranked_tokens = [token for token in title_tokens if len(token) > 2 and token not in stopwords]
    keyword_tokens = _keyword_tokens_from_text(fallback_keywords)
    for token in keyword_tokens:
        if token not in ranked_tokens:
            ranked_tokens.append(token)

    prioritized_groups = [
        ("silver", "supply", "deficit"),
        ("gold", "reserve"),
        ("ukraine", "russia"),
        ("eu", "ukraine"),
        ("ai", "chip"),
        ("google", "ai"),
        ("oil", "supply"),
        ("natural_gas", "supply"),
    ]
    for group in prioritized_groups:
        if all(any(part == token or part in token for token in ranked_tokens) for part in group):
            return "_".join(group)

    selected: list[str] = []
    for token in ranked_tokens:
        cleaned = _slugify_token(token)
        if cleaned and cleaned not in selected:
            selected.append(cleaned)
        if len(selected) == 3:
            break

    if not selected:
        if topic_family.endswith("_analysis"):
            return topic_family.replace("_analysis", "_topic")
        return f"{topic_family}_topic"
    return "_".join(selected[:3])


def _extract_sections(card_text: str) -> tuple[dict[str, str], str]:
    pattern = re.compile(r"(?ms)^(## [^\n]+)\n(.*?)(?=^## |\Z)")
    sections: dict[str, str] = {}
    for match in pattern.finditer(card_text.strip()):
        heading = match.group(1).strip()
        body = match.group(2).strip()
        sections[heading] = body

    remainder = pattern.sub("", card_text.strip()).strip()
    return sections, remainder


def _split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"[\r\n]+", "\n", text or "")
    chunks = re.split(r"[\n。！？!?；;]+", normalized)
    sentences: list[str] = []
    for chunk in chunks:
        cleaned = re.sub(r"^[\-\*\d\.\)\s]+", "", chunk).strip(" -\t")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if len(cleaned) >= 10 and cleaned not in sentences:
            sentences.append(cleaned)
    return sentences


def _classify_view_type(sentence: str) -> str:
    lowered = sentence.lower()
    methodology_tokens = ("framework", "method", "indicator", "watch", "track", "observe", "维度", "框架", "指标", "观察", "判断")
    event_tokens = ("risk", "trigger", "squeeze", "breakout", "shock", "event", "催化", "风险", "事件", "冲击", "爆发")
    tactical_tokens = ("current", "near term", "short term", "this year", "next", "近期", "当前", "阶段", "短期", "未来几个月")
    if any(token in lowered for token in methodology_tokens):
        return "methodology"
    if any(token in lowered for token in event_tokens):
        return "event_call"
    if any(token in lowered for token in tactical_tokens):
        return "tactical_view"
    return "thesis"


def _build_core_viewpoints_fallback(sections: dict[str, str]) -> str:
    source_blocks = [
        sections.get("## 专家主要观点", ""),
        sections.get("## 专家的判断框架", ""),
        sections.get("## 对当前国际局势/市场的影响", ""),
    ]
    candidates: list[str] = []
    for block in source_blocks:
        for sentence in _split_sentences(block):
            if sentence not in candidates:
                candidates.append(sentence)

    fallback_lines: list[str] = []
    for sentence in candidates[:5]:
        fallback_lines.append(f"- [{_classify_view_type(sentence)}] {sentence}")

    while len(fallback_lines) < 3:
        seeds = [
            "- [thesis] 专家围绕当前主题提出了可持续跟踪的核心判断。",
            "- [methodology] 专家的判断依赖于若干关键指标、背景条件和验证信号。",
            "- [tactical_view] 当前阶段的市场或局势变化需要结合后续数据继续验证。",
        ]
        for seed in seeds:
            if seed not in fallback_lines:
                fallback_lines.append(seed)
            if len(fallback_lines) >= 3:
                break

    return "\n".join(fallback_lines[:5])


def _ensure_topic_normalization_section(
    sections: dict[str, str],
    metadata: dict,
    domain: str,
    source_platform: str,
    language: str,
    fallback_keywords: str,
) -> None:
    if sections.get("## 主题归一化", "").strip():
        return

    topic_family = _derive_topic_family(metadata, domain, fallback_keywords)
    topic_key = _derive_topic_key(metadata, domain, fallback_keywords, topic_family)
    sections["## 主题归一化"] = (
        f"topic_key: {topic_key}\n"
        f"topic_family: {topic_family}\n"
        f"source_platform: {source_platform}\n"
        f"language: {language}"
    )


def _ensure_core_viewpoints_section(sections: dict[str, str]) -> None:
    if sections.get("## 核心观点对象", "").strip():
        return
    sections["## 核心观点对象"] = _build_core_viewpoints_fallback(sections)


def _ensure_keyword_section(sections: dict[str, str], fallback_keywords: str) -> None:
    if not fallback_keywords:
        return

    current = sections.get("## 适用检索关键词", "").strip()
    if not current:
        sections["## 适用检索关键词"] = fallback_keywords
        return

    if current.startswith("## "):
        sections["## 适用检索关键词"] = fallback_keywords


def _rebuild_card_content(sections: dict[str, str], remainder: str) -> str:
    ordered_parts: list[str] = []
    for heading in EXPECTED_CARD_SECTIONS:
        body = sections.get(heading, "").strip()
        if body:
            ordered_parts.append(f"{heading}\n\n{body}")

    if remainder:
        if "## 专家主要观点" in sections and sections.get("## 专家主要观点", "").strip():
            updated_body = sections["## 专家主要观点"].strip()
            if remainder not in updated_body:
                sections["## 专家主要观点"] = f"{updated_body}\n\n{remainder}".strip()
            ordered_parts = []
            for heading in EXPECTED_CARD_SECTIONS:
                body = sections.get(heading, "").strip()
                if body:
                    ordered_parts.append(f"{heading}\n\n{body}")
        else:
            ordered_parts.append(remainder)

    return "\n\n".join(ordered_parts).strip()


def _fallback_card(
    error_message: str = "",
    metadata: dict | None = None,
    domain: str = "general",
    source_platform: str = "article",
    language: str = "zh",
) -> str:
    headline = f"❌ {error_message}" if error_message else ""
    metadata = metadata or {}
    fallback_keywords = _build_keyword_fallback(metadata, domain, source_platform)
    sections = {
        "## 本期核心事实": headline,
        "## 专家主要观点": "",
        "## 专家的判断框架": "",
        "## 对当前国际局势/市场的影响": "",
        "## 后续观察点": "",
        "## 不确定性与保留意见": "",
    }
    _ensure_topic_normalization_section(
        sections,
        metadata,
        domain,
        source_platform,
        language,
        fallback_keywords,
    )
    _ensure_core_viewpoints_section(sections)
    _ensure_keyword_section(sections, fallback_keywords)
    return _rebuild_card_content(sections, "")


async def generate_expert_knowledge_card(
    metadata: dict,
    expert: str,
    domain: str,
    transcript_text: str,
) -> str:
    """使用 DeepSeek 将视频文稿整理成兼容知识演化层的专家知识卡片。"""
    normalized_domain = _normalize_business_domain(domain, metadata)
    source_platform = _infer_source_platform(metadata)
    language = _infer_language(metadata, transcript_text)
    fallback_keywords = _build_keyword_fallback(metadata, normalized_domain, source_platform)

    if not DEEPSEEK_API_KEY:
        return _fallback_card(
            "未配置 DEEPSEEK_API_KEY，无法生成知识卡片。",
            metadata=metadata,
            domain=normalized_domain,
            source_platform=source_platform,
            language=language,
        )

    if len((transcript_text or "").strip()) < 10 or "❌" in transcript_text:
        return transcript_text

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
        sections, remainder = _extract_sections(content)
        _ensure_topic_normalization_section(
            sections,
            metadata,
            normalized_domain,
            source_platform,
            language,
            fallback_keywords,
        )
        _ensure_core_viewpoints_section(sections)
        _ensure_keyword_section(sections, fallback_keywords)
        return _rebuild_card_content(sections, remainder)
    except Exception as exc:
        logger.error("DeepSeek 知识卡片生成失败: %s", exc)
        return _fallback_card(
            f"知识卡片生成失败: {exc}",
            metadata=metadata,
            domain=normalized_domain,
            source_platform=source_platform,
            language=language,
        )
