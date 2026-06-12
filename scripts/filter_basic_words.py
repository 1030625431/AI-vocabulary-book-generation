from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv
from wordfreq import top_n_list, zipf_frequency


INPUT_FILE = Path("wod.xls")
OUTPUT_DIR = Path("output")
RAW_DIR = OUTPUT_DIR / "basic_filter_raw"
CHUNK_SIZE = 120
MODEL = os.getenv("QWEN_TEXT_MODEL", "qwen-plus")
BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/api/v1").rstrip("/")
COMMON_RANK = {word: rank for rank, word in enumerate(top_n_list("en", 50000), start=1)}


def main() -> None:
    load_dotenv()
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    offline = os.getenv("BASIC_FILTER_OFFLINE", "").strip() == "1"
    if not api_key and not offline:
        raise SystemExit("Missing DASHSCOPE_API_KEY.")

    OUTPUT_DIR.mkdir(exist_ok=True)
    RAW_DIR.mkdir(exist_ok=True)

    df = pd.read_excel(INPUT_FILE, engine="xlrd")
    rows = []
    for idx, row in df.iterrows():
        word = str(row.iloc[1]).strip()
        if not word or word.lower() == "nan":
            continue
        rows.append(
            {
                "row_index": idx,
                "word": word,
                "word_lower": word.lower(),
                "definition": clean_cell(row.iloc[3]) if len(row) > 3 else "",
            }
        )

    decisions: dict[str, dict[str, Any]] = {}
    chunks = [rows[i : i + CHUNK_SIZE] for i in range(0, len(rows), CHUNK_SIZE)]
    for i, chunk in enumerate(chunks, start=1):
        raw_path = RAW_DIR / f"chunk_{i:03d}_{chunk[0]['word_lower']}_{chunk[-1]['word_lower']}.json"
        if raw_path.exists():
            result = json.loads(raw_path.read_text(encoding="utf-8"))
            print(f"[{i}/{len(chunks)}] cached {chunk[0]['word']}..{chunk[-1]['word']}", flush=True)
        elif offline:
            continue
        else:
            print(f"[{i}/{len(chunks)}] classify {chunk[0]['word']}..{chunk[-1]['word']}", flush=True)
            result = call_qwen(api_key, chunk)
            raw_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            time.sleep(0.8)
        for item in result.get("remove", []):
            if not isinstance(item, dict):
                continue
            word = str(item.get("word", "")).strip().lower()
            if not word:
                continue
            decisions[word] = {
                "remove_reason": clean_cell(item.get("reason", "")),
                "school_level": clean_cell(item.get("level", "初高中基础")),
                "confidence": clean_cell(item.get("confidence", "high")),
            }

    for item in rows:
        word = item["word_lower"]
        if word not in decisions and is_obvious_basic(word, item["definition"]):
            freq = zipf_frequency(word, "en")
            decisions[word] = {
                "remove_reason": f"本地规则：高频基础词，常见度 zipf={freq:.2f}，释义偏日常/基础",
                "school_level": "初高中基础",
                "confidence": "medium",
            }

    write_outputs(df, decisions)


def call_qwen(api_key: str, chunk: list[dict[str, str]]) -> dict[str, Any]:
    lines = []
    for idx, item in enumerate(chunk, start=1):
        lines.append(f"{idx}. {item['word']} :: {item['definition']}")
    prompt = f"""
你是中国英语教辅编辑。请从下面词表中找出“明显属于初高中基础水平、没必要放进六级视觉词汇书”的词。

删除标准：
- 只删除非常明显的初中/高中基础词，例如 youth, about, above, able, school, family, water, good 这类。
- 常见功能词、日常生活核心词、基础动作词、基础形容词可以删除。
- 宁可少删，不要误删六级常考抽象词、学术词、专业词、低频词。
- 如果一个词虽然中学可能见过，但六级里有重要抽象义或正式用法，不要删除。
- 只输出要删除的词；保留的词不要输出。

返回严格 JSON：
{{
  "remove": [
    {{"word": "youth", "level": "初高中基础", "confidence": "high", "reason": "常见基础名词，初高中已掌握"}}
  ]
}}

待判断词表：
{chr(10).join(lines)}
""".strip()
    payload = {
        "model": MODEL,
        "input": {
            "messages": [
                {"role": "system", "content": "Return valid JSON only. Be conservative."},
                {"role": "user", "content": prompt},
            ]
        },
        "parameters": {"result_format": "message", "temperature": 0.05},
    }
    url = f"{BASE_URL}/services/aigc/text-generation/generation"
    last_error: Exception | None = None
    for attempt in range(1, 5):
        try:
            response = requests.post(
                url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=180,
            )
            response.raise_for_status()
            content = response.json()["output"]["choices"][0]["message"]["content"]
            return parse_json(content)
        except Exception as exc:
            last_error = exc
            if attempt >= 4:
                break
            print(f"  retry {attempt}/3 after {exc.__class__.__name__}", flush=True)
            time.sleep(5 * attempt)
    raise RuntimeError(f"Qwen classification failed: {last_error}") from last_error


def parse_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        data = json.loads(match.group(0), strict=False)
    if not isinstance(data, dict):
        raise ValueError("Model returned non-object JSON.")
    data.setdefault("remove", [])
    return data


def write_outputs(df: pd.DataFrame, decisions: dict[str, dict[str, Any]]) -> None:
    work = df.copy()
    word_lower = work.iloc[:, 1].astype(str).str.strip().str.lower()
    removed_mask = word_lower.isin(decisions)

    reasons = []
    levels = []
    confidences = []
    for word in word_lower:
        decision = decisions.get(word, {})
        reasons.append(decision.get("remove_reason", ""))
        levels.append(decision.get("school_level", ""))
        confidences.append(decision.get("confidence", ""))

    annotated = work.copy()
    annotated["删除原因"] = reasons
    annotated["判断级别"] = levels
    annotated["置信度"] = confidences

    removed = annotated[removed_mask].copy()
    remaining = work[~removed_mask].copy()

    removed_xlsx = OUTPUT_DIR / "removed_basic_words.xlsx"
    removed_csv = OUTPUT_DIR / "removed_basic_words.csv"
    remaining_xlsx = OUTPUT_DIR / "word_filtered.xlsx"
    remaining_csv = OUTPUT_DIR / "word_filtered.csv"
    removed.to_excel(removed_xlsx, index=False)
    removed.to_csv(removed_csv, index=False, encoding="utf-8-sig")
    remaining.to_excel(remaining_xlsx, index=False)
    remaining.to_csv(remaining_csv, index=False, encoding="utf-8-sig")

    print(f"original rows: {len(work)}", flush=True)
    print(f"removed basic rows: {len(removed)}", flush=True)
    print(f"remaining rows: {len(remaining)}", flush=True)
    print(f"removed xlsx: {removed_xlsx.resolve()}", flush=True)
    print(f"remaining xlsx: {remaining_xlsx.resolve()}", flush=True)


def clean_cell(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    return re.sub(r"\s+", " ", text)


BASIC_ALWAYS = {
    "a",
    "an",
    "the",
    "about",
    "above",
    "across",
    "after",
    "again",
    "against",
    "age",
    "ago",
    "air",
    "all",
    "almost",
    "alone",
    "along",
    "already",
    "also",
    "although",
    "always",
    "among",
    "and",
    "animal",
    "another",
    "answer",
    "any",
    "apple",
    "area",
    "arm",
    "around",
    "as",
    "ask",
    "at",
    "away",
    "back",
    "bad",
    "bag",
    "ball",
    "bank",
    "beautiful",
    "because",
    "bed",
    "before",
    "begin",
    "behind",
    "below",
    "beside",
    "best",
    "better",
    "between",
    "big",
    "bird",
    "black",
    "blue",
    "body",
    "book",
    "boy",
    "bread",
    "break",
    "bring",
    "brother",
    "brown",
    "bus",
    "but",
    "buy",
    "by",
    "cake",
    "call",
    "can",
    "car",
    "carry",
    "cat",
    "chair",
    "child",
    "city",
    "class",
    "clean",
    "close",
    "cloud",
    "coat",
    "cold",
    "come",
    "cook",
    "cool",
    "country",
    "cut",
    "dark",
    "day",
    "desk",
    "die",
    "do",
    "doctor",
    "dog",
    "door",
    "down",
    "draw",
    "drink",
    "drive",
    "each",
    "ear",
    "early",
    "east",
    "easy",
    "eat",
    "egg",
    "eight",
    "either",
    "eleven",
    "else",
    "end",
    "enough",
    "even",
    "evening",
    "ever",
    "every",
    "eye",
    "face",
    "family",
    "far",
    "farm",
    "fast",
    "father",
    "few",
    "field",
    "fifteen",
    "find",
    "fine",
    "fire",
    "first",
    "fish",
    "five",
    "floor",
    "flower",
    "fly",
    "food",
    "foot",
    "for",
    "four",
    "free",
    "friend",
    "from",
    "front",
    "fruit",
    "full",
    "game",
    "garden",
    "girl",
    "give",
    "go",
    "good",
    "grade",
    "grass",
    "great",
    "green",
    "ground",
    "grow",
    "hair",
    "half",
    "hand",
    "happy",
    "hard",
    "have",
    "he",
    "head",
    "hear",
    "hello",
    "help",
    "her",
    "here",
    "high",
    "him",
    "his",
    "home",
    "horse",
    "hot",
    "hour",
    "house",
    "how",
    "hundred",
    "I",
    "ice",
    "if",
    "in",
    "inside",
    "into",
    "it",
    "job",
    "jump",
    "keep",
    "key",
    "kind",
    "king",
    "kitchen",
    "know",
    "lake",
    "land",
    "large",
    "last",
    "late",
    "laugh",
    "learn",
    "leave",
    "left",
    "leg",
    "lesson",
    "letter",
    "life",
    "light",
    "like",
    "line",
    "listen",
    "little",
    "live",
    "long",
    "look",
    "love",
    "low",
    "make",
    "man",
    "many",
    "map",
    "market",
    "may",
    "me",
    "meal",
    "meet",
    "middle",
    "milk",
    "minute",
    "miss",
    "money",
    "month",
    "moon",
    "morning",
    "mother",
    "mountain",
    "mouth",
    "move",
    "much",
    "music",
    "must",
    "my",
    "name",
    "near",
    "never",
    "new",
    "next",
    "night",
    "nine",
    "no",
    "north",
    "nose",
    "not",
    "now",
    "number",
    "of",
    "off",
    "often",
    "old",
    "on",
    "once",
    "one",
    "only",
    "open",
    "or",
    "orange",
    "other",
    "our",
    "out",
    "outside",
    "over",
    "own",
    "page",
    "paper",
    "park",
    "part",
    "party",
    "pass",
    "pen",
    "people",
    "picture",
    "place",
    "play",
    "please",
    "poor",
    "put",
    "quick",
    "rain",
    "read",
    "red",
    "rice",
    "rich",
    "ride",
    "right",
    "river",
    "road",
    "room",
    "run",
    "sad",
    "say",
    "school",
    "sea",
    "second",
    "see",
    "seven",
    "she",
    "ship",
    "shoe",
    "shop",
    "short",
    "show",
    "sing",
    "sister",
    "sit",
    "six",
    "sky",
    "sleep",
    "slow",
    "small",
    "snow",
    "so",
    "some",
    "son",
    "song",
    "soon",
    "sorry",
    "sound",
    "south",
    "speak",
    "spring",
    "stand",
    "star",
    "start",
    "stay",
    "story",
    "street",
    "strong",
    "student",
    "study",
    "summer",
    "sun",
    "sure",
    "swim",
    "table",
    "take",
    "talk",
    "tea",
    "teacher",
    "tell",
    "ten",
    "than",
    "that",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "thing",
    "think",
    "third",
    "this",
    "those",
    "three",
    "through",
    "time",
    "to",
    "today",
    "together",
    "tomorrow",
    "too",
    "tree",
    "try",
    "turn",
    "twelve",
    "two",
    "under",
    "up",
    "us",
    "use",
    "very",
    "visit",
    "wait",
    "walk",
    "wall",
    "want",
    "warm",
    "wash",
    "watch",
    "water",
    "way",
    "we",
    "weather",
    "week",
    "well",
    "west",
    "what",
    "when",
    "where",
    "which",
    "white",
    "who",
    "why",
    "wife",
    "will",
    "wind",
    "window",
    "winter",
    "with",
    "woman",
    "word",
    "work",
    "world",
    "write",
    "wrong",
    "year",
    "yellow",
    "yes",
    "yesterday",
    "you",
    "young",
    "your",
    "youth",
}


PROTECTED_SUFFIXES = (
    "tion",
    "sion",
    "ment",
    "ance",
    "ence",
    "ity",
    "ism",
    "ist",
    "ive",
    "ous",
    "cial",
    "tial",
    "ical",
    "ology",
    "graphy",
)


PROTECTED_WORDS = {
    "abrupt",
    "absorb",
    "absurd",
    "abundance",
    "abundant",
    "accelerate",
    "accumulate",
    "accommodation",
    "accordance",
    "accordingly",
    "economic",
    "economy",
    "government",
    "hierarchy",
    "deteriorate",
    "vulnerable",
    "compromise",
    "endeavor",
    "quota",
    "radiant",
}


HIGH_SCHOOL_ALWAYS = {
    "abandon",
    "ability",
    "able",
    "abnormal",
    "aboard",
    "abolish",
    "abroad",
    "absence",
    "absent",
    "absolute",
    "absolutely",
    "abstract",
    "academic",
    "academy",
    "accent",
    "accept",
    "acceptable",
    "acceptance",
    "accident",
    "accidental",
    "accompany",
    "accomplish",
    "account",
    "accuracy",
    "accurate",
    "accuse",
    "accustom",
    "accustomed",
    "achieve",
    "achievement",
}


def is_obvious_basic(word: str, definition: str) -> bool:
    lower = word.lower()
    if lower in HIGH_SCHOOL_ALWAYS:
        return True
    if lower in PROTECTED_WORDS:
        return False
    if lower in BASIC_ALWAYS:
        return True
    if lower.endswith(PROTECTED_SUFFIXES):
        return False
    freq = zipf_frequency(lower, "en")
    if freq < 4.85:
        return False
    if len(lower) <= 5:
        return True
    if len(lower) <= 8 and looks_daily_basic(definition):
        return True
    rank = COMMON_RANK.get(lower, 999999)
    if rank <= 16000 and looks_exam_basic(definition):
        return True
    return False


def looks_daily_basic(definition: str) -> bool:
    daily_markers = [
        "家庭",
        "父",
        "母",
        "儿",
        "女",
        "孩子",
        "青年",
        "青春",
        "身体",
        "头",
        "手",
        "脚",
        "眼",
        "口",
        "食",
        "水",
        "饭",
        "学校",
        "学生",
        "老师",
        "房",
        "门",
        "窗",
        "桌",
        "椅",
        "颜色",
        "动物",
        "天气",
        "时间",
        "今天",
        "昨天",
        "明天",
        "星期",
        "月份",
        "数字",
        "衣",
        "鞋",
        "路",
        "街",
        "城市",
        "国家",
        "朋友",
        "年轻",
        "高兴",
        "快乐",
        "大的",
        "小的",
        "好的",
        "坏的",
        "热",
        "冷",
        "走",
        "跑",
        "看",
        "听",
        "说",
        "吃",
        "喝",
        "买",
        "卖",
        "玩",
        "学习",
    ]
    return any(marker in definition for marker in daily_markers)


def looks_exam_basic(definition: str) -> bool:
    exam_markers = [
        "能力",
        "才能",
        "能够",
        "放弃",
        "遗弃",
        "离开",
        "正常",
        "反常",
        "不规则",
        "废除",
        "取消",
        "国外",
        "海外",
        "缺席",
        "缺乏",
        "绝对",
        "完全",
        "抽象",
        "摘要",
        "学术",
        "学院",
        "口音",
        "接受",
        "可接受",
        "接纳",
        "事故",
        "意外",
        "陪伴",
        "陪同",
        "完成",
        "账户",
        "账目",
        "理由",
        "描述",
        "准确",
        "精确",
        "指责",
        "控告",
        "习惯",
        "成就",
        "达到",
        "实现",
        "行为",
        "行动",
        "活动",
        "实际",
        "事实上",
        "增加",
        "另外",
        "地址",
        "承认",
        "优势",
        "冒险",
        "建议",
        "害怕",
        "同意",
        "空气",
        "允许",
        "几乎",
        "单独",
        "已经",
        "虽然",
        "总是",
        "愤怒",
        "生气",
        "答案",
        "任何",
        "公寓",
        "苹果",
        "接近",
        "四月",
        "地区",
        "军队",
        "艺术",
        "攻击",
        "平均",
        "避免",
    ]
    return any(marker in definition for marker in exam_markers)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
