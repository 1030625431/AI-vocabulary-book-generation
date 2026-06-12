from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv


BASE = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = BASE / "data" / "clusters" / "optimized_vocab_clusters.csv"
DEFAULT_OUTPUT = BASE / "data" / "vocab_book_text.jsonl"
DEFAULT_RAW_DIR = BASE / "data" / "vocab_book_text_raw"

MODEL = "deepseek-v4-pro"
BASE_URL = "https://api.deepseek.com"


def text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def load_local_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def compact(text: str, limit: int) -> str:
    return re.sub(r"\s+", " ", text_value(text))[:limit].rstrip()


def split_words(text: str) -> list[str]:
    return [item.strip() for item in re.split(r"[,，;；|/]+", text_value(text)) if item.strip()]


def read_existing(path: Path) -> dict[str, dict[str, Any]]:
    items: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return items
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            group_id = text_value(item.get("group_id"))
            if group_id:
                items[group_id] = item
    return items


def build_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "group_id": text_value(row.get("optimized_group_id")),
                "headword": text_value(row.get("headword")),
                "words": split_words(text_value(row.get("all_memory_words"))),
                "source_words": split_words(text_value(row.get("source_words"))),
                "added_forms": split_words(text_value(row.get("added_forms"))),
                "definitions": compact(text_value(row.get("definitions")), 800),
                "root_affix_existing": compact(text_value(row.get("root_affix")), 400),
                "group_explanation_existing": compact(text_value(row.get("group_explanation")), 400),
                "memory_link_existing": compact(text_value(row.get("memory_link")), 400),
                "mnemonic_existing": compact(text_value(row.get("mnemonic")), 300),
                "collocations_existing": compact(text_value(row.get("collocations")), 400),
                "related_existing": compact(text_value(row.get("related_recommendations")), 400),
            }
        )
    return records


def build_prompt_rules(exam_name: str = "CET-6") -> list[str]:
    return [
        f"为《{exam_name}》词汇书生成每组单词所需的全部正文文本。返回严格 JSON。",
        "每个 group 必须返回一个对应 group_id 的 item，字段都要给出，不能省略。",
        "core_meaning 是该组所有词共享的核心中文义，不超过 30 字。",
        f"frequent_translations 列出该组在{exam_name}常考的中文译法，3-5 条，去重。",
        "word_items 必须覆盖所有提供的词，每个词都给出 phonetic（IPA 国际音标，外加 / /）、pos、translation、usage_note、example_en、example_zh，并尽量再给一个 example_en_2/example_zh_2，例句要自然且体现常用搭配。",
        "usage_note 写成简短句子，提示常见搭配/与近义词的辨析/词性变化。",
        "root_affix 至少 50 字：拆解词根词缀来源，必要时追溯拉丁/希腊/古英语来源；如果是简单词，写它的派生关系或字形记忆点。",
        "group_explanation 至少 40 字：解释为什么这些词应被归到一组。",
        "memory_link 至少 100 字：编一个生动具体、可视化的故事或画面，把整组单词串联起来，方便看到图片就能联想。",
        "mnemonic 给出一句简洁有力的中文助记口诀，可夸张但不牵强；30-80 字。",
        f"study_note 至少 60 字：写学习者最容易踩的坑，比如发音、拼写、近义混淆、文化背景或在{exam_name}题中的命题角度。",
        "collocations 至少 6 条，每条形如 'English collocation - 中文意思'。",
        "related_recommendations 至少 5 条，覆盖近义、反义、形近、易混词，每条形如 'word - 简短中文说明'。",
        f"exam_note 至少 30 字：明确指出{exam_name}真题/模考中的高频考点或题型角度。",
        "不要生成图片提示词，这里只写词书正文。",
        "如已有 root_affix/group_explanation/mnemonic 等输入，可吸收优化但要重新写满字数下限。",
        "中文使用规范汉字，不要出现乱码、问号串或不完整字符。",
    ]


SCHEMA_EXAMPLE = {
    "items": [
        {
            "group_id": "Oxxxx",
            "headword": "core_word",
            "page_title": "core_word / form2, form3",
            "core_meaning": "本组核心中文义",
            "frequent_translations": ["译法1", "译法2"],
            "word_items": [
                {
                    "word": "core_word",
                    "phonetic": "/ˈeksæmpl/",
                    "pos": "v./n./adj.",
                    "translation": "中文常用义",
                    "usage_note": "辨析/常考用法/搭配习惯",
                    "example_en": "A natural CET-6 sentence.",
                    "example_zh": "对应中文翻译",
                    "example_en_2": "Another natural sentence.",
                    "example_zh_2": "对应中文翻译",
                }
            ],
            "root_affix": "至少 50 字的词根词缀分析",
            "group_explanation": "至少 40 字的归组解释",
            "memory_link": "至少 100 字的画面联想故事",
            "mnemonic": "30-80 字的助记口诀",
            "study_note": "至少 60 字的学习者笔记",
            "collocations": ["coll1 - 含义1", "coll2 - 含义2", "coll3 - 含义3", "coll4 - 含义4", "coll5 - 含义5", "coll6 - 含义6"],
            "related_recommendations": ["w1 - 说明", "w2 - 说明", "w3 - 说明", "w4 - 说明", "w5 - 说明"],
            "exam_note": "至少 30 字的六级考点提示",
        }
    ]
}


def build_payload(chunk: list[dict[str, Any]], exam_name: str = "CET-6") -> dict[str, Any]:
    user_prompt = {
        "rules": build_prompt_rules(exam_name),
        "return_schema": SCHEMA_EXAMPLE,
        "groups": chunk,
    }
    return {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    f"你是一名资深英语教研老师，正在为《{exam_name}》的考生编写"
                    "一本视觉记忆词汇书。每个词组将占用一整页，配有夸张漫画式记忆图。"
                    f"请只输出严格 JSON，对所有字段都给出充实、地道、{exam_name}风格的内容。"
                ),
            },
            {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
        ],
        "temperature": 0.3,
        "max_tokens": 12000,
        "response_format": {"type": "json_object"},
    }


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
        raise ValueError("DeepSeek returned non-object JSON")
    data.setdefault("items", [])
    return data


def call_deepseek(api_key: str, chunk: list[dict[str, Any]], exam_name: str = "CET-6") -> dict[str, Any]:
    payload = build_payload(chunk, exam_name)
    last_error: Exception | None = None
    for attempt in range(1, 5):
        try:
            response = requests.post(
                f"{BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=300,
            )
            if response.status_code >= 400:
                print(f"deepseek error body: {response.text[:1000]}", flush=True)
            response.raise_for_status()
            body = json.loads(response.content.decode("utf-8"))
            return parse_json(body["choices"][0]["message"]["content"])
        except Exception as exc:
            last_error = exc
            if attempt == 4:
                break
            print(f"retry {attempt}/3 after {exc.__class__.__name__}: {exc}", flush=True)
            time.sleep(8 * attempt)
    raise RuntimeError(f"DeepSeek request failed: {last_error}") from last_error


def fallback_item(record: dict[str, Any], error: str) -> dict[str, Any]:
    words = record.get("words") or [record.get("headword", "")]
    definitions = record.get("definitions", "")
    return {
        "group_id": record["group_id"],
        "headword": record["headword"],
        "page_title": " / ".join(words),
        "core_meaning": definitions,
        "frequent_translations": [definitions] if definitions else [],
        "word_items": [
            {
                "word": word,
                "phonetic": "",
                "pos": "",
                "translation": definitions if len(words) == 1 else "",
                "usage_note": "",
                "example_en": "",
                "example_zh": "",
            }
            for word in words
        ],
        "root_affix": record.get("root_affix_existing", ""),
        "group_explanation": record.get("group_explanation_existing", ""),
        "memory_link": record.get("memory_link_existing", ""),
        "mnemonic": record.get("mnemonic_existing", ""),
        "study_note": "",
        "collocations": [],
        "related_recommendations": [],
        "exam_note": "",
        "fallback_error": error,
    }


def normalize_items(data: dict[str, Any], chunk: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {record["group_id"]: record for record in chunk}
    seen: set[str] = set()
    items: list[dict[str, Any]] = []
    for item in data.get("items", []):
        if not isinstance(item, dict):
            continue
        group_id = text_value(item.get("group_id"))
        if group_id not in by_id or group_id in seen:
            continue
        item["group_id"] = group_id
        item.setdefault("headword", by_id[group_id]["headword"])
        item.setdefault("source_words", by_id[group_id].get("source_words", []))
        item.setdefault("added_forms", by_id[group_id].get("added_forms", []))
        item.setdefault("words", by_id[group_id].get("words", []))
        items.append(item)
        seen.add(group_id)
    missing = [record for record in chunk if record["group_id"] not in seen]
    return items, missing


def generate_chunk(api_key: str, chunk: list[dict[str, Any]], depth: int = 0,
                   exam_name: str = "CET-6") -> list[dict[str, Any]]:
    try:
        data = call_deepseek(api_key, chunk, exam_name)
    except Exception as exc:
        if len(chunk) == 1 or depth >= 3:
            print(f"fallback {chunk[0]['group_id']} after {exc}", flush=True)
            return [fallback_item(c, repr(exc)) for c in chunk]
        mid = len(chunk) // 2
        print(f"split chunk size={len(chunk)} after {exc}", flush=True)
        return generate_chunk(api_key, chunk[:mid], depth + 1, exam_name) + generate_chunk(api_key, chunk[mid:], depth + 1, exam_name)

    items, missing = normalize_items(data, chunk)
    if missing:
        if len(missing) == len(chunk):
            if len(chunk) == 1 or depth >= 3:
                print(f"fallback (empty) {chunk[0]['group_id']}", flush=True)
                return [fallback_item(c, "missing_from_response") for c in chunk]
            mid = len(chunk) // 2
            print(f"split chunk (empty) size={len(chunk)}", flush=True)
            return generate_chunk(api_key, chunk[:mid], depth + 1, exam_name) + generate_chunk(api_key, chunk[mid:], depth + 1, exam_name)
        print(f"retry missing {len(missing)}/{len(chunk)} ids={[m['group_id'] for m in missing]}", flush=True)
        items.extend(generate_chunk(api_key, missing, depth + 1, exam_name))
    return items


def main() -> int:
    parser = argparse.ArgumentParser(
        description="用 DeepSeek API 为词组 CSV 批量生成词汇书正文文本"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only-ids", type=str, default="", help="逗号分隔的 group_id，只生成这些组")
    parser.add_argument("--exam-name", default="CET-6",
                        help="考试/词表名称，嵌入 prompt（如 CET-4、考研、IELTS）")
    args = parser.parse_args()

    global MODEL, BASE_URL
    load_dotenv(BASE / ".env")
    load_local_env(BASE / ".env")
    MODEL = os.environ.get("DEEPSEEK_MODEL", MODEL).strip()
    BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", BASE_URL).rstrip("/")
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Missing DEEPSEEK_API_KEY")

    df = pd.read_csv(args.input, encoding="utf-8-sig")
    records = build_records(df)
    if args.only_ids:
        ids = {x.strip() for x in args.only_ids.split(",") if x.strip()}
        records = [r for r in records if r["group_id"] in ids]
    elif args.limit:
        records = records[: args.limit]
    existing = read_existing(args.output)
    pending = [record for record in records if record["group_id"] not in existing]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"records={len(records)} existing={len(existing)} pending={len(pending)} model={MODEL}", flush=True)

    with args.output.open("a", encoding="utf-8") as out:
        for start in range(0, len(pending), args.chunk_size):
            chunk = pending[start : start + args.chunk_size]
            if not chunk:
                break
            print(f"text chunk {start + 1}-{start + len(chunk)}", flush=True)
            items = generate_chunk(api_key, chunk, exam_name=args.exam_name)
            raw_path = args.raw_dir / f"chunk_{start // args.chunk_size + 1:04d}_{chunk[0]['group_id']}_{chunk[-1]['group_id']}.json"
            raw_path.write_text(json.dumps({"items": items}, ensure_ascii=False, indent=2), encoding="utf-8")
            for item in items:
                out.write(json.dumps(item, ensure_ascii=False) + "\n")
            out.flush()
            written = sum(1 for _ in args.output.open(encoding="utf-8", errors="replace"))
            print(f"written={written}", flush=True)
            time.sleep(0.4)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
