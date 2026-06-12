from __future__ import annotations

import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests
from dotenv import load_dotenv

import group_words


INPUT_FILE = Path("wod.xls")
OUTPUT_DIR = Path("output")
RAW_DIR = OUTPUT_DIR / "qwen_group_raw"
MODEL = os.getenv("QWEN_TEXT_MODEL", "qwen-plus")
BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/api/v1").rstrip("/")
MAX_ITEMS_PER_CHUNK = 150
OVERLAP = 18


def main() -> None:
    load_dotenv()
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    offline = os.getenv("QWEN_GROUP_OFFLINE", "").strip() == "1"
    if not api_key and not offline:
        raise SystemExit("Missing DASHSCOPE_API_KEY.")

    OUTPUT_DIR.mkdir(exist_ok=True)
    RAW_DIR.mkdir(exist_ok=True)

    df = pd.read_excel(INPUT_FILE, engine="xlrd")
    entries = group_words.read_entries(df)
    words = sorted(entries)
    dsu = group_words.DSU(words)
    reasons: dict[frozenset[str], set[str]] = defaultdict(set)

    # Start from deterministic local word-family rules.
    for word in words:
        for base, reason in group_words.candidate_bases(word):
            if base in entries:
                dsu.union(word, base)
                reasons[frozenset((word, base))].add(reason)
    for word in words:
        for prefix in group_words.NEGATIVE_PREFIXES:
            if word.startswith(prefix) and len(word) >= len(prefix) + 5:
                base = word[len(prefix) :]
                if base in entries:
                    dsu.union(word, base)
                    reasons[frozenset((word, base))].add(f"negative prefix {prefix}-")

    chunks = build_chunks(words)
    for index, chunk in enumerate(chunks, start=1):
        raw_path = RAW_DIR / f"chunk_{index:03d}_{chunk[0]}_{chunk[-1]}.json"
        if raw_path.exists():
            result = json.loads(raw_path.read_text(encoding="utf-8"))
            print(f"[{index}/{len(chunks)}] cached {chunk[0]}..{chunk[-1]}")
        elif offline:
            continue
        else:
            print(f"[{index}/{len(chunks)}] qwen {chunk[0]}..{chunk[-1]} ({len(chunk)} words)")
            result = call_qwen(api_key, chunk, entries)
            raw_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            time.sleep(1.2)
        apply_qwen_groups(result, set(chunk), dsu, reasons)

    rows = build_rows(entries, words, dsu, reasons)
    out_df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "word_groups_qwen.csv"
    xlsx_path = OUTPUT_DIR / "word_groups_qwen.xlsx"
    out_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    out_df.to_excel(xlsx_path, index=False)

    multi = out_df[out_df["count"] > 1]
    print(f"words: {len(words)}")
    print(f"groups/pages: {len(out_df)}")
    print(f"multi-word groups: {len(multi)}")
    print(f"single-word groups: {len(out_df) - len(multi)}")
    print(f"csv: {csv_path.resolve()}")
    print(f"xlsx: {xlsx_path.resolve()}")


def build_chunks(words: list[str]) -> list[list[str]]:
    by_initial: dict[str, list[str]] = defaultdict(list)
    for word in words:
        by_initial[word[0]].append(word)

    chunks: list[list[str]] = []
    for initial in sorted(by_initial):
        items = by_initial[initial]
        if len(items) <= MAX_ITEMS_PER_CHUNK:
            chunks.append(items)
            continue
        start = 0
        while start < len(items):
            end = min(start + MAX_ITEMS_PER_CHUNK, len(items))
            chunks.append(items[start:end])
            if end == len(items):
                break
            start = max(end - OVERLAP, start + 1)
    return chunks


def call_qwen(api_key: str, chunk: list[str], entries: dict[str, dict[str, str]]) -> dict[str, Any]:
    lines = []
    for idx, word in enumerate(chunk, start=1):
        definition = entries[word]["definition"]
        lines.append(f"{idx}. {word} :: {definition}")
    prompt = f"""
你是英语六级词汇书编辑。请把下面这批词中“应该放在同一页一起记”的词分组。

分组标准：
- 合并词形变化、派生词、同一词族、明显反义前缀词，例如 able/ability/unable。
- 可以合并非常明显的同根同义链，例如 economy/economic/economics。
- 不要仅因为字母相似、短词根片段相同就合并。
- 不要把只是同一话题、近义词但词形无关的词合并。
- 只输出本批词里确实强相关的多词组；单词组不要输出。
- 每个词最多放入一个最合适的组。

返回严格 JSON：
{{
  "groups": [
    {{"words": ["word1", "word2"], "reason": "中文说明"}}
  ]
}}

词表：
{chr(10).join(lines)}
""".strip()
    url = f"{BASE_URL}/services/aigc/text-generation/generation"
    payload = {
        "model": MODEL,
        "input": {
            "messages": [
                {"role": "system", "content": "Return valid JSON only. Be conservative and accurate."},
                {"role": "user", "content": prompt},
            ]
        },
        "parameters": {"result_format": "message", "temperature": 0.1},
    }
    last_error: Exception | None = None
    for attempt in range(1, 5):
        try:
            response = requests.post(
                url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=240,
            )
            response.raise_for_status()
            content = response.json()["output"]["choices"][0]["message"]["content"]
            return parse_json(content)
        except Exception as exc:
            last_error = exc
            if attempt == 4:
                break
            print(f"  retry {attempt}/3 after {exc.__class__.__name__}")
            time.sleep(6 * attempt)
    raise RuntimeError(f"Qwen grouping failed: {last_error}") from last_error


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
        raise ValueError("Qwen returned non-object JSON")
    data.setdefault("groups", [])
    return data


def apply_qwen_groups(
    result: dict[str, Any],
    allowed: set[str],
    dsu: group_words.DSU,
    reasons: dict[frozenset[str], set[str]],
) -> None:
    for item in result.get("groups", []):
        if not isinstance(item, dict):
            continue
        words = [str(w).strip().lower() for w in item.get("words", [])]
        words = [w for w in words if w in allowed]
        words = list(dict.fromkeys(words))
        if len(words) < 2:
            continue
        reason = clean_reason(item.get("reason", "Qwen strong related group"))
        if is_negative_qwen_reason(reason):
            continue
        first = words[0]
        for word in words[1:]:
            dsu.union(first, word)
            reasons[frozenset((first, word))].add(f"Qwen: {reason}")


def clean_reason(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip())[:120]


def is_negative_qwen_reason(reason: str) -> bool:
    negative_markers = [
        "不符合",
        "不应",
        "不要",
        "不能",
        "无共同",
        "无直接",
        "词源不同",
        "拼写巧合",
        "仅为",
        "不构成",
    ]
    return any(marker in reason for marker in negative_markers)


def build_rows(
    entries: dict[str, dict[str, str]],
    words: list[str],
    dsu: group_words.DSU,
    reasons: dict[frozenset[str], set[str]],
) -> list[dict[str, Any]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for word in words:
        groups[dsu.find(word)].append(word)

    rows = []
    for page_no, group in enumerate(sorted(groups.values(), key=group_words.group_sort_key), start=1):
        group = sorted(group, key=lambda w: (len(w), w))
        headword = group_words.choose_headword(group)
        rows.append(
            {
                "group_id": f"G{page_no:04d}",
                "page_no": page_no,
                "headword": headword,
                "count": len(group),
                "words": ", ".join(group),
                "reason": group_words.infer_group_reason(group, reasons),
                "definitions": " | ".join(
                    f"{word}: {entries[word]['definition']}" for word in group if entries[word]["definition"]
                ),
            }
        )
    return rows


if __name__ == "__main__":
    main()
