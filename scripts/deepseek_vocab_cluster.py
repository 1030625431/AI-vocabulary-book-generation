from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv


INPUT_FILE = Path(os.environ.get("VOCAB_CLUSTER_INPUT", "output/word_filtered_3500_keep200.xlsx"))
OUTPUT_DIR = Path(os.environ.get("VOCAB_CLUSTER_OUTPUT", "output/deepseek_vocab_cluster"))
RAW_DIR = OUTPUT_DIR / "raw"
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro")
BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
CHUNK_SIZE = int(os.environ.get("VOCAB_CLUSTER_CHUNK_SIZE", "80"))
OVERLAP = int(os.environ.get("VOCAB_CLUSTER_OVERLAP", "10"))


def main() -> None:
    load_dotenv()
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Missing DEEPSEEK_API_KEY")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(INPUT_FILE)
    entries = read_entries(df)
    words = sorted(entries)
    dsu = DSU(words)
    group_notes: dict[str, dict[str, Any]] = {}

    chunks = build_chunks(words)
    write_status(f"start words={len(words)} chunks={len(chunks)} model={MODEL}")
    for index, chunk in enumerate(chunks, start=1):
        raw_path = RAW_DIR / f"chunk_{index:03d}_{chunk[0]}_{chunk[-1]}.json"
        if raw_path.exists():
            result = json.loads(raw_path.read_text(encoding="utf-8"))
            print(f"[{index}/{len(chunks)}] cached {chunk[0]}..{chunk[-1]}", flush=True)
        else:
            print(f"[{index}/{len(chunks)}] deepseek {chunk[0]}..{chunk[-1]}", flush=True)
            result = call_deepseek(api_key, chunk, entries)
            raw_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            time.sleep(0.8)
        apply_groups(result, set(chunk), dsu, group_notes)
        if index % 5 == 0 or index == len(chunks):
            write_outputs(entries, words, dsu, group_notes)
            write_status(f"progress chunk={index}/{len(chunks)}")
    write_outputs(entries, words, dsu, group_notes)
    write_status("complete")


def read_entries(df: pd.DataFrame) -> dict[str, dict[str, str]]:
    entries: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        word = str(row.iloc[1]).strip().lower()
        if not re.fullmatch(r"[a-z][a-z'-]*", word):
            continue
        entries[word] = {
            "word": str(row.iloc[1]).strip(),
            "phonetic": clean_cell(row.iloc[2]) if len(row) > 2 else "",
            "definition": clean_cell(row.iloc[3]) if len(row) > 3 else "",
        }
    return entries


def build_chunks(words: list[str]) -> list[list[str]]:
    chunks: list[list[str]] = []
    start = 0
    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunks.append(words[start:end])
        if end == len(words):
            break
        start = max(end - OVERLAP, start + 1)
    return chunks


def call_deepseek(api_key: str, chunk: list[str], entries: dict[str, dict[str, str]]) -> dict[str, Any]:
    lines = []
    for idx, word in enumerate(chunk, start=1):
        item = entries[word]
        lines.append(f"{idx}. {word} :: {item['definition']}")
    prompt = f"""
你是六级词汇书编辑。请把下面这一批词中适合“一页一起记”的词聚成组，并给出记忆材料。

分组标准：
- 同词族、词根词缀强相关、派生词、反义前缀、常见搭配关联可以同组。
- 词根词缀不是硬性标准；只要你认为能辅助记忆、能建立稳定联想，也可以合并。
- 只按强相关分组；不要只因为主题相近或字母相似就硬合并。
- 每个词最多放进一个最合适的组。
- 单词如果不适合与本批其他词合并，可以不输出。
- 输出必须用于词汇书，所以说明要短、准、适合学生记忆。

返回严格 JSON，不要 Markdown：
{{
  "groups": [
    {{
      "headword": "核心词",
      "words": ["word1", "word2"],
      "root_affix": "词根词缀或构词关系说明；不能强拆就说明整体记忆",
      "group_explanation": "为什么这些词应该一起记",
      "memory_link": "如何把这一组联系到一个画面或故事里",
      "mnemonic": "一句中文助记语",
      "collocations": ["word/phrase - 中文搭配记忆"],
      "related_recommendations": ["建议顺便记的相关词或辨析点"]
    }}
  ]
}}

词表：
{chr(10).join(lines)}
""".strip()
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Return valid JSON only. Be accurate and conservative."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 8192,
        "response_format": {"type": "json_object"},
    }
    last_error: Exception | None = None
    for attempt in range(1, 5):
        try:
            response = requests.post(
                f"{BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=240,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            try:
                return parse_json(content)
            except Exception:
                debug_path = RAW_DIR / f"bad_response_{int(time.time())}.txt"
                debug_path.write_text(content or "", encoding="utf-8", errors="replace")
                raise
        except Exception as exc:
            last_error = exc
            if attempt >= 4:
                break
            print(f"  retry {attempt}/3 after {exc.__class__.__name__}: {exc}", flush=True)
            time.sleep(8 * attempt)
    raise RuntimeError(f"DeepSeek request failed: {last_error}") from last_error


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
    data.setdefault("groups", [])
    return data


def apply_groups(
    result: dict[str, Any],
    allowed: set[str],
    dsu: "DSU",
    group_notes: dict[str, dict[str, Any]],
) -> None:
    for group in result.get("groups", []):
        if not isinstance(group, dict):
            continue
        words = [str(word).strip().lower() for word in group.get("words", [])]
        words = [word for word in words if word in allowed]
        words = list(dict.fromkeys(words))
        if len(words) < 2:
            continue
        root = words[0]
        for word in words[1:]:
            dsu.union(root, word)
        note_key = dsu.find(root)
        group_notes[note_key] = group


def write_outputs(
    entries: dict[str, dict[str, str]],
    words: list[str],
    dsu: "DSU",
    group_notes: dict[str, dict[str, Any]],
) -> None:
    grouped: dict[str, list[str]] = defaultdict(list)
    for word in words:
        grouped[dsu.find(word)].append(word)
    rows = []
    for idx, group_words in enumerate(sorted(grouped.values(), key=lambda xs: (xs[0], len(xs))), start=1):
        group_words = sorted(group_words)
        note = find_note(group_words, dsu, group_notes)
        definitions = " | ".join(f"{word}: {entries[word]['definition']}" for word in group_words)
        rows.append(
            {
                "group_id": f"D{idx:04d}",
                "headword": note.get("headword") or group_words[0],
                "count": len(group_words),
                "words": ", ".join(group_words),
                "root_affix": clean_cell(note.get("root_affix", "")),
                "group_explanation": clean_cell(note.get("group_explanation", "")),
                "memory_link": clean_cell(note.get("memory_link", "")),
                "mnemonic": clean_cell(note.get("mnemonic", "")),
                "collocations": join_list(note.get("collocations", [])),
                "related_recommendations": join_list(note.get("related_recommendations", [])),
                "definitions": definitions,
            }
        )
    out = pd.DataFrame(rows)
    out.to_excel(OUTPUT_DIR / "deepseek_vocab_clusters.xlsx", index=False)
    out.to_csv(OUTPUT_DIR / "deepseek_vocab_clusters.csv", index=False, encoding="utf-8-sig")


def find_note(group_words: list[str], dsu: "DSU", notes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    for word in group_words:
        key = dsu.find(word)
        if key in notes:
            return notes[key]
    return {}


def join_list(value: Any) -> str:
    if isinstance(value, list):
        return " | ".join(clean_cell(item) for item in value)
    return clean_cell(value)


def clean_cell(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    return re.sub(r"\s+", " ", text)


def write_status(text: str) -> None:
    (OUTPUT_DIR / "status.txt").write_text(f"{time.strftime('%F %T')} {text}\n", encoding="utf-8")


class DSU:
    def __init__(self, items: list[str]):
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: str, b: str) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        write_status(f"error {type(exc).__name__}: {exc}")
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
