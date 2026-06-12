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
DEFAULT_OUTPUT = BASE / "data" / "qwen_visual_scenes.jsonl"
DEFAULT_RAW_DIR = BASE / "data" / "qwen_visual_scenes_raw"
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
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
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
                "definitions": compact(text_value(row.get("definitions")), 360),
                "root_affix": compact(text_value(row.get("root_affix")), 160),
                "memory_link": compact(text_value(row.get("memory_link")), 220),
                "mnemonic": compact(text_value(row.get("mnemonic")), 220),
                "group_explanation": compact(text_value(row.get("group_explanation")), 220),
            }
        )
    return records


def call_deepseek(api_key: str, chunk: list[dict[str, Any]]) -> dict[str, Any]:
    prompt = {
        "task": (
            "For each CET-6 vocabulary group, write one concrete visual_scene for an image model. "
            "The scene must help memory by using one absurd, emotional, easy-to-grasp metaphor. "
            "Return JSON only."
        ),
        "strict_rules": [
            "visual_scene must be English.",
            "Do not include the target English word, its derived forms, or the Chinese definition text in visual_scene.",
            "Do not mention text, letters, words, labels, signs, posters, books, paper, boards, classrooms, charts, or speech bubbles.",
            "Use one full-frame scene, not panels or multiple mini-scenes.",
            "Describe characters, objects, actions, scale, emotion, and spatial contrast only.",
            "Make it exaggerated and 一针见血: one dominant hook that a student can remember at first glance.",
        ],
        "return_schema": {
            "items": [
                {
                    "group_id": "O0001",
                    "visual_scene": "one English sentence, 35-70 words, visual only",
                }
            ]
        },
        "groups": chunk,
    }
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": "Return valid JSON only. Follow the no-text-in-scene constraints exactly.",
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        "temperature": 0.35,
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


def scene_is_suspicious(scene: str, record: dict[str, Any]) -> bool:
    lowered = scene.lower()
    banned = [
        "letter",
        "word",
        "label",
        "sign",
        "poster",
        "book",
        "paper",
        "board",
        "classroom",
        "chart",
        "speech bubble",
        "caption",
    ]
    if any(term in lowered for term in banned):
        return True
    for word in record.get("words", []):
        word = str(word).lower()
        if len(word) >= 4 and re.search(rf"\b{re.escape(word)}\b", lowered):
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--chunk-size", type=int, default=30)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    global MODEL, BASE_URL
    load_dotenv(BASE / ".env")
    load_local_env(BASE / ".env")
    MODEL = os.environ.get("DEEPSEEK_MODEL", MODEL).strip()
    BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", BASE_URL).rstrip("/")
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Missing DEEPSEEK_API_KEY. Set it in the environment, not in this script.")

    df = pd.read_csv(args.input, encoding="utf-8-sig")
    records = build_records(df)
    if args.limit:
        records = records[: args.limit]
    by_id = {record["group_id"]: record for record in records}
    existing = read_existing(args.output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    pending = [record for record in records if record["group_id"] not in existing]
    print(f"records={len(records)} existing={len(existing)} pending={len(pending)} model={MODEL}", flush=True)

    with args.output.open("a", encoding="utf-8") as out:
        for start in range(0, len(pending), args.chunk_size):
            chunk = pending[start : start + args.chunk_size]
            if not chunk:
                break
            raw_path = args.raw_dir / f"chunk_{start // args.chunk_size + 1:04d}_{chunk[0]['group_id']}_{chunk[-1]['group_id']}.json"
            if raw_path.exists():
                data = json.loads(raw_path.read_text(encoding="utf-8"))
            else:
                print(f"deepseek chunk {start + 1}-{start + len(chunk)}", flush=True)
                data = call_deepseek(api_key, chunk)
                raw_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                time.sleep(0.8)
            for item in data.get("items", []):
                group_id = text_value(item.get("group_id"))
                scene = compact(text_value(item.get("visual_scene")), 900)
                if not group_id or group_id not in by_id or not scene:
                    continue
                out_item = {
                    "group_id": group_id,
                    "headword": by_id[group_id]["headword"],
                    "visual_scene": scene,
                    "suspicious": scene_is_suspicious(scene, by_id[group_id]),
                }
                out.write(json.dumps(out_item, ensure_ascii=False) + "\n")
                out.flush()
            print(f"written={sum(1 for _ in args.output.open(encoding='utf-8'))}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
