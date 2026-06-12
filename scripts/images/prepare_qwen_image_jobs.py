from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


BASE = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = BASE / "data" / "clusters" / "optimized_vocab_clusters.csv"
DEFAULT_OUTPUT = BASE / "data" / "qwen_image_jobs.jsonl"


def text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def split_words(text: str) -> list[str]:
    words = []
    for item in re.split(r"[,，;；|/]+", text):
        item = item.strip()
        if item:
            words.append(item)
    return words


def slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text.strip().lower()).strip("_")
    return slug or "word_group"


def compact(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text_value(text))
    return text[:limit].rstrip()


def visual_safe_learning_text(text: str) -> str:
    text = text_value(text)
    text = re.sub(r"\b[A-Za-z][A-Za-z' -]{1,40}:\s*", "", text)
    text = re.sub(r"\b(adj|adv|n|v|vt|vi|prep|conj|pron)\.\s*", "", text)
    return compact(text, 420)


def build_prompt(row: pd.Series) -> str:
    visual_scene = compact(text_value(row.get("visual_scene")), 900)
    if visual_scene:
        return (
            "A full-frame, completely wordless, high-quality 3D animated film still. "
            f"{visual_scene} "
            "Use one dominant focal action, absurd scale, strong emotion, and simple spatial contrast. "
            "Every surface is plain, natural, and texture-only."
        )

    headword = text_value(row.get("headword"))
    words = split_words(text_value(row.get("all_memory_words"))) or [headword]
    definitions = visual_safe_learning_text(text_value(row.get("definitions")))
    root_affix = compact(text_value(row.get("root_affix")), 180)
    memory_link = compact(text_value(row.get("memory_link")), 260)
    mnemonic = compact(text_value(row.get("mnemonic")), 260)
    group_explanation = compact(text_value(row.get("group_explanation")), 260)

    learning_signal = "; ".join(
        part
        for part in [
            f"core meaning: {definitions}",
            f"root or affix clue: {root_affix}",
            f"memory link: {memory_link}",
            f"mnemonic: {mnemonic}",
            f"group relation: {group_explanation}",
        ]
        if part and not part.endswith(": ")
    )

    return (
        "A completely wordless cinematic cartoon scene. Show only characters, objects, actions, colors, and emotions. "
        "Do not create a poster, book page, worksheet, infographic, chart, classroom board, signboard, label, banner, "
        "speech bubble, comic text, user interface, logo, or watermark. "
        "The picture must encode a vocabulary meaning through one single absurd visual metaphor, not through spelling. "
        "Make it bizarre and instantly memorable in one glance: one dominant focal action, oversized objects, impossible "
        "contrast, strong facial expression, and clear cause-and-effect. "
        f"Visual idea to encode: {learning_signal}. "
        "If several related forms belong together, show the shared idea as one central object transforming, "
        "not as separate panels. Every surface in the scene must be blank and texture-only; no marks that resemble writing."
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--visual-scenes", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    df = pd.read_csv(args.input, encoding="utf-8-sig")
    if args.visual_scenes:
        scenes = load_visual_scenes(args.visual_scenes)
        df["visual_scene"] = [
            scenes.get(text_value(row.get("optimized_group_id")), "") for _, row in df.iterrows()
        ]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with args.output.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            group_id = text_value(row.get("optimized_group_id"))
            headword = text_value(row.get("headword"))
            words = split_words(text_value(row.get("all_memory_words"))) or [headword]
            job = {
                "group_id": group_id,
                "headword": headword,
                "words": words,
                "filename": f"{group_id}_{slugify(headword)}.png",
                "definitions": text_value(row.get("definitions")),
                "root_affix": text_value(row.get("root_affix")),
                "group_explanation": text_value(row.get("group_explanation")),
                "memory_link": text_value(row.get("memory_link")),
                "mnemonic": text_value(row.get("mnemonic")),
                "prompt": build_prompt(row),
            }
            f.write(json.dumps(job, ensure_ascii=False) + "\n")
            count += 1

    print(f"wrote {count} jobs to {args.output}")
    return 0


def load_visual_scenes(path: Path) -> dict[str, str]:
    scenes: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            group_id = text_value(item.get("group_id"))
            scene = text_value(item.get("visual_scene"))
            if group_id and scene:
                scenes[group_id] = scene
    return scenes


if __name__ == "__main__":
    raise SystemExit(main())
