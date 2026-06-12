from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import fitz
import pandas as pd
from wordfreq import top_n_list, zipf_frequency


WORD_XLS = Path("wod.xls")
HIGH_SCHOOL_PDF = Path("3500.pdf")
OUTPUT_DIR = Path("output")
KEEP_HARD_LIMIT = int(__import__("os").environ.get("KEEP_HARD_LIMIT", "500"))
COMMON_RANK = {word: rank for rank, word in enumerate(top_n_list("en", 60000), start=1)}


FORMAL_SUFFIXES = (
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
    "ary",
    "ory",
    "ial",
    "ical",
    "ate",
    "ize",
    "ise",
    "ology",
)


FORMAL_MARKERS = (
    "抽象",
    "制度",
    "理论",
    "学术",
    "行政",
    "管理",
    "政策",
    "经济",
    "法律",
    "科学",
    "技术",
    "心理",
    "社会",
    "政治",
    "哲学",
    "医学",
    "金融",
    "正式",
    "专业",
    "现象",
    "过程",
    "性质",
    "状态",
)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    word_df = pd.read_excel(WORD_XLS, engine="xlrd")
    high_school_words = extract_high_school_words(HIGH_SCHOOL_PDF)

    word_col = word_df.columns[1]
    word_lower = word_df[word_col].astype(str).str.strip().str.lower()
    high_school_mask = word_lower.isin(high_school_words)
    high_school_df = word_df[high_school_mask].copy()

    scored = []
    for idx, row in high_school_df.iterrows():
        word = str(row[word_col]).strip().lower()
        definition = clean_cell(row.iloc[3]) if len(row) > 3 else ""
        scored.append((difficulty_score(word, definition), idx, word))

    scored.sort(reverse=True)
    keep_indices = {idx for _, idx, _ in scored[:KEEP_HARD_LIMIT]}
    remove_indices = set(high_school_df.index) - keep_indices

    hard_keep = annotate(high_school_df.loc[sorted(keep_indices)].copy(), "保留：3500高中词中相对较难，保留进词汇书")
    removed = annotate(high_school_df.loc[sorted(remove_indices)].copy(), "删除：3500高中词表常见词，未进入保留的500个难词")
    remaining = word_df.drop(index=sorted(remove_indices)).copy()

    hard_keep = add_scores(hard_keep)
    removed = add_scores(removed)

    suffix = f"keep{KEEP_HARD_LIMIT}"
    removed.to_excel(OUTPUT_DIR / f"removed_3500_common_words_{suffix}.xlsx", index=False)
    removed.to_csv(OUTPUT_DIR / f"removed_3500_common_words_{suffix}.csv", index=False, encoding="utf-8-sig")
    hard_keep.to_excel(OUTPUT_DIR / f"kept_3500_hard_{KEEP_HARD_LIMIT}.xlsx", index=False)
    hard_keep.to_csv(OUTPUT_DIR / f"kept_3500_hard_{KEEP_HARD_LIMIT}.csv", index=False, encoding="utf-8-sig")
    remaining.to_excel(OUTPUT_DIR / f"word_filtered_3500_{suffix}.xlsx", index=False)
    remaining.to_csv(OUTPUT_DIR / f"word_filtered_3500_{suffix}.csv", index=False, encoding="utf-8-sig")

    print(f"3500 pdf words extracted: {len(high_school_words)}")
    print(f"original rows: {len(word_df)}")
    print(f"rows also in 3500 pdf: {len(high_school_df)}")
    print(f"kept hard high-school rows: {len(hard_keep)}")
    print(f"removed common high-school rows: {len(removed)}")
    print(f"remaining rows: {len(remaining)}")
    print((OUTPUT_DIR / f"removed_3500_common_words_{suffix}.xlsx").resolve())
    print((OUTPUT_DIR / f"kept_3500_hard_{KEEP_HARD_LIMIT}.xlsx").resolve())
    print((OUTPUT_DIR / f"word_filtered_3500_{suffix}.xlsx").resolve())


def extract_high_school_words(pdf_path: Path) -> set[str]:
    words: set[str] = set()
    doc = fitz.open(pdf_path)
    for page in doc:
        lines = [clean_line(line) for line in page.get_text("text").splitlines()]
        lines = [line for line in lines if line]
        for idx, line in enumerate(lines[:-1]):
            if re.fullmatch(r"\d{1,4}", line):
                for word in expand_word(lines[idx + 1]):
                    words.add(word)
            else:
                match = re.match(r"^(\d{1,4})\s+(.+)$", line)
                if match:
                    for word in expand_word(match.group(2)):
                        words.add(word)
    return {word for word in words if re.fullmatch(r"[a-z][a-z'-]*", word)}


def expand_word(raw: str) -> set[str]:
    text = raw.strip().lower()
    text = text.replace("\u00a0", " ")
    text = re.sub(r"^[*•·\s]+", "", text)
    text = text.split()[0]
    text = text.strip(".,;:，；：")
    variants = {text}
    if "(" in text and ")" in text:
        variants.add(re.sub(r"\(([^)]*)\)", r"\1", text))
        variants.add(re.sub(r"\([^)]*\)", "", text))
    expanded: set[str] = set()
    for item in variants:
        item = item.replace("/", "")
        item = re.sub(r"[^a-z'-]", "", item)
        if item:
            expanded.add(item)
    return expanded


def difficulty_score(word: str, definition: str) -> float:
    rank = COMMON_RANK.get(word, 80000)
    zipf = zipf_frequency(word, "en")
    rarity = min(rank, 80000) / 1000
    length_bonus = max(len(word) - 6, 0) * 1.8
    suffix_bonus = 8 if word.endswith(FORMAL_SUFFIXES) else 0
    definition_bonus = 10 if any(marker in definition for marker in FORMAL_MARKERS) else 0
    short_penalty = 8 if len(word) <= 5 else 0
    very_common_penalty = max(0, 5.2 - zipf) * -3
    return rarity + length_bonus + suffix_bonus + definition_bonus - short_penalty + very_common_penalty


def annotate(df: pd.DataFrame, reason: str) -> pd.DataFrame:
    out = df.copy()
    out["处理原因"] = reason
    return out


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    word_col = df.columns[1]
    scores = []
    ranks = []
    freqs = []
    for _, row in df.iterrows():
        word = str(row[word_col]).strip().lower()
        definition = clean_cell(row.iloc[3]) if len(row) > 3 else ""
        scores.append(round(difficulty_score(word, definition), 3))
        ranks.append(COMMON_RANK.get(word, ""))
        freqs.append(round(zipf_frequency(word, "en"), 3))
    out = df.copy()
    out["难度分"] = scores
    out["英文常见度排名"] = ranks
    out["zipf词频"] = freqs
    return out


def clean_line(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\u00a0", " ")).strip()


def clean_cell(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    return re.sub(r"\s+", " ", text)


if __name__ == "__main__":
    main()
