from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd


INPUT_FILE = Path("wod.xls")
OUTPUT_DIR = Path("output")


NEGATIVE_PREFIXES = ("un", "in", "im", "ir", "il", "non", "dis")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = pd.read_excel(INPUT_FILE, engine="xlrd")
    entries = read_entries(df)
    words = sorted(entries)
    wordset = set(words)

    dsu = DSU(words)
    reasons: dict[frozenset[str], set[str]] = defaultdict(set)

    for word in words:
        for base, reason in candidate_bases(word):
            if base in wordset:
                dsu.union(word, base)
                reasons[frozenset((word, base))].add(reason)

    # Keep antonym pairs together only when the remaining base is a full word.
    for word in words:
        for prefix in NEGATIVE_PREFIXES:
            if word.startswith(prefix) and len(word) >= len(prefix) + 5:
                base = word[len(prefix) :]
                if base in wordset:
                    dsu.union(word, base)
                    reasons[frozenset((word, base))].add(f"negative prefix {prefix}-")

    groups = defaultdict(list)
    for word in words:
        groups[dsu.find(word)].append(word)

    rows = []
    for page_no, group_words in enumerate(sorted(groups.values(), key=group_sort_key), start=1):
        group_words = sorted(group_words, key=lambda w: (len(w), w))
        headword = choose_headword(group_words)
        group_reason = infer_group_reason(group_words, reasons)
        rows.append(
            {
                "group_id": f"G{page_no:04d}",
                "page_no": page_no,
                "headword": headword,
                "count": len(group_words),
                "words": ", ".join(group_words),
                "reason": group_reason,
                "definitions": " | ".join(
                    f"{w}: {entries[w]['definition']}" for w in group_words if entries[w]["definition"]
                ),
            }
        )

    out_df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "word_groups.csv"
    xlsx_path = OUTPUT_DIR / "word_groups.xlsx"
    out_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    out_df.to_excel(xlsx_path, index=False)

    multi = out_df[out_df["count"] > 1]
    print(f"words: {len(words)}")
    print(f"groups/pages: {len(out_df)}")
    print(f"multi-word groups: {len(multi)}")
    print(f"single-word groups: {len(out_df) - len(multi)}")
    print(f"csv: {csv_path.resolve()}")
    print(f"xlsx: {xlsx_path.resolve()}")


def read_entries(df: pd.DataFrame) -> dict[str, dict[str, str]]:
    entries: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        word = str(row.iloc[1]).strip().lower()
        if not re.fullmatch(r"[a-z][a-z'-]*", word):
            continue
        current = entries.setdefault(
            word,
            {
                "phonetic": clean_cell(row.iloc[2]) if len(row) > 2 else "",
                "definition": clean_cell(row.iloc[3]) if len(row) > 3 else "",
            },
        )
        if not current["definition"] and len(row) > 3:
            current["definition"] = clean_cell(row.iloc[3])
    return entries


def clean_cell(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    return re.sub(r"\s+", " ", text)


def candidate_bases(word: str) -> Iterable[tuple[str, str]]:
    # Inflections.
    if word.endswith("ies") and len(word) > 4:
        yield word[:-3] + "y", "plural/verb inflection"
    if word.endswith("ied") and len(word) > 4:
        yield word[:-3] + "y", "past-tense inflection"
    if word.endswith("ing") and len(word) > 5:
        stem = word[:-3]
        yield stem, "-ing inflection"
        yield stem + "e", "-ing inflection"
        if len(stem) > 3 and stem[-1] == stem[-2]:
            yield stem[:-1], "-ing doubled consonant"
    if word.endswith("ed") and len(word) > 4:
        stem = word[:-2]
        yield stem, "-ed inflection"
        yield stem + "e", "-ed inflection"
        if len(stem) > 3 and stem[-1] == stem[-2]:
            yield stem[:-1], "-ed doubled consonant"
    if word.endswith("es") and len(word) > 4:
        yield word[:-2], "plural/verb inflection"
    if word.endswith("s") and len(word) > 3:
        yield word[:-1], "plural/verb inflection"

    # Adverb/noun/adjective derivations. These are intentionally conservative:
    # a candidate is useful only when the full base word exists in the list.
    if word.endswith("ly") and len(word) > 4:
        yield word[:-2], "adverb from adjective"
        if word.endswith("ily"):
            yield word[:-3] + "y", "adverb from -y adjective"
        if word.endswith("ally"):
            yield word[:-4], "adverb from -al adjective"
            yield word[:-2], "adverb from adjective"
    if word.endswith("ness") and len(word) > 6:
        yield word[:-4], "noun from adjective"
        if word.endswith("iness"):
            yield word[:-5] + "y", "noun from -y adjective"
    if word.endswith("ment") and len(word) > 6:
        yield word[:-4], "noun from verb"
        yield word[:-4] + "e", "noun from verb"
    if word.endswith("less") and len(word) > 6:
        yield word[:-4], "adjective from noun"
    if word.endswith("ful") and len(word) > 6:
        yield word[:-3], "adjective from noun"

    # Agent nouns. Require longer bases to avoid false pairs like form/former.
    if word.endswith("er") and len(word) > 6:
        stem = word[:-2]
        yield stem, "agent/comparative form"
        yield stem + "e", "agent/comparative form"
        if len(stem) > 4 and stem[-1] == stem[-2]:
            yield stem[:-1], "agent/comparative doubled consonant"
    if word.endswith("or") and len(word) > 6:
        stem = word[:-2]
        yield stem, "agent noun"
        yield stem + "e", "agent noun"
        yield stem + "ate", "agent noun"

    # Common derivative suffixes.
    if word.endswith("able") and len(word) > 7:
        stem = word[:-4]
        yield stem, "adjective from verb"
        yield stem + "e", "adjective from verb"
    if word.endswith("ible") and len(word) > 7:
        stem = word[:-4]
        yield stem, "adjective from verb"
        yield stem + "e", "adjective from verb"
    if word.endswith("ability") and len(word) > 10:
        yield word[:-7] + "able", "noun from -able adjective"
    if word.endswith("ibility") and len(word) > 10:
        yield word[:-7] + "ible", "noun from -ible adjective"
    if word.endswith("al") and len(word) > 6:
        yield word[:-2], "adjective/noun derivative"
    if word.endswith("ial") and len(word) > 6:
        yield word[:-3] + "y", "adjective derivative"
    if word.endswith("ic") and len(word) > 6:
        yield word[:-2] + "y", "adjective derivative"
    if word.endswith("ical") and len(word) > 8:
        yield word[:-2], "adjective derivative"
        yield word[:-4] + "y", "adjective derivative"
    if word.endswith("ity") and len(word) > 7:
        stem = word[:-3]
        yield stem, "noun from adjective"
        yield stem + "e", "noun from adjective"
        yield stem + "al", "noun from adjective"
        yield stem + "ive", "noun from adjective"
    if word.endswith("cy") and len(word) > 6:
        stem = word[:-2]
        yield stem + "t", "noun from adjective"
        yield stem + "te", "noun from adjective"
        yield stem + "nt", "noun from adjective"

    # Verb/noun families such as operate/operation. These are guarded by
    # complete existing bases, so they do not create broad root groups.
    if word.endswith("ation") and len(word) > 8:
        stem = word[:-5]
        yield stem, "noun from verb"
        yield stem + "e", "noun from verb"
        yield stem + "ate", "noun from verb"
    if word.endswith("tion") and len(word) > 7:
        stem = word[:-3]
        yield stem, "noun from verb"
        yield stem + "e", "noun from verb"
        yield word[:-4], "noun from verb"
    if word.endswith("sion") and len(word) > 7:
        stem = word[:-4]
        yield stem, "noun from verb"
        yield stem + "e", "noun from verb"
        yield stem + "d", "noun from verb"
        yield stem + "de", "noun from verb"
    if word.endswith("ance") and len(word) > 7:
        stem = word[:-4]
        yield stem, "noun from verb/adjective"
        yield stem + "e", "noun from verb/adjective"
        yield stem + "ant", "noun/adjective family"
    if word.endswith("ence") and len(word) > 7:
        stem = word[:-4]
        yield stem, "noun from verb/adjective"
        yield stem + "e", "noun from verb/adjective"
        yield stem + "ent", "noun/adjective family"
    if word.endswith("ancy") and len(word) > 7:
        yield word[:-4] + "ant", "noun/adjective family"
    if word.endswith("ency") and len(word) > 7:
        yield word[:-4] + "ent", "noun/adjective family"


def infer_group_reason(group_words: list[str], reasons: dict[frozenset[str], set[str]]) -> str:
    found: set[str] = set()
    group_set = set(group_words)
    for pair, pair_reasons in reasons.items():
        if pair <= group_set:
            found.update(pair_reasons)
    if not found:
        return "single word"
    return "; ".join(sorted(found))


def choose_headword(words: list[str]) -> str:
    # Prefer a short base-like word, but keep stable alphabetical behavior.
    return sorted(words, key=lambda w: (len(w), w))[0]


def group_sort_key(words: list[str]) -> tuple[str, int, str]:
    headword = choose_headword(words)
    return headword, len(words), ",".join(words)


class DSU:
    def __init__(self, items: Iterable[str]):
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
    main()
