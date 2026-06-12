from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parents[2]
DEFAULT_TEXT = BASE / "data" / "vocab_book_text.jsonl"
DEFAULT_IMAGE_DIR = BASE / "data" / "images"
DEFAULT_OUTPUT = BASE / "build" / "full_book.html"


def esc(s: Any) -> str:
    if s is None:
        return ""
    return html.escape(str(s))


def load_records(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    items.sort(key=lambda x: str(x.get("group_id", "")))
    return items


def find_image(image_dir: Path, group_id: str, headword: str) -> str | None:
    candidates = [
        image_dir / f"{group_id}_{headword}.png",
        image_dir / f"{group_id}.png",
    ]
    for cand in candidates:
        if cand.exists():
            return cand.name
    matches = list(image_dir.glob(f"{group_id}_*.png"))
    if matches:
        return matches[0].name
    return None


def render_word_items(items: list[dict[str, Any]]) -> str:
    if not items:
        return ""
    rows = []
    for it in items:
        word = esc(it.get("word", ""))
        phonetic = esc(it.get("phonetic", ""))
        pos = esc(it.get("pos", ""))
        translation = esc(it.get("translation", ""))
        usage = esc(it.get("usage_note", ""))
        en = esc(it.get("example_en", ""))
        zh = esc(it.get("example_zh", ""))
        en2 = esc(it.get("example_en_2", ""))
        zh2 = esc(it.get("example_zh_2", ""))
        parts = [
            '<div class="wordcard">',
            '  <div class="wordhead">',
            f'    <span class="word">{word}</span>',
        ]
        if phonetic:
            parts.append(f'    <span class="phon">{phonetic}</span>')
        if pos:
            parts.append(f'    <span class="pos">{pos}</span>')
        if translation:
            parts.append(f'    <span class="trans">{translation}</span>')
        parts.append('  </div>')
        if usage:
            parts.append(f'  <div class="usage">{usage}</div>')
        for e_en, e_zh in ((en, zh), (en2, zh2)):
            if e_en or e_zh:
                parts.append('  <div class="example">')
                if e_en:
                    parts.append(f'    <div class="example-en">{e_en}</div>')
                if e_zh:
                    parts.append(f'    <div class="example-zh">{e_zh}</div>')
                parts.append('  </div>')
        parts.append('</div>')
        rows.append("\n".join(parts))
    return '<div class="wordlist">\n' + "\n".join(rows) + "\n</div>"


def render_list(label: str, items: list[Any]) -> str:
    if not items:
        return ""
    cleaned = [esc(x) for x in items if str(x).strip()]
    if not cleaned:
        return ""
    body = "".join(f"<li>{x}</li>" for x in cleaned)
    return f'<section class="block"><h3>{esc(label)}</h3><ul>{body}</ul></section>'


def render_text(label: str, value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return f'<section class="block"><h3>{esc(label)}</h3><p>{esc(text)}</p></section>'


def render_mnemonic(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return f'<section class="mnemonic"><h3>助记口诀</h3><p>{esc(text)}</p></section>'


def render_memory(label: str, value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return f'<section class="memory"><h3>{esc(label)}</h3><p>{esc(text)}</p></section>'


def render_note(label: str, value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return f'<section class="note"><h3>{esc(label)}</h3><p>{esc(text)}</p></section>'


def render_article(record: dict[str, Any], image_rel: str | None,
                   exam_label: str = "六级考点") -> str:
    group_id = esc(record.get("group_id", ""))
    page_title = esc(record.get("page_title") or record.get("headword", ""))
    core = esc(record.get("core_meaning", ""))
    freq = record.get("frequent_translations") or []
    if isinstance(freq, str):
        freq = [freq]
    freq_html = (
        '<div class="freq">' + " · ".join(esc(x) for x in freq if str(x).strip()) + "</div>"
        if freq
        else ""
    )

    img_html = (
        f'<div class="visual"><img src="../data/images/{esc(image_rel)}" alt="{page_title}" loading="lazy"></div>'
        if image_rel
        else '<div class="visual missing">无图</div>'
    )

    blocks = []
    blocks.append(render_word_items(record.get("word_items") or []))
    blocks.append(render_memory("画面联想", record.get("memory_link")))
    blocks.append(render_mnemonic(record.get("mnemonic")))
    blocks.append(render_note("学习笔记", record.get("study_note")))
    blocks.append(render_list("常用搭配", record.get("collocations") or []))
    blocks.append(render_list("拓展 / 辨析", record.get("related_recommendations") or []))
    blocks.append(render_text(exam_label, record.get("exam_note")))
    blocks_html = "\n".join(b for b in blocks if b)

    root_html = render_text("词根词缀 / 构词", record.get("root_affix"))

    return f'''<article id="{group_id}">
  <div class="left">
    {img_html}
    <div class="meta">
      <div class="gid">{group_id}</div>
      <h2>{page_title}</h2>
      {f'<div class="core">{core}</div>' if core else ''}
      {freq_html}
      {root_html}
    </div>
  </div>
  <div class="right">
    {blocks_html}
  </div>
</article>'''


CSS = """
:root {
  --ink: #1f2933;
  --muted: #667085;
  --line: #d8dee8;
  --paper: #fbfaf7;
  --accent: #1f7a8c;
  --warm: #f2c14e;
  --soft: #f3f6fa;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: "Segoe UI", "Microsoft YaHei", "PingFang SC", Arial, sans-serif;
  color: var(--ink);
  background: var(--paper);
  line-height: 1.6;
}
header {
  padding: 42px min(6vw, 72px) 28px;
  border-bottom: 1px solid var(--line);
  background: #fff;
}
h1 {
  margin: 0 0 8px;
  font-size: clamp(28px, 4vw, 48px);
}
header p { margin: 0; color: var(--muted); }
main {
  display: grid;
  gap: 24px;
  padding: 28px min(5vw, 64px) 64px;
}
article {
  display: grid;
  grid-template-columns: minmax(280px, 380px) minmax(0, 1fr);
  gap: 28px;
  padding: 22px;
  background: #fff;
  border: 1px solid var(--line);
  border-radius: 10px;
  break-inside: avoid;
  page-break-inside: avoid;
}
.left { display: flex; flex-direction: column; gap: 14px; }
.visual img {
  width: 100%;
  aspect-ratio: 1 / 1;
  object-fit: cover;
  border-radius: 8px;
  border: 1px solid var(--line);
  background: var(--soft);
}
.visual.missing {
  width: 100%;
  aspect-ratio: 1 / 1;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--muted);
  background: var(--soft);
  border-radius: 8px;
  border: 1px dashed var(--line);
}
.gid { color: var(--muted); font-size: 13px; letter-spacing: 1px; }
h2 { margin: 0; font-size: 26px; line-height: 1.2; word-break: break-word; }
.core { color: var(--ink); margin-top: 6px; font-weight: 600; }
.freq { color: var(--accent); margin-top: 4px; font-size: 14px; }
.right { display: flex; flex-direction: column; gap: 14px; }
.block h3 {
  margin: 0 0 6px;
  font-size: 14px;
  color: var(--accent);
  letter-spacing: 0.5px;
}
.block p { margin: 0; }
.block ul { margin: 0; padding-left: 22px; }
.block li { margin: 3px 0; }
.wordlist { display: flex; flex-direction: column; gap: 10px; }
.wordcard {
  border-left: 3px solid var(--accent);
  padding: 8px 12px;
  background: var(--soft);
  border-radius: 6px;
}
.wordhead { display: flex; flex-wrap: wrap; gap: 8px; align-items: baseline; }
.word { font-size: 18px; font-weight: 700; }
.phon { color: #555; font-size: 13px; font-family: "Lucida Sans Unicode", "DejaVu Sans", sans-serif; }
.pos { color: var(--muted); font-style: italic; font-size: 13px; }
.trans { font-size: 14px; }
.usage { color: var(--muted); font-size: 13px; margin-top: 4px; }
.example { margin-top: 6px; font-size: 14px; }
.example-en { font-weight: 600; }
.example-zh { color: var(--muted); }
.mnemonic {
  padding: 10px 14px;
  border-left: 4px solid var(--warm);
  background: #fff8e1;
  border-radius: 6px;
}
.mnemonic h3 { color: #8a6500; margin-top: 0; }
.mnemonic p { margin: 0; font-weight: 600; }
.memory {
  padding: 10px 14px;
  border-left: 4px solid #5a87c4;
  background: #eef3fa;
  border-radius: 6px;
}
.memory h3 { color: #2a4d80; margin-top: 0; }
.memory p { margin: 0; }
.note {
  padding: 10px 14px;
  border-left: 4px solid #6c8e7a;
  background: #eff4ee;
  border-radius: 6px;
}
.note h3 { color: #2f5a44; margin-top: 0; }
.note p { margin: 0; }
.quiz {
  padding: 24px min(5vw, 64px);
  background: #fff;
  border-bottom: 1px solid var(--line);
}
.quiz h2 { margin: 0 0 6px; font-size: 20px; }
.quiz .hint { margin: 0 0 14px; color: var(--muted); font-size: 13px; }
.quiz-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
.quiz-table th, .quiz-table td {
  border-bottom: 1px solid var(--line);
  padding: 6px 10px;
  text-align: left;
  vertical-align: top;
}
.quiz-table th {
  background: var(--soft);
  color: var(--accent);
  font-weight: 600;
}
.quiz-table a { color: var(--ink); text-decoration: none; font-weight: 600; }
.quiz-table a:hover { text-decoration: underline; }
.quiz-table td:nth-child(1) { width: 70px; color: var(--muted); }
.quiz-table td:nth-child(2) { width: 30%; }
.toc {
  padding: 28px min(5vw, 64px);
  background: #fff;
  border-bottom: 1px solid var(--line);
}
.toc h2 { font-size: 18px; margin: 0 0 10px; }
.toc-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
  gap: 4px 14px;
  font-size: 13px;
}
.toc a { color: var(--ink); text-decoration: none; }
.toc a:hover { text-decoration: underline; }
@media (max-width: 760px) {
  article { grid-template-columns: 1fr; padding: 16px; }
  h2 { font-size: 22px; }
}
@media print {
  body { background: #fff; }
  header, .toc, .quiz { display: none; }
  main { gap: 0; padding: 0; }
  article {
    grid-template-columns: 1fr;
    border: none;
    border-radius: 0;
    padding: 12mm 14mm;
    page-break-after: always;
  }
}
"""


def build(records: list[dict[str, Any]], image_dir: Path,
          book_title: str = "CET-6 视觉词汇书",
          exam_label: str = "六级考点") -> str:
    articles = []
    toc_entries = []
    quiz_entries = []
    for rec in records:
        group_id = str(rec.get("group_id", "")).strip()
        headword = str(rec.get("headword", "")).strip()
        if not group_id:
            continue
        image_rel = find_image(image_dir, group_id, headword)
        articles.append(render_article(rec, image_rel, exam_label))
        toc_entries.append(f'<a href="#{esc(group_id)}">{esc(group_id)} {esc(headword)}</a>')
        page_title = str(rec.get("page_title") or headword).strip()
        core = str(rec.get("core_meaning") or "").strip()
        quiz_entries.append(
            f'<tr><td><a href="#{esc(group_id)}">{esc(group_id)}</a></td>'
            f'<td>{esc(page_title)}</td><td>{esc(core)}</td></tr>'
        )

    toc_html = (
        '<section class="toc"><h2>目录索引（共 {n} 组）</h2><div class="toc-grid">{links}</div></section>'.format(
            n=len(toc_entries), links="".join(toc_entries)
        )
        if toc_entries
        else ""
    )

    quiz_html = ""
    if quiz_entries:
        rows = "".join(quiz_entries)
        quiz_html = (
            '<section class="quiz">'
            '<h2>中英对照表（自我检验）</h2>'
            '<p class="hint">遮住右侧中文，能脱口而出说出该组核心义即过关。点击词组跳到详细页。</p>'
            '<table class="quiz-table">'
            '<thead><tr><th>编号</th><th>词组</th><th>核心释义</th></tr></thead>'
            f'<tbody>{rows}</tbody>'
            '</table>'
            '</section>'
        )

    return f'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(book_title)}</title>
<style>{CSS}</style>
</head>
<body>
<header>
  <h1>{esc(book_title)}</h1>
  <p>共 {len(articles)} 组 · 一组一页 · 配图 + 释义 + 词根词缀 + 例句 + 助记 + 画面联想</p>
</header>
{quiz_html}
{toc_html}
<main>
{chr(10).join(articles)}
</main>
</body>
</html>'''


def main() -> int:
    parser = argparse.ArgumentParser(
        description="把 vocab_book_text.jsonl + 图片目录渲染成网页版 HTML"
    )
    parser.add_argument("--text", type=Path, default=DEFAULT_TEXT)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--title", default="CET-6 视觉词汇书",
                        help="书名，显示在页面标题和页眉")
    parser.add_argument("--exam-label", default="六级考点",
                        help="exam_note 字段的显示标签，如'考研考点'")
    args = parser.parse_args()

    records = load_records(args.text)
    print(f"loaded {len(records)} records from {args.text}")
    html_text = build(records, args.images, args.title, args.exam_label)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_text, encoding="utf-8")
    missing = sum(1 for r in records if not find_image(args.images, str(r.get("group_id", "")), str(r.get("headword", ""))))
    print(f"wrote {args.output} ({len(html_text):,} bytes), missing images: {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
