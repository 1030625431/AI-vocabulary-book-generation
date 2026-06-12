from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parents[2]
DEFAULT_TEXT = BASE / "data" / "vocab_book_text.jsonl"
DEFAULT_IMAGE_DIR = BASE / "data" / "images"
DEFAULT_OUTPUT = BASE / "build" / "full_book_portrait.html"
DEFAULT_ASSET_DIR = BASE / "build" / "portrait_assets"


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


def find_source_image(image_dir: Path, group_id: str, headword: str) -> Path | None:
    for name in (f"{group_id}_{headword}.png", f"{group_id}.png"):
        p = image_dir / name
        if p.exists():
            return p
    matches = list(image_dir.glob(f"{group_id}_*.png"))
    return matches[0] if matches else None


def prepare_print_image(source: Path, target: Path, max_px: int = 800, quality: int = 50) -> Path:
    if target.exists():
        return target
    try:
        from PIL import Image

        with Image.open(source) as image:
            image = image.convert("RGB")
            image.thumbnail((max_px, max_px))
            image.save(target, "JPEG", quality=quality, optimize=True)
        return target
    except Exception:
        fallback = target.with_suffix(source.suffix)
        if not fallback.exists():
            shutil.copy2(source, fallback)
        return fallback


CSS = """
@page {
  size: A4 portrait;
  margin: 8mm 9mm 8mm;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
  color: #18202a;
  background: #fff;
  line-height: 1.28;
  font-size: 8.6pt;
}
.cover {
  height: 280mm;
  page-break-after: always;
  display: flex;
  flex-direction: column;
  justify-content: center;
  border: 1px solid #cfd8df;
  padding: 28mm 22mm;
}
.cover h1 { margin: 0; font-size: 36pt; line-height: 1.1; }
.cover p { margin: 8mm 0 0; font-size: 13pt; color: #53606c; }
.cover .meta {
  margin-top: 22mm;
  padding-top: 8mm;
  border-top: 2px solid #1f7a8c;
  font-size: 12pt;
}
.page {
  min-height: 281mm;
  page-break-after: always;
  display: flex;
  flex-direction: column;
  gap: 1.6mm;
}
.topline {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  padding-bottom: 1mm;
  border-bottom: 1px solid #d8dee8;
  font-size: 8.4pt;
  color: #667085;
  flex-shrink: 0;
}
.topline .gid { letter-spacing: 1px; }
.hero {
  display: grid;
  grid-template-columns: 78mm 1fr;
  gap: 5mm;
  align-items: stretch;
  flex-shrink: 0;
}
.hero img {
  width: 78mm;
  height: 78mm;
  object-fit: cover;
  border: 1px solid #d8dee8;
  border-radius: 1.5mm;
}
.hero-info {
  display: flex;
  flex-direction: column;
  gap: 1.5mm;
  min-width: 0;
}
.hero-info .block {
  margin: 0;
  font-size: 8pt;
  line-height: 1.32;
}
.hero-info h3 {
  margin: 0 0 0.5mm;
  font-size: 8.4pt;
}
.hero-info .block p {
  margin: 0;
}
.title {
  font-size: 17pt;
  line-height: 1.1;
  margin: 0;
  font-weight: 700;
  word-break: break-word;
}
.core {
  margin-top: 1.6mm;
  padding: 1.6mm 2.6mm;
  background: #eef7f8;
  border-left: 2.5px solid #1f7a8c;
  font-size: 9.4pt;
  font-weight: 600;
}
.freq {
  margin-top: 1.2mm;
  font-size: 8.6pt;
  color: #1f7a8c;
}
.content {
  font-size: 8.4pt;
  display: flex;
  flex-direction: column;
  gap: 1.2mm;
  flex: 1 1 auto;
  min-height: 0;
}
h3 {
  margin: 0 0 0.6mm;
  font-size: 8.6pt;
  color: #1f7a8c;
  letter-spacing: 0.3px;
}
ul { margin: 0; padding-left: 4mm; }
li { margin: 0.2mm 0; }
.wordcard {
  border-left: 2px solid #1f7a8c;
  padding: 1mm 2.2mm;
  background: #f3f6fa;
  border-radius: 1mm;
  margin-bottom: 1mm;
  break-inside: avoid;
}
.wordhead { display: flex; flex-wrap: wrap; gap: 1.6mm; align-items: baseline; }
.word { font-size: 9.6pt; font-weight: 700; }
.phon { color: #555; font-size: 7.8pt; font-family: "Lucida Sans Unicode", "DejaVu Sans", sans-serif; }
.pos { color: #667085; font-style: italic; font-size: 7.8pt; }
.trans { font-size: 8.4pt; }
.usage { color: #53606c; font-size: 7.8pt; margin-top: 0.4mm; }
.example { margin-top: 0.5mm; font-size: 8pt; }
.example-en { font-weight: 600; }
.example-zh { color: #53606c; }
.block { break-inside: avoid; }
.block p { margin: 0; }
.mnemonic {
  padding: 1.2mm 2.2mm;
  background: #fff7df;
  border-left: 2.5px solid #f2c14e;
  border-radius: 0.8mm;
}
.mnemonic h3 { color: #8a6500; margin-top: 0; }
.mnemonic p { font-weight: 600; }
.memory {
  padding: 1.2mm 2.2mm;
  background: #eef3fa;
  border-left: 2.5px solid #5a87c4;
  border-radius: 0.8mm;
}
.memory h3 { color: #2a4d80; margin-top: 0; }
.note {
  padding: 1.2mm 2.2mm;
  background: #eff4ee;
  border-left: 2.5px solid #6c8e7a;
  border-radius: 0.8mm;
}
.note h3 { color: #2f5a44; margin-top: 0; }
.quizblock {
  padding: 4mm 0;
  page-break-after: always;
}
.quizblock h2 { margin: 0 0 3mm; font-size: 16pt; }
.quizblock .hint { margin: 0 0 4mm; color: #667085; font-size: 9.5pt; }
.quiz-list {
  columns: 2;
  column-gap: 6mm;
  column-rule: 0.4pt solid #d8dee8;
  font-size: 8.6pt;
}
.quiz-row {
  display: grid;
  grid-template-columns: 12mm 28mm 1fr;
  gap: 2mm;
  padding: 0.8mm 1mm;
  border-bottom: 0.4pt solid #eef0f4;
  break-inside: avoid;
  page-break-inside: avoid;
  line-height: 1.28;
}
.qz-id { color: #667085; }
.qz-word { font-weight: 600; }
.qz-zh { color: #18202a; }
.twocol {
  columns: 2;
  column-gap: 4mm;
}
.twocol .block { margin-bottom: 1.2mm; }
.footer {
  margin-top: auto;
  border-top: 1px solid #d8dee8;
  padding-top: 1mm;
  font-size: 7.6pt;
  color: #667085;
  text-align: right;
  flex-shrink: 0;
}
"""


def render_word_items(items: list[dict[str, Any]]) -> str:
    if not items:
        return ""
    cards = []
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
            '<div class="wordhead">',
            f'<span class="word">{word}</span>',
        ]
        if phonetic:
            parts.append(f'<span class="phon">{phonetic}</span>')
        if pos:
            parts.append(f'<span class="pos">{pos}</span>')
        if translation:
            parts.append(f'<span class="trans">{translation}</span>')
        parts.append('</div>')
        if usage:
            parts.append(f'<div class="usage">{usage}</div>')
        for e_en, e_zh in ((en, zh), (en2, zh2)):
            if e_en or e_zh:
                parts.append('<div class="example">')
                if e_en:
                    parts.append(f'<div class="example-en">{e_en}</div>')
                if e_zh:
                    parts.append(f'<div class="example-zh">{e_zh}</div>')
                parts.append('</div>')
        parts.append('</div>')
        cards.append("".join(parts))
    return "<div>" + "".join(cards) + "</div>"


def render_list_block(label: str, items: list[Any]) -> str:
    if not items:
        return ""
    cleaned = [esc(x) for x in items if str(x).strip()]
    if not cleaned:
        return ""
    body = "".join(f"<li>{x}</li>" for x in cleaned)
    return f'<section class="block"><h3>{esc(label)}</h3><ul>{body}</ul></section>'


def render_text_block(label: str, value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return f'<section class="block"><h3>{esc(label)}</h3><p>{esc(text)}</p></section>'


def render_mnemonic(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return f'<section class="block mnemonic"><h3>助记口诀</h3><p>{esc(text)}</p></section>'


def render_memory(label: str, value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return f'<section class="block memory"><h3>{esc(label)}</h3><p>{esc(text)}</p></section>'


def render_note(label: str, value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return f'<section class="block note"><h3>{esc(label)}</h3><p>{esc(text)}</p></section>'


def render_page(index: int, total: int, record: dict[str, Any], image_rel: str | None,
                book_title: str = "CET-6 视觉词汇书", exam_label: str = "六级考点") -> str:
    group_id = esc(record.get("group_id", ""))
    page_title = esc(record.get("page_title") or record.get("headword", ""))
    core = esc(record.get("core_meaning", ""))
    freq = record.get("frequent_translations") or []
    if isinstance(freq, str):
        freq = [freq]
    freq_html = " · ".join(esc(x) for x in freq if str(x).strip())

    img_html = (
        f'<img src="{esc(image_rel)}" alt="{page_title}">'
        if image_rel
        else '<div style="width:70mm;height:70mm;background:#f3f6fa;display:flex;align-items:center;justify-content:center;color:#667085;border-radius:2mm;border:1px dashed #d8dee8;">无图</div>'
    )

    word_items_html = render_word_items(record.get("word_items") or [])
    root_html = render_text_block("词根词缀 / 构词", record.get("root_affix"))
    blocks_two = []
    blocks_two.append(render_memory("画面联想", record.get("memory_link")))
    blocks_two.append(render_note("学习笔记", record.get("study_note")))
    blocks_two.append(render_list_block("常用搭配", record.get("collocations") or []))
    blocks_two.append(render_list_block("拓展 / 辨析", record.get("related_recommendations") or []))
    blocks_two.append(render_text_block(exam_label, record.get("exam_note")))
    twocol_html = "".join(b for b in blocks_two if b)
    mnem_html = render_mnemonic(record.get("mnemonic"))

    return f'''<section class="page">
  <div class="topline"><span>{esc(book_title)}</span><span class="gid">{group_id} · {index}/{total}</span></div>
  <div class="hero">
    {img_html}
    <div class="hero-info">
      <h2 class="title">{page_title}</h2>
      {f'<div class="core">{core}</div>' if core else ''}
      {f'<div class="freq">{freq_html}</div>' if freq_html else ''}
      {root_html}
    </div>
  </div>
  <div class="content">
    {word_items_html}
    <div class="twocol">{twocol_html}</div>
    {mnem_html}
  </div>
  <div class="footer">{group_id} · {esc(record.get("headword", ""))}</div>
</section>'''


def build(records: list[dict[str, Any]], image_dir: Path, asset_dir: Path,
          img_max: int = 800, img_quality: int = 50,
          book_title: str = "CET-6 视觉词汇书",
          exam_label: str = "六级考点") -> str:
    asset_dir.mkdir(parents=True, exist_ok=True)
    pages = []
    quiz_rows = []
    total = len(records)
    for i, rec in enumerate(records, start=1):
        gid = str(rec.get("group_id", "")).strip()
        word = str(rec.get("headword", "")).strip()
        if not gid:
            continue
        src = find_source_image(image_dir, gid, word)
        rel = None
        if src is not None:
            target = asset_dir / f"{gid}.jpg"
            real = prepare_print_image(src, target, img_max, img_quality)
            rel = real.relative_to(asset_dir.parent).as_posix()
        if i % 200 == 0:
            print(f"prepared {i}/{total}", flush=True)
        pages.append(render_page(i, total, rec, rel, book_title, exam_label))
        page_title = str(rec.get("page_title") or word).strip()
        core = str(rec.get("core_meaning") or "").strip()
        quiz_rows.append(
            '<div class="quiz-row">'
            f'<span class="qz-id">{esc(gid)}</span>'
            f'<span class="qz-word">{esc(page_title)}</span>'
            f'<span class="qz-zh">{esc(core)}</span>'
            '</div>'
        )

    cover = f'''<section class="cover">
  <h1>{esc(book_title)}</h1>
  <p>共 {total} 组 · 一组一页 · 配图 + 释义 + 词根词缀 + 例句 + 助记 + 画面联想</p>
  <div class="meta">Portrait Print Edition · A4</div>
</section>'''

    rows_html = "".join(quiz_rows)
    quiz_html = f'''<section class="quizblock">
  <h2>中英对照表 · 自我检验</h2>
  <p class="hint">遮住右侧释义，能脱口而出说出该组核心义即过关。共 {len(quiz_rows)} 组。</p>
  <div class="quiz-list">{rows_html}</div>
</section>'''

    return f'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{esc(book_title)} · 打印版</title>
<style>{CSS}</style>
</head>
<body>
{cover}
{quiz_html}
{chr(10).join(pages)}
</body>
</html>'''


def main() -> int:
    parser = argparse.ArgumentParser(
        description="把 vocab_book_text.jsonl + 图片目录渲染成 A4 打印版 HTML"
    )
    parser.add_argument("--text", type=Path, default=DEFAULT_TEXT)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSET_DIR)
    parser.add_argument("--img-max", type=int, default=800,
                        help="插图最长边像素 (默认 800)")
    parser.add_argument("--img-quality", type=int, default=50,
                        help="插图 JPEG 质量 1-95 (默认 50)")
    parser.add_argument("--title", default="CET-6 视觉词汇书",
                        help="书名，显示在封面和每页页眉")
    parser.add_argument("--exam-label", default="六级考点",
                        help="exam_note 字段的显示标签，如'考研考点'")
    args = parser.parse_args()

    records = load_records(args.text)
    print(f"loaded {len(records)} records", flush=True)
    html_text = build(records, args.images, args.assets,
                      args.img_max, args.img_quality,
                      args.title, args.exam_label)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_text, encoding="utf-8")
    print(f"wrote {args.output} ({len(html_text):,} bytes)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
