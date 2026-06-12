"""Generate vocabulary illustration images via DashScope (Qwen-Image API).

This is the API-based alternative to scripts/images/run_qwen_stream_worker.py
(local GPU inference). It reads a jobs.jsonl produced by
`prepare_qwen_image_jobs.py`, calls the DashScope wanx text-to-image endpoint
for each job, and writes PNGs into an output directory.

Resumable: existing PNGs are skipped. Each call records progress in a manifest.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


BASE = Path(__file__).resolve().parents[2]

CREATE_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis"
QUERY_URL_TMPL = "https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"


def load_local_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
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
    return items


def submit(api_key: str, prompt: str, model: str, size: str, negative: str) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-DashScope-Async": "enable",
    }
    payload = {
        "model": model,
        "input": {
            "prompt": prompt,
            "negative_prompt": negative,
        },
        "parameters": {
            "size": size,
            "n": 1,
        },
    }
    r = requests.post(CREATE_URL, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    body = r.json()
    return body["output"]["task_id"]


def poll(api_key: str, task_id: str, *, interval: float = 4.0, timeout: float = 240.0) -> str:
    headers = {"Authorization": f"Bearer {api_key}"}
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = requests.get(QUERY_URL_TMPL.format(task_id=task_id), headers=headers, timeout=30)
        r.raise_for_status()
        body = r.json()
        status = body["output"]["task_status"]
        if status == "SUCCEEDED":
            return body["output"]["results"][0]["url"]
        if status in {"FAILED", "CANCELED"}:
            raise RuntimeError(f"task {task_id} {status}: {body!r}")
        time.sleep(interval)
    raise TimeoutError(f"task {task_id} did not finish in {timeout}s")


def download(url: str, target: Path) -> int:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    target.write_bytes(r.content)
    return len(r.content)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate vocabulary images via DashScope API")
    parser.add_argument("--jobs", type=Path, required=True, help="jobs jsonl from prepare_qwen_image_jobs.py")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--model", type=str, default="wanx-v1")
    parser.add_argument("--size", type=str, default="1024*1024")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.5)
    args = parser.parse_args()

    load_dotenv(BASE / ".env")
    load_local_env(BASE / ".env")
    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Missing DASHSCOPE_API_KEY in environment or .env")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or args.output_dir.parent / f"{args.output_dir.name}_manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    negative = (
        "letters, words, readable text, alphabet, typography, caption, label, watermark, logo, "
        "misspelled word, poster, book page, worksheet, infographic, chart, classroom board, "
        "signboard, banner, speech bubble, comic text, user interface, blurry, low quality, cluttered layout"
    )

    jobs = read_jsonl(args.jobs)
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"jobs={len(jobs)} model={args.model} out={args.output_dir}")

    with manifest_path.open("a", encoding="utf-8") as manifest:
        for index, job in enumerate(jobs):
            filename = str(job.get("filename") or f"job_{index:05d}.png")
            out = args.output_dir / filename
            if out.exists() and out.stat().st_size > 0:
                print(f"[{index + 1}/{len(jobs)}] skip {filename}")
                continue

            prompt = job.get("prompt") or ""
            if not prompt:
                print(f"[{index + 1}/{len(jobs)}] no prompt for {filename}, skip")
                continue

            t0 = time.time()
            record = {
                "event": "done",
                "index": index,
                "group_id": job.get("group_id"),
                "headword": job.get("headword"),
                "filename": filename,
            }
            try:
                task_id = submit(api_key, prompt, args.model, args.size, negative)
                url = poll(api_key, task_id)
                size_bytes = download(url, out)
                record.update({"task_id": task_id, "bytes": size_bytes, "seconds": round(time.time() - t0, 2)})
                print(f"[{index + 1}/{len(jobs)}] done {filename} ({size_bytes // 1024} KB, {record['seconds']}s)")
            except Exception as exc:
                record = {
                    "event": "error",
                    "index": index,
                    "group_id": job.get("group_id"),
                    "headword": job.get("headword"),
                    "filename": filename,
                    "error": repr(exc),
                }
                print(f"[{index + 1}/{len(jobs)}] error {filename}: {exc}")

            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            manifest.flush()
            time.sleep(args.sleep)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
