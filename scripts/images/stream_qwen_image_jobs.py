from __future__ import annotations

import argparse
import base64
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import pandas as pd

from prepare_qwen_image_jobs import build_prompt, slugify, split_words, text_value


BASE = Path(__file__).resolve().parent
DEFAULT_CLUSTERS = BASE / "output" / "deepseek_vocab_cluster_keep500_optimized" / "optimized_vocab_clusters.csv"
DEFAULT_SCENES = BASE / "output" / "qwen_visual_scenes_keep500_optimized.jsonl"
DEFAULT_SENT = BASE / "output" / "qwen_stream_sent_ids.txt"
DEFAULT_LOCAL_JOBS = BASE / "output" / "qwen_stream_jobs_sent.jsonl"
REMOTE_JOBS = "/mnt/sda/jijun/t2i/jobs/qwen_stream_jobs.jsonl"


def read_scenes(path: Path) -> dict[str, dict[str, Any]]:
    scenes: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return scenes
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            group_id = text_value(item.get("group_id"))
            if group_id:
                scenes[group_id] = item
    return scenes


def read_sent(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def build_job(row: pd.Series, scene: dict[str, Any]) -> dict[str, Any]:
    row = row.copy()
    row["visual_scene"] = text_value(scene.get("visual_scene"))
    group_id = text_value(row.get("optimized_group_id"))
    headword = text_value(row.get("headword"))
    return {
        "group_id": group_id,
        "headword": headword,
        "words": split_words(text_value(row.get("all_memory_words"))) or [headword],
        "filename": f"{group_id}_{slugify(headword)}.png",
        "suspicious_scene": bool(scene.get("suspicious")),
        "prompt": build_prompt(row),
    }


def run_remote_append(work_root: Path, jobs: list[dict[str, Any]]) -> None:
    payload = "".join(json.dumps(job, ensure_ascii=False) + "\n" for job in jobs)
    encoded = base64.b64encode(payload.encode("utf-8")).decode("ascii")
    command = f"""set -e
mkdir -p /mnt/sda/jijun/t2i/jobs
base64 -d <<'__B64__' >> {REMOTE_JOBS}
{encoded}
__B64__
wc -l {REMOTE_JOBS}
"""
    subprocess.run(
        [r"C:\Users\Nichts\anaconda3\python.exe", "tian_cmd.py", "-", "240"],
        input=command.encode("utf-8"),
        cwd=work_root,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters", type=Path, default=DEFAULT_CLUSTERS)
    parser.add_argument("--scenes", type=Path, default=DEFAULT_SCENES)
    parser.add_argument("--sent", type=Path, default=DEFAULT_SENT)
    parser.add_argument("--local-jobs", type=Path, default=DEFAULT_LOCAL_JOBS)
    parser.add_argument("--poll-seconds", type=float, default=45)
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    work_root = BASE.parent
    df = pd.read_csv(args.clusters, encoding="utf-8-sig")
    rows = {text_value(row.get("optimized_group_id")): row for _, row in df.iterrows()}
    args.sent.parent.mkdir(parents=True, exist_ok=True)
    args.local_jobs.parent.mkdir(parents=True, exist_ok=True)

    while True:
        scenes = read_scenes(args.scenes)
        sent = read_sent(args.sent)
        ready_ids = [group_id for group_id in rows if group_id in scenes and group_id not in sent]
        if ready_ids:
            batch_ids = ready_ids[: args.batch_size]
            jobs = [build_job(rows[group_id], scenes[group_id]) for group_id in batch_ids]
            run_remote_append(work_root, jobs)
            with args.local_jobs.open("a", encoding="utf-8") as f:
                for job in jobs:
                    f.write(json.dumps(job, ensure_ascii=False) + "\n")
            with args.sent.open("a", encoding="utf-8") as f:
                for group_id in batch_ids:
                    f.write(group_id + "\n")
            print(
                json.dumps(
                    {
                        "event": "sent",
                        "count": len(batch_ids),
                        "total_sent": len(sent) + len(batch_ids),
                        "scenes": len(scenes),
                        "first": batch_ids[0],
                        "last": batch_ids[-1],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        else:
            print(json.dumps({"event": "idle", "scenes": len(scenes), "sent": len(sent)}, ensure_ascii=False), flush=True)
        if args.once:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
