from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from diffusers import QwenImagePipeline


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


def read_done(manifest: Path) -> set[str]:
    done: set[str] = set()
    for item in read_jsonl(manifest):
        if item.get("event") in {"done", "skip"}:
            done.add(str(item.get("filename") or Path(str(item.get("out", ""))).name))
    return done


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="Path to local Qwen-Image weights directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where generated PNGs go")
    parser.add_argument("--manifest", type=Path, required=True, help="JSONL manifest path for resumable progress")
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--cfg", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--stop-file", type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True

    run_start = time.time()
    print(
        json.dumps(
            {
                "event": "worker_load_model_start",
                "jobs": str(args.jobs),
                "model": str(args.model),
                "cuda_visible_count": torch.cuda.device_count(),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    pipe = QwenImagePipeline.from_pretrained(str(args.model), torch_dtype=torch.bfloat16, local_files_only=True)
    print(json.dumps({"event": "worker_model_loaded", "seconds": round(time.time() - run_start, 2)}), flush=True)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=False)
    print(json.dumps({"event": "worker_model_on_cuda", "seconds": round(time.time() - run_start, 2)}), flush=True)

    with args.manifest.open("a", encoding="utf-8") as manifest:
        while True:
            jobs = read_jsonl(args.jobs)
            done = read_done(args.manifest)
            pending = [
                (index, job)
                for index, job in enumerate(jobs)
                if str(job.get("filename", "")) not in done
            ]
            if not pending:
                if args.stop_file and args.stop_file.exists():
                    print(json.dumps({"event": "worker_stop", "jobs": len(jobs), "seconds": round(time.time() - run_start, 2)}), flush=True)
                    return 0
                print(json.dumps({"event": "worker_idle", "jobs": len(jobs), "done": len(done)}), flush=True)
                time.sleep(args.poll_seconds)
                continue

            index, job = pending[0]
            filename = str(job.get("filename") or f"job_{index:05d}.png")
            out = args.output_dir / filename
            if out.exists() and out.stat().st_size > 0:
                record = {"event": "skip", "index": index, "filename": filename, "out": str(out)}
                print(json.dumps(record, ensure_ascii=False), flush=True)
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                manifest.flush()
                continue

            item_start = time.time()
            record = {
                "event": "done",
                "index": index,
                "group_id": job.get("group_id"),
                "headword": job.get("headword"),
                "filename": filename,
                "out": str(out),
            }
            try:
                generator = torch.Generator(device="cuda").manual_seed(args.seed + index)
                with torch.inference_mode():
                    image = pipe(
                        prompt=job["prompt"],
                        negative_prompt=(
                            "letters, words, readable text, alphabet, typography, caption, label, watermark, logo, "
                            "misspelled word, poster, book page, worksheet, infographic, chart, classroom board, "
                            "signboard, banner, speech bubble, comic text, user interface, blurry, low quality, cluttered layout"
                        ),
                        height=args.height,
                        width=args.width,
                        num_inference_steps=args.steps,
                        true_cfg_scale=args.cfg,
                        generator=generator,
                    ).images[0]
                image.save(out)
                record.update({"seconds": round(time.time() - item_start, 2), "bytes": out.stat().st_size})
            except Exception as exc:
                record = {
                    "event": "error",
                    "index": index,
                    "group_id": job.get("group_id"),
                    "headword": job.get("headword"),
                    "filename": filename,
                    "error": repr(exc),
                }
            print(json.dumps(record, ensure_ascii=False), flush=True)
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            manifest.flush()


if __name__ == "__main__":
    raise SystemExit(main())
