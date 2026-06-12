from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from diffusers import QwenImagePipeline


def load_jobs(path: Path) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                jobs.append(json.loads(line))
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("/mnt/sda/jijun/t2i/models/qwen-image"))
    parser.add_argument("--output-dir", type=Path, default=Path("/mnt/sda/jijun/t2i/outputs/vocab_qwen"))
    parser.add_argument("--manifest", type=Path, default=Path("/mnt/sda/jijun/t2i/outputs/vocab_qwen_manifest.jsonl"))
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--cfg", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    jobs = load_jobs(args.jobs)
    if args.limit:
        jobs = jobs[args.start_index : args.start_index + args.limit]
    else:
        jobs = jobs[args.start_index :]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True

    run_start = time.time()
    print(
        json.dumps(
            {
                "event": "load_model_start",
                "jobs": len(jobs),
                "model": str(args.model),
                "cuda_visible_count": torch.cuda.device_count(),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    pipe = QwenImagePipeline.from_pretrained(
        str(args.model),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    print(json.dumps({"event": "model_loaded", "seconds": round(time.time() - run_start, 2)}), flush=True)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=False)
    print(json.dumps({"event": "model_on_cuda", "seconds": round(time.time() - run_start, 2)}), flush=True)

    done = 0
    skipped = 0
    failed = 0
    with args.manifest.open("a", encoding="utf-8") as manifest:
        for offset, job in enumerate(jobs, start=args.start_index):
            filename = job.get("filename") or f"{job.get('group_id', offset)}.png"
            out = args.output_dir / filename
            if out.exists() and out.stat().st_size > 0 and not args.force:
                skipped += 1
                print(json.dumps({"event": "skip", "index": offset, "out": str(out)}, ensure_ascii=False), flush=True)
                continue

            item_start = time.time()
            record = {
                "event": "done",
                "index": offset,
                "group_id": job.get("group_id"),
                "headword": job.get("headword"),
                "out": str(out),
            }
            try:
                generator = torch.Generator(device="cuda").manual_seed(args.seed + offset)
                with torch.inference_mode():
                    image = pipe(
                        prompt=job["prompt"],
                        negative_prompt=(
                            "letters, words, readable text, alphabet, typography, caption, label, "
                            "watermark, logo, misspelled word, poster, book page, worksheet, infographic, "
                            "chart, classroom board, signboard, banner, speech bubble, comic text, user interface, "
                            "blurry, low quality, cluttered layout"
                        ),
                        height=args.height,
                        width=args.width,
                        num_inference_steps=args.steps,
                        true_cfg_scale=args.cfg,
                        generator=generator,
                    ).images[0]
                image.save(out)
                record.update({"seconds": round(time.time() - item_start, 2), "bytes": out.stat().st_size})
                done += 1
            except Exception as exc:  # Keep long batches resumable.
                record = {
                    "event": "error",
                    "index": offset,
                    "group_id": job.get("group_id"),
                    "headword": job.get("headword"),
                    "error": repr(exc),
                }
                failed += 1
            print(json.dumps(record, ensure_ascii=False), flush=True)
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            manifest.flush()

    print(
        json.dumps(
            {
                "event": "batch_finished",
                "done": done,
                "skipped": skipped,
                "failed": failed,
                "seconds": round(time.time() - run_start, 2),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
