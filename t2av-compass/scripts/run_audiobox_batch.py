#!/usr/bin/env python3
"""
Batch AudioBox aesthetics scoring for a folder of audio files.
Outputs a single JSON file with per-file scores and summary stats.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from audiobox_aesthetics.infer import main_predict


SUPPORTED_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


def collect_audio_files(audio_dir: Path) -> List[Path]:
    files: List[Path] = []
    for ext in SUPPORTED_EXTS:
        files.extend(audio_dir.glob(f"*{ext}"))
        files.extend(audio_dir.glob(f"*{ext.upper()}"))
    return sorted(set(files))


def build_metadata(files: List[Path]) -> List[Dict]:
    return [{"path": str(p), "metadata": {"filename": p.name}} for p in files]


def summarize(results: List[Dict]) -> Dict[str, float]:
    keys = ["CE", "CU", "PC", "PQ"]
    summary: Dict[str, float] = {}
    for key in keys:
        values = [r["scores"].get(key) for r in results if key in r["scores"]]
        values = [v for v in values if isinstance(v, (int, float))]
        if values:
            summary[key] = float(sum(values) / len(values))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch AudioBox aesthetics scoring")
    parser.add_argument("--audio_dir", required=True, help="Folder with audio files")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--ckpt", default=None, help="Optional checkpoint path")
    parser.add_argument("--batch_size", type=int, default=10, help="Model batch size")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir).expanduser().resolve()
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    audio_files = collect_audio_files(audio_dir)
    if not audio_files:
        raise FileNotFoundError(f"No audio files found in: {audio_dir}")

    metadata = build_metadata(audio_files)
    outputs = main_predict(metadata, ckpt=args.ckpt, batch_size=args.batch_size)

    results: List[Dict] = []
    for file_path, output in zip(audio_files, outputs):
        try:
            scores = json.loads(output)
        except json.JSONDecodeError:
            scores = {}
        results.append(
            {
                "file": str(file_path),
                "filename": file_path.name,
                "scores": scores,
            }
        )

    output_data = {
        "metric": "audiobox_aesthetics",
        "summary": summarize(results),
        "results": results,
    }

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Saved AudioBox results to: {output_path}")


if __name__ == "__main__":
    main()
