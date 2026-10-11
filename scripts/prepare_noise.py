#!/usr/bin/env python
"""Build the training/validation noise shards (data/noise_train.h5,
data/noise_val.h5): ~10,000 (+500 validation) non-speech 6-second clips,
16 kHz, substituting for the paper's commercial sound-ideas.com library
(see CLAUDE.md / REPRODUCTION.md "Known paper ambiguities").

Three sources are supported via --source:

  dns    (cluster-scale, matches the plan's primary choice): the DNS-
         Challenge noise set. Too large to fetch from a login node without
         internet-enabled compute, and too large for this repo's local dev
         disk budget. This script prints the official fetch script
         instructions and expects the *directory* of already-downloaded
         DNS noise wavs to be passed via --raw-dir; it does not itself
         invoke the ~100GB+ Azure blob download.
             https://github.com/microsoft/DNS-Challenge
             (download-dns-challenge-4.sh, "noise_fullband" set)

  musan  (cluster-scale fallback): the MUSAN noise subset
         (openslr.org/resources/17). ~11GB combined archive (speech+music+
         noise) — download+extract it yourself (see CLUSTER.md), then pass
         --raw-dir at the extracted musan/noise tree. That tree has only
         ~930 files, too few for 10k+500 *non-overlapping* 6s segments; see
         --hop-seconds below (this script auto-densifies the segmentation
         hop by default, so plain --source musan --raw-dir ... usually
         reaches the target count without any extra flags).

  esc50  (small, free, fully automatic — used for local development and
         the smoke test in this repo, since dns/musan are impractical to
         fetch inside this sandbox's disk/network budget): the ESC-50
         environmental sound dataset (2000 5-second clips, ~600MB zip,
         github.com/karolpiczak/ESC-50). Clips from speech-adjacent
         categories (crying_baby, sneezing, coughing, laughing,
         breathing, footsteps applause etc. involving the human voice)
         are excluded so the shard stays "non-speech" like the paper's
         training noise.

All sources funnel through the same pipeline: collect source clips ->
resample to 16kHz -> concatenate/segment into 6-second chunks -> shuffle ->
split into train/val -> write via deepanc.data.noise.write_noise_shard.

Usage (local dev):
  python scripts/prepare_noise.py --source esc50 --n-train 200 --n-val 20

Usage (cluster, after downloading DNS noise separately):
  python scripts/prepare_noise.py --source dns --raw-dir /path/to/dns/noise \
      --n-train 10000 --n-val 500
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from tqdm import tqdm

from deepanc.audio.io import load_resampled
from deepanc.data.noise import write_noise_shard

SEG_SECONDS = 6
SR = 16000
SEG_LEN = SEG_SECONDS * SR

ESC50_ZIP_URL = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
# ESC-50 categories that are speech-adjacent / vocal; excluded to keep the
# shard "non-speech", analogous to how DNS noise excludes speech classes.
ESC50_SPEECH_ADJACENT = {
    "crying_baby",
    "sneezing",
    "coughing",
    "laughing",
    "breathing",
    "snoring",
    "clapping",
}


def _segment(sig: np.ndarray, seg_len: int = SEG_LEN, hop: int | None = None) -> list[np.ndarray]:
    """Chop a waveform into seg_len-sample segments with the given hop
    (defaults to seg_len, i.e. non-overlapping), dropping any final
    partial segment shorter than half a segment and zero-padding a final
    partial segment that's at least half-length.

    A smaller hop (overlapping windows) multiplies the segment yield from
    the same source files roughly by seg_len/hop — needed for sources like
    MUSAN's noise/ subset, which has few underlying files (~930) that are
    individually longer than 6s but not numerous enough to reach 10k+
    *non-overlapping* segments (see prepare_noise.py's --hop-seconds).
    """
    if hop is None:
        hop = seg_len
    segs = []
    n = len(sig)
    for start in range(0, max(n, 1), hop):
        chunk = sig[start : start + seg_len]
        if len(chunk) == seg_len:
            segs.append(chunk)
        elif len(chunk) >= seg_len // 2:
            segs.append(np.pad(chunk, (0, seg_len - len(chunk))))
        if start + seg_len >= n:
            break
    return segs


def collect_esc50(cache_dir: Path) -> list[Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    root = cache_dir / "ESC-50-master"
    if not root.exists():
        print(f"downloading {ESC50_ZIP_URL} ...")
        with urllib.request.urlopen(ESC50_ZIP_URL) as resp:
            data = resp.read()
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            zf.extractall(cache_dir)
    meta_path = root / "meta" / "esc50.csv"
    audio_dir = root / "audio"
    import csv

    wavs = []
    with open(meta_path) as f:
        for row in csv.DictReader(f):
            if row["category"] in ESC50_SPEECH_ADJACENT:
                continue
            wavs.append(audio_dir / row["filename"])
    return wavs


def collect_raw_dir(raw_dir: Path) -> list[Path]:
    exts = {".wav", ".flac"}
    return [p for p in sorted(raw_dir.rglob("*")) if p.suffix.lower() in exts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["dns", "musan", "esc50"], default="esc50")
    ap.add_argument("--raw-dir", type=str, default=None, help="pre-downloaded wav directory (dns/musan)")
    ap.add_argument("--cache-dir", default="data/_cache/noise_src")
    ap.add_argument("--out-train", default="data/noise_train.h5")
    ap.add_argument("--out-val", default="data/noise_val.h5")
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument(
        "--hop-seconds", type=float, default=None,
        help="segmentation hop in seconds; default is non-overlapping (hop=6s). "
        "If the source has too few underlying files to reach --n-train + --n-val "
        "non-overlapping segments (e.g. MUSAN's noise/ subset, ~930 files), this "
        "script automatically retries with a smaller hop (overlapping windows) "
        "before giving up — pass this explicitly to skip straight to a known-good value.",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.source == "esc50":
        files = collect_esc50(Path(args.cache_dir))
    else:
        if not args.raw_dir:
            raise SystemExit(
                f"--source {args.source} requires --raw-dir pointing at an already-downloaded "
                "wav directory (see this script's module docstring for fetch instructions)."
            )
        files = collect_raw_dir(Path(args.raw_dir))

    print(f"found {len(files)} source files")

    rng = np.random.default_rng(args.seed)
    rng.shuffle(files)  # type: ignore[arg-type]

    n_needed = args.n_train + args.n_val

    # Cap how many (already-shuffled) source files we decode — for a huge
    # source tree (full DNS-Challenge/MUSAN with all categories) there's no
    # need to load everything just to reach n_needed segments; 4x is a
    # generous margin even at non-overlapping segmentation for typical
    # multi-second clips, and the auto-densify hop search below covers the
    # rest.
    files = files[: max(n_needed * 4, 2000)]

    # Cache decoded+resampled audio once; re-segmenting at a denser hop
    # (below) shouldn't mean re-decoding every file from scratch.
    print(f"loading up to {len(files)} source audio files...")
    signals = []
    for path in tqdm(files, desc="loading"):
        try:
            signals.append(load_resampled(path, SR))
        except Exception as e:  # pragma: no cover - defensive against odd files
            print(f"skip {path}: {e}")

    def segment_all(hop_seconds: float | None) -> list[np.ndarray]:
        hop = int(hop_seconds * SR) if hop_seconds is not None else None
        segs: list[np.ndarray] = []
        for sig in signals:
            segs.extend(_segment(sig, hop=hop))
        return segs

    # Try non-overlapping first (or the user's explicit --hop-seconds), then
    # auto-densify if that's not enough source material — e.g. MUSAN's
    # noise/ subset has only ~930 files, too few for 10k+ *non-overlapping*
    # 6s segments even though the files themselves have enough total
    # duration once windows are allowed to overlap.
    hop_candidates = [args.hop_seconds] if args.hop_seconds is not None else [None, 3.0, 1.5, 1.0]
    segments: list[np.ndarray] = []
    for hop_seconds in hop_candidates:
        segments = segment_all(hop_seconds)
        label = "non-overlapping" if hop_seconds is None else f"{hop_seconds}s hop"
        print(f"  {label}: {len(segments)} segments")
        if len(segments) >= n_needed:
            break

    if len(segments) < n_needed:
        print(
            f"warning: only produced {len(segments)} segments, wanted {n_needed}, "
            f"even at the densest hop tried ({hop_candidates[-1]}s); shard will be "
            "smaller than requested. Add more source files (e.g. MUSAN's music/ "
            "subset, or a lower --hop-seconds floor) to close the gap."
        )

    idx = np.arange(len(segments))
    rng.shuffle(idx)
    segments = [segments[i] for i in idx]

    val_clips = segments[: args.n_val]
    train_clips = segments[args.n_val : args.n_val + args.n_train]

    write_noise_shard(args.out_train, train_clips, sample_rate=SR)
    write_noise_shard(args.out_val, val_clips, sample_rate=SR)
    print(f"wrote {args.out_train} ({len(train_clips)} clips), {args.out_val} ({len(val_clips)} clips)")


if __name__ == "__main__":
    main()
