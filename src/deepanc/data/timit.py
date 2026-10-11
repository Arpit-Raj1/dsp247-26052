"""TIMIT manifest and speaker/utterance selection (Section 4.2 of the
paper): 200 training speakers (100 M / 100 F) with 7 training + 3 test
utterances each, and 20 untrained speakers (10 M / 10 F) drawn from TIMIT
TEST for Table 4's speaker-robustness condition.

TIMIT speaker directory names encode gender as the first letter (e.g.
"FCJF0" = female, "MDAB0" = male). Each speaker has 10 utterances: SA1, SA2
(dialect calibration sentences, shared by all speakers), SI*x3, SX*x5.
We hold out 3 SI/SX sentences per speaker for testing and keep the rest
(including SA1/SA2) for training, so the shared SA sentences never leak
into the held-out test utterances.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SpeakerManifest:
    speaker_id: str
    gender: str  # "M" or "F"
    train_utts: list[Path] = field(default_factory=list)
    test_utts: list[Path] = field(default_factory=list)


def _list_speakers(timit_root: Path, split: str) -> list[Path]:
    split_dir = timit_root / split
    speakers = []
    for dr in sorted(split_dir.iterdir()):
        if not dr.is_dir():
            continue
        for spk in sorted(dr.iterdir()):
            if spk.is_dir():
                speakers.append(spk)
    return speakers


def _speaker_utterances(spk_dir: Path) -> list[Path]:
    return sorted(spk_dir.glob("*.WAV")) or sorted(spk_dir.glob("*.wav"))


def build_manifest(
    timit_root: str | Path,
    n_train_speakers: int = 200,
    n_test_utts_per_speaker: int = 3,
    n_untrained_speakers: int = 20,
    seed: int = 0,
) -> tuple[list[SpeakerManifest], list[SpeakerManifest]]:
    """Returns (train_speakers, untrained_speakers).

    train_speakers: n_train_speakers from TIMIT/TRAIN (balanced M/F), each
        with train_utts (7) and test_utts (3) split out.
    untrained_speakers: n_untrained_speakers from TIMIT/TEST (balanced
        M/F), all of whose utterances are usable as "untrained speaker"
        test material (Table 4).
    """
    timit_root = Path(timit_root)
    rng = random.Random(seed)

    train_pool = _list_speakers(timit_root, "TRAIN")
    males = [p for p in train_pool if p.name.startswith("M")]
    females = [p for p in train_pool if p.name.startswith("F")]
    rng.shuffle(males)
    rng.shuffle(females)
    half = n_train_speakers // 2
    chosen = males[:half] + females[:half]
    rng.shuffle(chosen)

    train_speakers = []
    for spk_dir in chosen:
        utts = _speaker_utterances(spk_dir)
        assert len(utts) >= n_test_utts_per_speaker, spk_dir
        # keep SA1/SA2 (and everything else) available for training; only
        # SI/SX utterances are eligible for the held-out test split.
        si_sx = [u for u in utts if not u.stem.startswith("SA")]
        rng.shuffle(si_sx)
        test_utts = si_sx[:n_test_utts_per_speaker]
        train_utts = [u for u in utts if u not in test_utts]
        train_speakers.append(
            SpeakerManifest(
                speaker_id=spk_dir.name,
                gender=spk_dir.name[0],
                train_utts=train_utts,
                test_utts=test_utts,
            )
        )

    test_pool = _list_speakers(timit_root, "TEST")
    males_t = [p for p in test_pool if p.name.startswith("M")]
    females_t = [p for p in test_pool if p.name.startswith("F")]
    rng.shuffle(males_t)
    rng.shuffle(females_t)
    half_u = n_untrained_speakers // 2
    chosen_u = males_t[:half_u] + females_t[:half_u]

    untrained_speakers = [
        SpeakerManifest(
            speaker_id=spk_dir.name,
            gender=spk_dir.name[0],
            train_utts=[],
            test_utts=_speaker_utterances(spk_dir),
        )
        for spk_dir in chosen_u
    ]

    return train_speakers, untrained_speakers
