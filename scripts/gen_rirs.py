#!/usr/bin/env python
"""Generate all RIR sets used by the reproduction and save them to
data/rirs.npz (Section 4.2 room/geometry, Section 5.3 quiet zone).

Sets written:
  train       - 5 (primary, secondary) pairs, one per training T60
  test        - 1 pair at T60=0.2s (default test condition, Tables 1/2/4/5)
  test_untrained_t60 - 6 pairs at the untrained T60s used for Fig. 8
  quiet_zone_train   - 100 pairs, error mic sampled in the 5cm sphere
  quiet_zone_test_r0 - 1 pair, error mic at the sphere center (r=0)
  quiet_zone_test_r2 - 1 pair, error mic 2cm from center (random direction)
  quiet_zone_test_r5 - 10 pairs, error mic at 0<=r<=5cm (random points)

Usage:
  python scripts/gen_rirs.py --out data/rirs.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deepanc.acoustics import rir as rirmod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/rirs.npz")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("generating training RIRs (5 T60s)...")
    train_set = rirmod.build_training_set()

    print("generating default test RIRs (T60=0.2s)...")
    test_set = rirmod.build_test_set([rirmod.TEST_T60], tag="test")

    print("generating untrained-T60 test RIRs (Fig. 8)...")
    untrained_t60_set = rirmod.build_test_set(rirmod.UNTRAINED_T60S, tag="test_untrained_t60")

    print("generating quiet-zone training RIRs (100 points in 5cm sphere)...")
    qz_train = rirmod.build_quiet_zone_set(
        n_points=rirmod.QUIET_ZONE_N_POINTS, radius=rirmod.QUIET_ZONE_RADIUS, seed=args.seed
    )

    print("generating quiet-zone test RIRs (r=0, r=2cm, 0<=r<=5cm)...")
    import numpy as np

    p0, s0 = rirmod.generate_pair(rirmod.TEST_T60, rirmod.ERROR_MIC)

    qz_test_r0 = rirmod.RirSet(
        primary=p0[None, :], secondary=s0[None, :], t60=np.array([rirmod.TEST_T60], dtype="float32"),
        error_mic_pos=np.array([rirmod.ERROR_MIC], dtype="float32"), tag="quiet_zone_test_r0",
    )

    r2_points = rirmod.sample_sphere_points(1, radius=0.02, center=rirmod.ERROR_MIC, seed=args.seed + 1)
    # force exact radius 2cm (sample_sphere_points samples *within* radius)
    center = np.array(rirmod.ERROR_MIC, dtype="float64")
    direction = (r2_points[0] - center)
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    r2_point = (center + direction * 0.02).astype("float32")
    p2, s2 = rirmod.generate_pair(rirmod.TEST_T60, r2_point)
    qz_test_r2 = rirmod.RirSet(
        primary=p2[None, :], secondary=s2[None, :], t60=np.array([rirmod.TEST_T60], dtype="float32"),
        error_mic_pos=r2_point[None, :], tag="quiet_zone_test_r2",
    )

    r5_points = rirmod.sample_sphere_points(10, radius=rirmod.QUIET_ZONE_RADIUS, center=rirmod.ERROR_MIC, seed=args.seed + 2)
    primaries, secondaries = [], []
    for pt in r5_points:
        p, s = rirmod.generate_pair(rirmod.TEST_T60, pt)
        primaries.append(p)
        secondaries.append(s)
    qz_test_r5 = rirmod.RirSet(
        primary=np.stack(primaries), secondary=np.stack(secondaries),
        t60=np.full(10, rirmod.TEST_T60, dtype="float32"), error_mic_pos=r5_points, tag="quiet_zone_test_r5",
    )

    sets = {
        "train": train_set,
        "test": test_set,
        "test_untrained_t60": untrained_t60_set,
        "quiet_zone_train": qz_train,
        "quiet_zone_test_r0": qz_test_r0,
        "quiet_zone_test_r2": qz_test_r2,
        "quiet_zone_test_r5": qz_test_r5,
    }
    rirmod.save_rir_sets(str(out_path), sets)
    print(f"wrote {out_path}")
    for name, s in sets.items():
        print(f"  {name}: {s.primary.shape[0]} pairs")


if __name__ == "__main__":
    main()
