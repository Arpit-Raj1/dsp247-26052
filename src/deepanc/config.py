"""Minimal config loader for configs/exp/*.yaml.

Not using Hydra (too heavy for this project's needs); instead a small
hand-rolled "defaults" convention:

  configs/exp/base.yaml has a top-level `defaults: {data: default, model: crn}`
  mapping, resolved against configs/data/<name>.yaml and
  configs/model/<name>.yaml.

  configs/exp/<name>.yaml has a top-level `defaults: base`, meaning "start
  from configs/exp/base.yaml (with its own data/model defaults resolved),
  then merge this file's keys on top".

CLI overrides (dotlist strings like "train.epochs=2") are merged last via
OmegaConf.from_dotlist, exactly like Hydra's `key=value` overrides.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

CONFIGS_ROOT = Path(__file__).resolve().parents[2] / "configs"


def _load_base() -> "OmegaConf":
    base_path = CONFIGS_ROOT / "exp" / "base.yaml"
    base_cfg = OmegaConf.load(base_path)
    defaults = base_cfg.pop("defaults")
    data_cfg = OmegaConf.load(CONFIGS_ROOT / "data" / f"{defaults['data']}.yaml")
    model_cfg = OmegaConf.load(CONFIGS_ROOT / "model" / f"{defaults['model']}.yaml")
    return OmegaConf.merge({"data": data_cfg, "model": model_cfg}, base_cfg)


def load_config(exp_path: str | Path, overrides: list[str] | None = None) -> "OmegaConf":
    exp_cfg = OmegaConf.load(exp_path)
    defaults = exp_cfg.pop("defaults", None)
    if defaults == "base":
        merged = OmegaConf.merge(_load_base(), exp_cfg)
    else:
        merged = exp_cfg
    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(overrides))
    OmegaConf.resolve(merged)
    return merged
