"""Configuration loading.

Layered: config/default.yaml  <-  config/seasons/<version_season>.yaml (optional)  <-  environment.
Paths come from a *profile* (macos | vm) so the same config works on the Mac and in the Cowork VM.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"


PATH_KEYS = ("video_root", "subtitle_root", "work_root", "survivor_snapshot")


class Paths(BaseModel):
    video_root: Path
    subtitle_root: Path
    work_root: Path
    survivor_snapshot: Path

    @classmethod
    def from_profile(cls, d: dict[str, str]) -> "Paths":
        return cls(**{k: Path(os.path.expanduser(d[k])) for k in PATH_KEYS})


class Layout(BaseModel):
    db: str = "db/survspk.sqlite"
    survivor_db: str = "survivor_data/survivor.sqlite"
    raw: str = "raw"
    center: str = "center"
    vocals: str = "vocals"
    embeddings: str = "embeddings"
    frames: str = "frames"


class SurvivorSource(BaseModel):
    raw_url: str
    tables: list[str]


class InventoryCfg(BaseModel):
    video_extensions: list[str]
    skip_title_patterns: list[str] = Field(default_factory=list)
    prefer_sdh: bool = True
    max_duration_delta_s: float = 20
    max_length_delta_min: float = 3
    ffprobe_timeout_s: int = 60


class SubparseCfg(BaseModel):
    name_prefix_min_lines: int = 20
    marker_min_lines: int = 20


class AudioCfg(BaseModel):
    sample_rate: int = 16000
    variant: str = "vocals"                 # default variant for align/embed: raw | center | vocals | vocals_center
    separation_source: str = "raw"          # what demucs sees: raw | center (center only exists for 5.1 sources)
    make_center: str = "when_51"            # when_51 | never
    keep_raw: bool = True                   # keep raw FLAC after separation (M1: yes, for the ablation)
    demucs_model: str = "htdemucs"
    device: str = "mps"


class AlignCfg(BaseModel):
    pad_s: float = 0.4                      # audio padding around each cue before word alignment
    min_word_score: float = 0.3
    offset_windows: int = 5                 # windows for the global offset/drift fit
    offset_window_s: float = 240.0
    max_lag_s: float = 90.0
    apply_min_offset_s: float = 0.25        # below this, treat the subtitle as already in sync
    device: str = "mps"
    anchor: str = "always"                  # ASR-anchored time map: always | never (envelope fit only)
    anchor_model: str = "small.en"          # faster-whisper model; the words are cached under reports/align
    anchor_backend: str = "mlx"             # mlx (Apple GPU, scripts/asr_mlx.py) | faster-whisper (CPU)
    anchor_mlx_model: str = "mlx-community/whisper-small.en-mlx"
    anchor_min_share: float = 0.3           # share of cue starts ASR must hear, else the envelope fit is kept


class SegmentCfg(BaseModel):
    merge_gap_s: float = 0.8
    merge_max_s: float = 12.0
    sdh_propagate_gap_s: float = 2.5
    bank_min_duration_s: float = 1.5
    recap_search_s: float = 240.0
    recap_gap_s: float = 4.0
    recap_max_s: float = 420.0
    run_gap_s: float = 1.5                  # a run (monologue) continues across gaps shorter than this
    confessional_run_s: float = 6.0         # runs at least this long are labelled domain 'confessional'


class EmbedCfg(BaseModel):
    model: str = "speechbrain/spkrec-ecapa-voxceleb"
    pad_s: float = 0.1
    batch_size: int = 32
    device: str = "mps"
    min_duration_s: float = 0.5


class Franchise(BaseModel):
    host_id: str
    host_names: list[str]
    chyron_crop: list[float]


class Settings(BaseModel):
    profile: str
    sqlite_journal: str = "WAL"     # WAL on macOS; TRUNCATE on the FUSE mount in the VM (cannot delete journals)
    paths: Paths
    layout: Layout
    survivor: SurvivorSource
    inventory: InventoryCfg
    subparse: SubparseCfg
    audio: AudioCfg
    align: AlignCfg
    segment: SegmentCfg
    embed: EmbedCfg
    franchises: dict[str, Franchise]
    raw: dict[str, Any]  # everything else, untyped for now (thresholds, models, ...)

    # ----- derived paths -----
    @property
    def db_path(self) -> Path:
        return self.paths.work_root / self.layout.db

    @property
    def survivor_db_path(self) -> Path:
        return self.paths.work_root / self.layout.survivor_db

    def work(self, key: str) -> Path:
        return self.paths.work_root / getattr(self.layout, key)

    def audio_path(self, variant: str, version_season: str, episode: int) -> Path:
        """FLAC path for an audio variant: raw | center | vocals | vocals_center."""
        sub = {"raw": self.layout.raw, "center": self.layout.center, "vocals": self.layout.vocals,
               "vocals_center": self.layout.vocals + "_center"}[variant]
        return self.paths.work_root / sub / version_season / f"E{episode:02d}.flac"

    def franchise_for(self, version_season: str) -> Franchise:
        return self.franchises[version_season[:2]]

    @property
    def aliases_path(self) -> Path:
        return CONFIG_DIR / "aliases.yaml"


def detect_profile() -> str:
    env = os.environ.get("SURVSPK_PROFILE")
    if env:
        return env
    if (Path.home() / "mnt" / "Stargazer").exists():
        return "vm"
    return "macos"


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@lru_cache(maxsize=8)
def load_settings(version_season: str | None = None, config_path: Path | None = None) -> Settings:
    base_path = config_path or Path(os.environ.get("SURVSPK_CONFIG", CONFIG_DIR / "default.yaml"))
    with open(base_path) as f:
        cfg = yaml.safe_load(f)
    if version_season:
        override = CONFIG_DIR / "seasons" / f"{version_season}.yaml"
        if override.exists():
            with open(override) as f:
                cfg = _deep_merge(cfg, yaml.safe_load(f) or {})
    profile = detect_profile()
    if profile not in cfg["profiles"]:
        raise KeyError(f"profile {profile!r} not in config; have {list(cfg['profiles'])}")
    prof = cfg["profiles"][profile]
    return Settings(
        profile=profile,
        sqlite_journal=prof.get("sqlite_journal", "WAL"),
        paths=Paths.from_profile(prof),
        layout=Layout(**cfg.get("layout", {})),
        survivor=SurvivorSource(**cfg["survivor"]),
        inventory=InventoryCfg(**cfg["inventory"]),
        subparse=SubparseCfg(**cfg.get("subparse", {})),
        audio=AudioCfg(**cfg.get("audio", {})),
        align=AlignCfg(**cfg.get("align", {})),
        segment=SegmentCfg(**cfg.get("segment", {})),
        embed=EmbedCfg(**cfg.get("embed", {})),
        franchises={k: Franchise(**v) for k, v in cfg["franchises"].items()},
        raw=cfg,
    )
