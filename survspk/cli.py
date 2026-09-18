"""survspk command line.

  survspk info                                  show profile, paths, db status
  survspk refresh-survivor [--snapshot-only]    pull survivoR tables -> survivor.sqlite
  survspk inventory [--season 47 ...] [--no-probe] [--force]
  survspk ingest-subs [--season US47 ...] [--force]
  survspk names-report [--min-lines 3]
  survspk report                                inventory + names summary to work_root/reports/
  survspk grab-frame US47 2 162.0 [--crop]      chyron frame grab
  survspk missing-subs / fetch-subs              subtitle gaps via Plex

M1 (audio; macOS):
  survspk run US47 2 --through embed [--variants raw,center,vocals,vocals_center]
  survspk extract US47 2 | separate US47 2 [--source raw|center] | align US47 2 [--variant vocals]
  survspk segment US47 2 | embed US47 2 --variant vocals | ablation US47 2
  survspk sync-check US11 1              offset/drift estimate only (needs extract)
  survspk run-errors US47 2              which labelled runs the LOO gets wrong, and why

M2/M3 (bank + assign):
  survspk bank US47 --episodes 1         fit the speaker bank from episode 1's labels (as_of 1)
  survspk assign US47 2                  label episode 2's runs from that bank; agreement vs explicit SDH names
  survspk text-prior US47 2 --eval-only  [experiment] local LLM vote on runs; precision vs explicit names
  survspk review                         review UI (FastAPI) at http://127.0.0.1:8765

Corpus / fine-tuning:
  survspk export-corpus corpus/          rttm + uem + database.yml + spk manifest/trials from every labelled episode
  survspk finetune-ecapa corpus/ models/ecapa-survivor   fine-tune the embedding model; prints baseline EER first
  uv run python scripts/finetune_pyannote_seg.py --corpus corpus/ --out models/seg-survivor
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from rich import print as rprint
from rich.logging import RichHandler
from rich.table import Table

from . import db as dbm
from .config import load_settings

app = typer.Typer(add_completion=False, no_args_is_help=True, pretty_exceptions_enable=False)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 200)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(message)s",
                        handlers=[RichHandler(show_path=False, rich_tracebacks=False)])


def _df_table(df: pd.DataFrame, title: str, max_rows: int = 60) -> None:
    t = Table(title=f"{title}  ({len(df)} rows)")
    for c in df.columns:
        t.add_column(str(c))
    for _, r in df.head(max_rows).iterrows():
        t.add_row(*["" if pd.isna(v) else str(v) for v in r.values])
    rprint(t)


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    _setup_logging(verbose)


@app.command()
def info() -> None:
    """Show the active profile, resolved paths, and database row counts."""
    s = load_settings()
    rprint(f"[bold]profile[/bold]: {s.profile}")
    for k, v in s.paths.model_dump().items():
        rprint(f"  {k:18} {v}  {'[green]ok[/green]' if Path(v).exists() else '[red]missing[/red]'}")
    rprint(f"  {'db':18} {s.db_path}  {'[green]ok[/green]' if s.db_path.exists() else '[yellow]not created[/yellow]'}")
    rprint(f"  {'survivor_db':18} {s.survivor_db_path}  {'[green]ok[/green]' if s.survivor_db_path.exists() else '[yellow]not created[/yellow]'}")
    if s.db_path.exists():
        con = dbm.connect(s.db_path, s.sqlite_journal)
        for t in ("episodes", "subtitle_files", "cues", "utterances", "labels", "name_tokens"):
            n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            rprint(f"  {t:18} {n}")


@app.command("refresh-survivor")
def refresh_survivor(snapshot_only: bool = typer.Option(False, help="Copy from the Gamebot snapshot only; no download"),
                     force: bool = typer.Option(False, help="Re-download cached .rda files")) -> None:
    """Pull survivoR tables (castaways, castaway_details, boot_mapping, episodes, confessionals)."""
    from .refresh_survivor import coverage_summary, refresh

    s = load_settings()
    rep = refresh(s, force=force, snapshot_only=snapshot_only)
    for t, v in rep.items():
        rprint(f"  {t:18} {v['rows']:>7} rows  from {v['source']}")
    cov = coverage_summary(s)
    _df_table(cov.tail(12), "survivoR coverage (last 12 seasons)")
    us50 = cov[cov.version_season == "US50"]
    if not us50.empty and int(us50.n_ep_boot_mapping.iloc[0]) == 0:
        rprint("[yellow]US50 has no boot_mapping rows upstream yet — S50 cannot be assigned until survivoR adds them.[/yellow]")


@app.command()
def inventory(season: Optional[list[int]] = typer.Option(None, "--season", "-s", help="Season number(s); default all"),
              probe: bool = typer.Option(True, help="Run ffprobe on each video (slow over the network)"),
              force: bool = typer.Option(False, help="Re-inventory episodes already probed")) -> None:
    """Scan videos + subtitles, probe media, choose subtitle files, compare with survivoR."""
    from .inventory import build_inventory

    s = load_settings()
    con = dbm.init_db(s.db_path, s.sqlite_journal)
    df = build_inventory(s, con, seasons=season or None, probe=probe, force=force)
    rprint(f"[green]inventoried {len(df)} episodes[/green]")


@app.command("ingest-subs")
def ingest_subs(season: Optional[list[str]] = typer.Option(None, "--season", "-s", help="version_season(s) e.g. US47"),
                include_reunions: bool = False,
                force: bool = typer.Option(False, help="Re-parse episodes already ingested")) -> None:
    """Parse chosen subtitle files into the cues table; detect conventions; tally SDH names."""
    from .stage_ingest_subs import ingest_all

    s = load_settings()
    con = dbm.init_db(s.db_path, s.sqlite_journal)
    df = ingest_all(s, con, seasons=season or None, include_reunions=include_reunions, force=force)
    if not df.empty:
        errs = df[df.get("error").notna()] if "error" in df else df.iloc[0:0]
        rprint(f"[green]ingested {len(df) - len(errs)} episodes[/green]" + (f", [red]{len(errs)} errors[/red]" if len(errs) else ""))
        if len(errs):
            _df_table(errs, "errors")


@app.command("names-report")
def names_report_cmd(min_lines: int = typer.Option(3, help="Only list unresolved tokens with at least this many lines")) -> None:
    """Resolve every SDH NAME: token against survivoR + aliases.yaml and list what is unresolved."""
    from .stage_ingest_subs import names_report

    s = load_settings()
    con = dbm.connect(s.db_path, s.sqlite_journal)
    rep = names_report(s, con, min_lines=min_lines)
    _df_table(rep["per_season"], "SDH name resolution per season", max_rows=60)
    _df_table(rep["unresolved"][["version_season", "token", "n_lines", "n_files"]], "unresolved tokens", max_rows=120)
    out = s.paths.work_root / "reports"
    out.mkdir(parents=True, exist_ok=True)
    rep["per_season"].to_csv(out / "names_per_season.csv", index=False)
    rep["unresolved"].to_csv(out / "names_unresolved.csv", index=False)
    rep["all_tokens"].to_csv(out / "names_all_tokens.csv", index=False)
    rprint(f"written to {out}")


@app.command()
def report() -> None:
    """Inventory summary: per-season coverage, missing subtitles, duration mismatches."""
    from .inventory import inventory_report

    s = load_settings()
    con = dbm.connect(s.db_path, s.sqlite_journal)
    rep = inventory_report(con, s)
    out = s.paths.work_root / "reports"
    out.mkdir(parents=True, exist_ok=True)
    for k, df in rep.items():
        _df_table(df, k)
        df.to_csv(out / f"inventory_{k}.csv", index=False)
    rprint(f"written to {out}")


@app.command("grab-frame")
def grab_frame_cmd(version_season: str, episode: int, t_s: float,
                   crop: bool = typer.Option(False, help="Crop to the franchise chyron band"),
                   out: Optional[Path] = None) -> None:
    """Grab a frame from an episode's video at t_s seconds (chyron check)."""
    from .chyron import grab_frame

    s = load_settings()
    con = dbm.connect(s.db_path, s.sqlite_journal)
    r = con.execute("SELECT video_path FROM episodes WHERE version_season=? AND episode=?", (version_season, episode)).fetchone()
    if not r:
        raise typer.BadParameter("episode not in inventory")
    crop_frac = tuple(s.franchise_for(version_season).chyron_crop) if crop else None
    out = out or s.work("frames") / f"{version_season}E{episode:02d}_{int(t_s * 1000):08d}{'_crop' if crop else ''}.png"
    p = grab_frame(dbm.localize(con, s, r["video_path"]), t_s, out, crop_frac=crop_frac)
    rprint(f"wrote {p}")


@app.command("missing-subs")
def missing_subs() -> None:
    """List episodes whose subtitle is missing or truncated (candidates for fetch-subs)."""
    from .plex_subs import flagged_episodes

    s = load_settings()
    con = dbm.connect(s.db_path, s.sqlite_journal)
    for r in flagged_episodes(con):
        why = "no subtitle" if r["subtitle_source"] == "none" else \
            f"truncated: ends at {100 * r['last_cue_end_s'] / r['duration_s']:.0f}% of file"
        rprint(f"  {r['version_season']} E{r['episode']:02d}  {why:38}  {r['video_basename']}")


@app.command("fetch-subs")
def fetch_subs(version_season: Optional[str] = typer.Option(None, "--season", "-s", help="e.g. US50; default: all flagged"),
               episode: Optional[int] = typer.Option(None, "--episode", "-e"),
               show_title: str = typer.Option("Survivor", help="Show title in Plex"),
               tvdb_id: Optional[int] = typer.Option(76733, help="TVDB id of the show (from the folder name); disambiguates from Australian Survivor"),
               year: Optional[int] = typer.Option(2000, help="Show year, fallback disambiguation"),
               dry_run: bool = typer.Option(False, help="Only list Plex's subtitle candidates"),
               reingest: bool = typer.Option(True, help="Re-run inventory + ingest for fetched episodes")) -> None:
    """Fetch subtitles through Plex's subtitle agent for flagged (or given) episodes. Needs PLEX_URL/PLEX_TOKEN."""
    from .inventory import build_inventory
    from .plex_subs import _find_show, _plex, fetch_for_episode, flagged_episodes
    from .stage_ingest_subs import ingest_all

    s = load_settings()
    con = dbm.init_db(s.db_path, s.sqlite_journal)
    if version_season and episode:
        rows = con.execute("SELECT e.*, NULL AS duration_delta_s, NULL AS last_cue_end_s FROM episodes e "
                           "WHERE version_season=? AND episode=?", (version_season, episode)).fetchall()
    else:
        rows = [r for r in flagged_episodes(con) if not version_season or r["version_season"] == version_season]
    plex = _plex()
    show = _find_show(plex, show_title, tvdb_id, year)
    rprint(f"Plex show: [bold]{show.title}[/bold] ({getattr(show, 'year', '?')})  guid={getattr(show, 'guid', '')}")
    fetched = []
    for r in rows:
        try:
            p = fetch_for_episode(s, plex, show, r, dry_run=dry_run)
            if p:
                fetched.append(r)
        except Exception as e:  # noqa: BLE001
            rprint(f"[red]{r['version_season']} E{r['episode']:02d}: {e}[/red]")
    if fetched and reingest:
        for r in fetched:
            build_inventory(s, con, seasons=[int(r["version_season"][2:])], probe=True, force=True)
        ingest_all(s, con, seasons=sorted({r["version_season"] for r in fetched}), force=True)
    rprint(f"[green]fetched {len(fetched)} subtitle files[/green]")


# ============================================================================= M1: audio stages

STAGES = ["extract", "separate", "align", "segment", "embed"]


def _resolver(s):
    from .aliases import Resolver
    return Resolver(s) if s.survivor_db_path.exists() else None


@app.command()
def extract(version_season: str, episode: int, force: bool = False) -> None:
    """ffmpeg: raw 16 kHz mono FLAC (+ front-center channel for 5.1 sources)."""
    from .stage_extract import extract_episode

    s = load_settings()
    con = dbm.init_db(s.db_path, s.sqlite_journal)
    rprint(extract_episode(s, con, version_season, episode, force=force))


@app.command()
def separate(version_season: str, episode: int,
             source: Optional[str] = typer.Option(None, help="raw | center (default: config audio.separation_source)"),
             force: bool = False) -> None:
    """demucs-mlx vocals stem -> vocals (from raw) or vocals_center (from center)."""
    from .stage_separate import separate_episode

    s = load_settings()
    con = dbm.init_db(s.db_path, s.sqlite_journal)
    rprint(str(separate_episode(s, con, version_season, episode, source=source, force=force)))


@app.command()
def align(version_season: str, episode: int,
          variant: Optional[str] = typer.Option(None, help="audio variant to align against (default config audio.variant)"),
          force: bool = False) -> None:
    """Global offset/drift estimate + WhisperX word timestamps."""
    from .stage_align import align_episode

    s = load_settings()
    con = dbm.init_db(s.db_path, s.sqlite_journal)
    st = align_episode(s, con, version_season, episode, variant=variant, force=force)
    rprint({k: v for k, v in st.items() if k != "windows"})


@app.command("sync-check")
def sync_check(version_season: str, episode: Optional[int] = typer.Option(None, "--episode", "-e"),
               variant: str = typer.Option("raw", help="raw | center | vocals — vocals is far more reliable")) -> None:
    """Offset/drift estimate only (no WhisperX). Runs extract if needed (raw/center only). Use on the M0 drift list."""
    from .stage_align import estimate_offset_drift, load_audio, speech_envelope
    from .stage_extract import extract_episode

    s = load_settings()
    con = dbm.init_db(s.db_path, s.sqlite_journal)
    eps = [episode] if episode else [r[0] for r in con.execute(
        "SELECT episode FROM episodes WHERE version_season=? AND is_reunion=0 ORDER BY episode", (version_season,))]
    for ep in eps:
        p = s.audio_path(variant, version_season, ep)
        if not p.exists() and variant in ("raw", "center"):
            extract_episode(s, con, version_season, ep)
        if not p.exists():
            rprint(f"[red]{p} missing (run separate first)[/red]")
            continue
        cues = [(r[0], r[1]) for r in con.execute("SELECT start_s, end_s FROM cues WHERE version_season=? AND episode=? ORDER BY idx",
                                                  (version_season, ep))]
        env, kind = speech_envelope(load_audio(p, s.audio.sample_rate), s.audio.sample_rate)
        fit = estimate_offset_drift(env, cues, s.align.offset_windows, s.align.offset_window_s, s.align.max_lag_s,
                                    apply_min_offset_s=s.align.apply_min_offset_s)
        rprint(f"{version_season} E{ep:02d} ({variant}/{kind})  offset={fit.offset_s:+.2f}s  drift={fit.drift:+.4f}  applied={fit.applied}  {fit.note}")
        for w in fit.windows:
            rprint(f"      t={w['t_center']:6.0f}s  lag={w['lag_s']:+.2f}s  peak={w['peak']:.3f}")


@app.command()
def segment(version_season: str, episode: int,
            force: bool = typer.Option(False, help="re-segment even if human labels exist (they will be lost)")) -> None:
    """Cues -> turns -> utterances; SDH names; recap/preview."""
    from .stage_segment import segment_episode

    s = load_settings()
    con = dbm.init_db(s.db_path, s.sqlite_journal)
    rprint(segment_episode(s, con, version_season, episode, resolver=_resolver(s), allow_relabel=force))


@app.command()
def embed(version_season: str, episode: int,
          variant: Optional[str] = typer.Option(None, help="raw | center | vocals | vocals_center"),
          model: Optional[str] = typer.Option(None, help="speechbrain/... or pyannote/... (default: config embed.model)"),
          force: bool = False) -> None:
    """Speaker embeddings per utterance for one audio variant."""
    from .stage_embed import embed_episode

    s = load_settings()
    con = dbm.init_db(s.db_path, s.sqlite_journal)
    rprint(str(embed_episode(s, con, version_season, episode, variant=variant, force=force, model=model)))


@app.command()
def ablation(version_season: str, episode: int,
             variants: Optional[str] = typer.Option(None, help="comma list; default: every variant with embeddings"),
             model: Optional[str] = typer.Option(None, help="embedding model whose parquet to score"),
             plot: bool = True) -> None:
    """Leave-one-out speaker ID on SDH-labelled utterances and runs, per audio variant (+ t-SNE plots)."""
    from .stage_embed import ablation as _ablation

    s = load_settings()
    con = dbm.init_db(s.db_path, s.sqlite_journal)
    df = _ablation(s, con, version_season, episode, variants.split(",") if variants else None, plot=plot, model=model)
    _df_table(df, f"ablation {version_season} E{episode:02d}")
    rprint(f"csv + t-SNE plots in {s.paths.work_root / 'reports'}")


@app.command()
def bank(version_season: str,
         episodes: str = typer.Option(..., help="comma list of episodes whose labels fit the bank, e.g. 1 or 1,2,3"),
         as_of: Optional[int] = typer.Option(None, help="store as as_of_episode (default: max of --episodes)"),
         variant: Optional[str] = typer.Option(None), model: Optional[str] = typer.Option(None),
         show_dropped: bool = typer.Option(True, help="list utterances dropped as label-inconsistent")) -> None:
    """Fit the speaker bank (spec §7.9) from explicit SDH / human / chyron / confident-auto labels."""
    from .stage_bank import build_bank

    s = load_settings(version_season)
    con = dbm.init_db(s.db_path, s.sqlite_journal)
    eps = [int(x) for x in episodes.split(",")]
    st = build_bank(s, con, version_season, eps, variant=variant, model=model, as_of=as_of, resolver=_resolver(s))
    rprint(f"bank {version_season} as of E{st['as_of']:02d}: {st['n_speakers']} speakers, {st['n_entries']} entries, "
           f"{st['n_kept']} utts kept / {st['n_dropped']} dropped ({st['n_self_mention']} named themselves); "
           f"sources {st['sources']}")
    per = st["per_speaker"].copy()
    per.columns = [f"{a}_{b}" for a, b in per.columns]
    _df_table(per.round(1).reset_index(), "labelled utterances per speaker and domain")
    d = st["dropped"]
    if len(d):
        by = d.groupby("speaker_id").agg(n_dropped=("utt_id", "size"), dur_dropped=("dur", "sum")).round(1)
        by["n_kept"] = st["per_speaker"]["n"].sum(axis=1).reindex(by.index).fillna(0).astype(int)
        _df_table(by.reset_index(), "dropped per speaker (a speaker losing most of its pool = check the labels by ear)")
    if show_dropped and len(d):
        d = d.assign(mmss=d.start_s.apply(lambda x: f"{int(x // 60):02d}:{int(x % 60):02d}"),
                     text=d.text.str.slice(0, 80))
        _df_table(d[["episode", "mmss", "speaker_id", "pred", "sim_true", "sim_pred", "dur", "label_source", "text"]].round(2),
                  "dropped as inconsistent (label says X, voice says Y)")


@app.command()
def assign(version_season: str, episode: int,
           bank_as_of: Optional[int] = typer.Option(None, help="use the bank stored as of this episode (default: episode-1 or earlier)"),
           variant: Optional[str] = typer.Option(None), model: Optional[str] = typer.Option(None),
           write: bool = typer.Option(True, help="--no-write: score only, touch nothing"),
           show: int = typer.Option(30, help="rows of disagreements / queue to print")) -> None:
    """Label an episode's runs from the speaker bank (spec §7.7); report agreement with explicit SDH names."""
    from .stage_assign import assign_episode
    from .stage_bank import StaleEmbeddings

    s = load_settings(version_season)
    con = dbm.init_db(s.db_path, s.sqlite_journal)
    try:
        df, st = assign_episode(s, con, version_season, episode, variant=variant, model=model, bank_as_of=bank_as_of,
                                resolver=_resolver(s), write=write)
    except StaleEmbeddings as e:
        rprint(f"[yellow]{e}[/yellow]\n-> re-embedding {version_season} E{episode:02d} ({variant or s.audio.variant}) first")
        from .stage_embed import embed_episode
        embed_episode(s, con, version_season, episode, variant=variant, model=model, force=True)
        df, st = assign_episode(s, con, version_season, episode, variant=variant, model=model, bank_as_of=bank_as_of,
                                resolver=_resolver(s), write=write)
    keys = ["bank_as_of", "n_candidates", "n_bankable", "unbankable", "n_runs", "n_split_runs", "decisions",
            "body_auto_rate", "body_auto_dur_share", "sdh_agreement_runs", "n_sdh_runs", "sdh_agreement_confident",
            "n_sdh_runs_confident", "sdh_agreement_confessional", "n_sdh_runs_confessional",
            "sdh_agreement_confessional_confident", "n_sdh_runs_confessional_confident", "auto_dur_share_confessional",
            "sdh_agreement_field", "n_sdh_runs_field", "sdh_agreement_field_confident", "n_sdh_runs_field_confident",
            "auto_dur_share_field", "inherited_agreement", "n_inherited_runs", "n_name_mentioned",
            "n_labels_written", "n_queued", "queued_dur_by_pred"]
    for k in keys:
        if k in st:
            rprint(f"  {k:40s} {st[k]}")
    body = df[df.segment == "body"]
    dis = body[body.explicit.notna() & (body.pred != body.explicit)].sort_values("margin", ascending=False)
    cols = ["mmss", "domain", "dur", "n_utts", "split", "explicit", "pred", "score", "margin", "decision", "text"]
    if len(dis):
        _df_table(dis[cols], "runs where the bank disagrees with the explicit SDH name", max_rows=show)
    nm = df[df.decision == "name_mentioned"]
    if len(nm):
        _df_table(nm[cols], "confident but the run names its predicted speaker -> queued (mention rule)", max_rows=show)
    q = body[body.decision.isin(["low_margin", "no_candidate"]) & body.explicit.isna()].sort_values("dur", ascending=False)
    if len(q):
        _df_table(q[cols], "queued for review (longest first)", max_rows=show)
    rep = s.paths.work_root / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    out = rep / f"assign_{version_season}E{episode:02d}.csv"
    df.to_csv(out, index=False)
    rprint(f"all runs -> {out}")


@app.command("text-prior")
def text_prior_cmd(version_season: str, episode: int,
                   all_runs: bool = typer.Option(False, "--all", help="score every body run, not just queued + labelled ones"),
                   limit: Optional[int] = typer.Option(None, help="stop after N runs (smoke test)"),
                   force: bool = typer.Option(False, help="re-ask runs already scored"),
                   base_url: Optional[str] = typer.Option(None, help="e.g. http://127.0.0.1:8080/v1 (or OMLX_BASE_URL in .env)"),
                   model: Optional[str] = typer.Option(None, help="model id as GET /v1/models lists it (or OMLX_MODEL)"),
                   thinking: bool = typer.Option(False, help="let the model reason before answering (slower)"),
                   eval_only: bool = typer.Option(False, "--eval-only", help="score only runs with an explicit SDH name and do not store: A/B a prompt or model"),
                   clear: bool = typer.Option(False, help="delete stored text_prior rows for this episode and exit"),
                   show: int = typer.Option(40)) -> None:
    """[experiment] Ask a local LLM (oMLX) who speaks each queued run, from text + cast sheet + context; report
    precision against explicit SDH names. US47E02: 26-41% — not usable as a labeler. Off the default path."""
    from .text_prior import LLMCfg, evaluate, ping, run_text_prior

    s = load_settings(version_season)
    con = dbm.init_db(s.db_path, s.sqlite_journal)
    if clear:
        try:
            n = con.execute("DELETE FROM text_prior WHERE version_season=? AND episode=?", (version_season, episode)).rowcount
            con.commit()
        except Exception:  # noqa: BLE001
            n = 0
        rprint(f"removed {n} text_prior rows for {version_season} E{episode:02d}")
        raise typer.Exit()
    cfg = LLMCfg.from_settings(s, base_url=base_url, model=model)
    if thinking:
        cfg.disable_thinking = False
    try:
        models = ping(cfg)
    except Exception as e:  # noqa: BLE001
        rprint(f"[red]cannot reach {cfg.base_url}: {e}[/red]")
        rprint("oMLX's port is in its settings (~/.omlx/settings.json or the admin panel URL); pass --base-url "
               "http://127.0.0.1:<port>/v1 or put OMLX_BASE_URL / OMLX_API_KEY in .env")
        raise typer.Exit(1)
    rprint(f"oMLX models: {models}  -> using {cfg.model}" + ("" if cfg.model in models else "  [yellow](not listed)[/yellow]"))
    r = _resolver(s)
    if r is None:
        rprint("[red]survivoR db missing; run refresh-survivor first[/red]")
        raise typer.Exit(1)
    df = run_text_prior(s, con, version_season, episode, r, only_queued=not all_runs, limit=limit, force=force or eval_only,
                        cfg=cfg, eval_only=eval_only, write=not eval_only)
    if df.empty:
        rprint("nothing new scored (use --force to redo)")
        raise typer.Exit()
    ev = evaluate(df)
    for k, v in ev.items():
        rprint(f"  {k:28s} {v}")
    rprint(f"  {'mean latency s':28s} {df.latency_s.mean():.1f}")
    df["dur"] = df.dur.round(1)
    df["reason"] = df.reason.str.slice(0, 90)
    cols = ["mmss", "dur", "explicit", "speaker_id", "speaker_name", "confidence", "queued", "reason", "text"]
    wrong = df[df.explicit.notna() & df.speaker_id.notna() & (df.speaker_id != df.explicit)]
    if len(wrong):
        _df_table(wrong[cols].sort_values("confidence", ascending=False), "text prior disagrees with explicit SDH name", max_rows=show)
    q = df[df.queued & df.explicit.isna()].sort_values("confidence", ascending=False)
    if len(q):
        _df_table(q[cols], "text prior on queued runs without a label (what it would add)", max_rows=show)
    rep = s.paths.work_root / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    out = rep / f"text_prior_{version_season}E{episode:02d}.csv"
    df.to_csv(out, index=False)
    rprint(f"-> {out}")


@app.command()
def review(port: int = typer.Option(8765), host: str = typer.Option("127.0.0.1")) -> None:
    """Start the review UI (needs `uv sync --extra review --extra audio`): http://127.0.0.1:8765"""
    try:
        from .review_app import serve
    except ImportError as e:
        rprint(f"[red]{e}[/red]  -> uv sync --extra review --extra audio")
        raise typer.Exit(1)
    rprint(f"review UI -> http://{host}:{port}   (Ctrl-C to stop)")
    serve(host, port)


@app.command("export-corpus")
def export_corpus_cmd(out: Path = typer.Argument(..., help="corpus directory to write"),
                      season: Optional[list[str]] = typer.Option(None, "--season", "-s", help="version_season(s); default: every labelled episode"),
                      variant: str = typer.Option("raw", help="audio the corpus points at: raw (natural mix) | vocals"),
                      test_season: Optional[list[str]] = typer.Option(None, help="whole seasons held out as test"),
                      no_vad: bool = typer.Option(False, help="do not bridge silent gaps into the annotated regions")) -> None:
    """Export every labelled episode as a pyannote diarization database (rttm/uem/lists/database.yml) and a
    speaker-embedding manifest with dev verification trials (spk/). Trusted labels only: human + what the bank keeps."""
    from .export_corpus import export_corpus

    s = load_settings()
    con = dbm.init_db(s.db_path, s.sqlite_journal)
    st = export_corpus(s, con, out, seasons=season or None, variant=variant, resolver=_resolver(s),
                       test_seasons=test_season or None, use_vad=not no_vad)
    rprint(f"{st['n_episodes']} episodes ({st['splits']}), {st['speech_hours']} h labelled speech, "
           f"{st['annotated_hours']} h annotated, {st['n_speakers']} speakers, spk utts {st['spk_utts']}, "
           f"{st['n_trials_dev']} dev trials -> {out}")
    per = pd.DataFrame([{"speaker_id": k, "seconds": v} for k, v in st["per_speaker_s"].items()])
    if len(per):
        _df_table(per.head(40), "labelled seconds per speaker")


@app.command("finetune-ecapa")
def finetune_ecapa_cmd(corpus: Path = typer.Argument(..., help="directory written by export-corpus"),
                       out: Path = typer.Argument(..., help="checkpoint directory (usable as `embed --model <out>`)"),
                       epochs: int = 10, batch_size: int = 32, crop_s: float = 3.0,
                       lr_encoder: float = 1e-4, lr_head: float = 1e-3, freeze_epochs: int = 1,
                       min_utts_per_speaker: int = 8, device: str = "auto",
                       smoke: bool = typer.Option(False, help="random-init model, no downloads: exercise the loop")) -> None:
    """Fine-tune ECAPA on the exported speaker manifest; prints the pretrained model's dev EER first (epoch 0)."""
    from .finetune_ecapa import TrainCfg, train

    cfg = TrainCfg(corpus=corpus, out=out, epochs=epochs, batch_size=batch_size, crop_s=crop_s, lr_encoder=lr_encoder,
                   lr_head=lr_head, freeze_epochs=freeze_epochs, min_utts_per_speaker=min_utts_per_speaker,
                   device=device, smoke=smoke)
    res = train(cfg)
    _df_table(pd.DataFrame(res["history"]), f"fine-tune history ({res['n_speakers']} speakers, {res['device']})")
    rprint(f"checkpoint -> {res['out']}   (use: survspk embed US47 2 --model {res['out']})")


@app.command("run-errors")
def run_errors_cmd(version_season: str, episode: int,
                   variant: Optional[str] = typer.Option(None, help="audio variant; default audio.variant"),
                   model: Optional[str] = typer.Option(None, help="embedding model whose parquet to score"),
                   domain: str = typer.Option("confessional", help="confessional | field | all"),
                   min_run_s: float = typer.Option(0.0, help="only runs at least this long"),
                   show: int = typer.Option(40, help="max error rows to print")) -> None:
    """Which runs does the leave-one-out speaker ID get wrong, and why? Prints per-speaker accuracy, the confusion
    pairs, and every misclassified run with its time (mm:ss), label provenance (n_explicit NAME: lines) and text,
    so each can be checked by ear. Also writes reports/run_errors_<vs>E<ep>_<variant>_<model>.csv."""
    from .stage_embed import model_tag, run_errors

    s = load_settings()
    con = dbm.init_db(s.db_path, s.sqlite_journal)
    v = variant or s.audio.variant
    res = run_errors(s, con, version_season, episode, v, model=model, domain=None if domain == "all" else domain,
                     min_run_s=min_run_s)
    runs, errors = res["runs"], res["errors"]
    if runs.empty:
        rprint("no labelled runs match")
        raise typer.Exit()
    rprint(f"{len(runs)} runs, {len(errors)} wrong  ->  accuracy {1 - len(errors) / len(runs):.3f}   "
           f"({v}, {model_tag(model or s.embed.model)}, domain={domain})")
    per = res["per_speaker"].reset_index()
    _df_table(per, "per speaker (acc = own runs correct; n_wrong_as = other speakers' runs predicted as this one)")
    if not res["confusions"].empty:
        _df_table(res["confusions"], "confusion pairs (true -> pred)")
    cols = ["mmss", "speaker_id", "pred", "second", "margin", "sim_true", "run_dur", "n_utts", "n_explicit", "text"]
    e = errors[cols].copy()
    for c in ("margin", "sim_true", "run_dur"):
        e[c] = e[c].round(2)
    _df_table(e, "misclassified runs (highest-confidence mistakes first)", max_rows=show)
    rep = s.paths.work_root / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    out = rep / f"run_errors_{version_season}E{episode:02d}_{v}_{model_tag(model or s.embed.model)}.csv"
    runs.drop(columns=["vector"]).to_csv(out, index=False)
    rprint(f"all scored runs -> {out}")


@app.command()
def run(version_season: str, episode: int,
        through: str = typer.Option("embed", help="last stage to run: " + " | ".join(STAGES)),
        from_stage: str = typer.Option("extract", "--from", help="first stage to run (earlier outputs are reused)"),
        variants: str = typer.Option("raw,center,vocals,vocals_center",
                                     help="audio variants to produce/embed (center ones only exist for 5.1)"),
        model: Optional[str] = typer.Option(None, help="embedding model (default: config embed.model)"),
        force: bool = False) -> None:
    """Run the M1 pipeline for one episode: extract -> separate -> align -> segment -> embed."""
    from .stage_align import align_episode
    from .stage_embed import _load_encoder, embed_episode
    from .stage_extract import extract_episode
    from .stage_segment import segment_episode
    from .stage_separate import separate_episode

    s = load_settings()
    con = dbm.init_db(s.db_path, s.sqlite_journal)
    want = [v.strip() for v in variants.split(",") if v.strip()]
    first, last = STAGES.index(from_stage), STAGES.index(through)
    vs, ep = version_season, episode

    def do(stage: str) -> bool:
        return first <= STAGES.index(stage) <= last

    def f(stage: str) -> bool:      # --force applies only to stages at/after --from
        return force and STAGES.index(stage) >= first

    out = extract_episode(s, con, vs, ep, force=f("extract"))   # cheap no-op when files exist
    have_center = "center" in out
    if do("separate"):
        if "vocals" in want:
            separate_episode(s, con, vs, ep, source="raw", force=f("separate"))
        if "vocals_center" in want and have_center:
            separate_episode(s, con, vs, ep, source="center", force=f("separate"))
    if do("align"):
        align_episode(s, con, vs, ep, force=f("align"))
    if do("segment"):
        segment_episode(s, con, vs, ep, resolver=_resolver(s), allow_relabel=force)
    if do("embed"):
        enc = _load_encoder(model or s.embed.model, s.embed.device, s.embed.batch_size)
        for v in want:
            if s.audio_path(v, vs, ep).exists():
                embed_episode(s, con, vs, ep, variant=v, force=f("embed"), encoder=enc, model=model)
            else:
                rprint(f"[yellow]skip {v}: no audio[/yellow]")
    rprint(f"[green]{vs} E{ep:02d} done through {through}[/green]")


if __name__ == "__main__":
    app()
