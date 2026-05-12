from __future__ import annotations

import argparse
import sys
from pathlib import Path

import build_graphs as bg

KAPPA_GRID = [0.05, 0.10, 0.15, 0.20, 0.25]
FD_GRID = [0.3, 0.5, 0.9]
PRIMARY_FD = 0.5
PRIMARY_KAPPA = 0.10


def _manifest_path(results_dir: Path, fd: float, kappa: float) -> Path:
    return results_dir / f"manifest_FD{fd}_kappa{kappa}.csv"


def _qc_dir(results_dir: Path, fd: float, kappa: float) -> Path:
    return results_dir / f"qc_FD{fd}_kappa{kappa}"


def _run_one(cohort: list[str], fd: float, kappa: float, results_dir: Path,
             cache_dir: Path, stats: bg.CacheStats) -> tuple[bg.pd.DataFrame, dict]:
    cfg = bg.Config(fd_threshold=fd, kappa=kappa,
                    cache_dir=cache_dir, results_dir=results_dir)
    manifest = bg.build_all(cohort, cfg, _manifest_path(results_dir, fd, kappa), stats=stats)
    flags = bg.qc_report(manifest, _qc_dir(results_dir, fd, kappa))
    return manifest, flags


def cmd_primary(cohort, results_dir, cache_dir):
    print(f"--- PRIMARY: FD={PRIMARY_FD}, κ={PRIMARY_KAPPA}, N={len(cohort)} ---")
    stats = bg.CacheStats()
    manifest, flags = _run_one(cohort, PRIMARY_FD, PRIMARY_KAPPA, results_dir, cache_dir, stats)
    print(f"  L1 misses={stats.layer1_misses} hits={stats.layer1_hits} | "
          f"L2 misses={stats.layer2_misses} hits={stats.layer2_hits} | "
          f"L3 misses={stats.layer3_misses} hits={stats.layer3_hits}")
    print(f"  flags: {flags}")
    return manifest


def cmd_kappa(cohort, results_dir, cache_dir):
    print(f"--- κ SWEEP: FD={PRIMARY_FD}, κ ∈ {KAPPA_GRID}, N={len(cohort)} ---")
    primary_cfg = bg.Config(fd_threshold=PRIMARY_FD, kappa=PRIMARY_KAPPA,
                            cache_dir=cache_dir, results_dir=results_dir)
    primary_path = _manifest_path(results_dir, PRIMARY_FD, PRIMARY_KAPPA)
    if not primary_path.exists():
        sys.exit("ERROR: run `sweep_driver.py primary` first — κ sweep depends on its Layer 2 cache.")
    primary_manifest = bg.pd.read_csv(primary_path)
    n_with_l2 = int(primary_manifest["layer2_cache"].fillna("").str.len().gt(0).sum())

    for kappa in KAPPA_GRID:
        stats = bg.CacheStats()
        _, flags = _run_one(cohort, PRIMARY_FD, kappa, results_dir, cache_dir, stats)
        print(f"  κ={kappa}: L1 hits/miss={stats.layer1_hits}/{stats.layer1_misses}  "
              f"L2 hits/miss={stats.layer2_hits}/{stats.layer2_misses}  "
              f"L3 hits/miss={stats.layer3_hits}/{stats.layer3_misses}  flags={flags}")
        assert stats.layer2_misses == 0, (
            f"κ sweep recomputed Layer 2 ({stats.layer2_misses} misses) — "
            f"cache key drift or primary did not populate Layer 2."
        )
        assert stats.layer2_hits == n_with_l2, (
            f"Layer 2 hit count {stats.layer2_hits} ≠ N_with_L2 from primary "
            f"({n_with_l2}). Cache key drift?"
        )


def cmd_fd(cohort, results_dir, cache_dir):
    print(f"--- FD SWEEP: FD ∈ {FD_GRID}, κ={PRIMARY_KAPPA}, N={len(cohort)} ---")
    for fd in FD_GRID:
        stats = bg.CacheStats()
        _, flags = _run_one(cohort, fd, PRIMARY_KAPPA, results_dir, cache_dir, stats)
        print(f"  FD={fd}: L1 hits/miss={stats.layer1_hits}/{stats.layer1_misses}  "
              f"L2 hits/miss={stats.layer2_hits}/{stats.layer2_misses}  "
              f"L3 hits/miss={stats.layer3_hits}/{stats.layer3_misses}  flags={flags}")
        if abs(fd - PRIMARY_FD) < 1e-9:
            assert stats.layer1_misses == 0, (
                f"FD={fd} is the primary value but recomputed Layer 1 ({stats.layer1_misses} misses) — "
                f"primary cache missing or key drift."
            )
        else:
            assert stats.layer1_misses > 0, (
                f"FD={fd} should miss Layer 1 (new key) but had 0 misses — stale cache?"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["primary", "kappa", "fd", "all"])
    ap.add_argument("--cache-dir", default="cache", type=Path)
    ap.add_argument("--results-dir", default="results", type=Path)
    ap.add_argument("--cohort-from", default=None,
                    help="Path to a text file with one subject_id per line. Default: complete cohort from bucket.")
    args = ap.parse_args()

    if args.cohort_from is not None:
        cohort = [ln.strip() for ln in Path(args.cohort_from).read_text().splitlines() if ln.strip()]
    else:
        cohort = bg.complete_cohort()
    print(f"cohort size: {len(cohort)} subjects")

    if args.mode == "primary":
        cmd_primary(cohort, args.results_dir, args.cache_dir)
    elif args.mode == "kappa":
        cmd_kappa(cohort, args.results_dir, args.cache_dir)
    elif args.mode == "fd":
        cmd_fd(cohort, args.results_dir, args.cache_dir)
    elif args.mode == "all":
        cmd_primary(cohort, args.results_dir, args.cache_dir)
        cmd_kappa(cohort, args.results_dir, args.cache_dir)
        cmd_fd(cohort, args.results_dir, args.cache_dir)


if __name__ == "__main__":
    main()
