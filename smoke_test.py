"""smoke_test.py — sanity check at primary values (FD=0.5, κ=0.10) on 5 subjects.

Confirms: bucket access, full pipeline, QC flags reasonable, cache disk usage.
"""
from __future__ import annotations

import sys
from pathlib import Path
from pprint import pformat

import build_graphs as bg


def _du_bytes(p: Path) -> int:
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def _human(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


def main() -> int:
    cfg = bg.Config(fd_threshold=0.5, kappa=0.10,
                    cache_dir=Path("cache"), results_dir=Path("results/smoke"))

    print("=" * 60); print("1) inspect_bucket()"); print("=" * 60)
    report = bg.inspect_bucket()
    cohort = report.pop("complete_cohort")
    print(pformat(report))
    print(f"complete_cohort: {len(cohort)} subjects (showing first 5: {cohort[:5]})")

    print("\n" + "=" * 60); print("2) build 5 subjects"); print("=" * 60)
    stats = bg.CacheStats()
    subset = cohort[:5]
    manifest_path = cfg.results_dir / "manifest_smoke.csv"
    manifest = bg.build_all(subset, cfg, manifest_path, stats=stats)
    print(manifest[["subject_id", "age", "age_group", "included",
                    "n_volumes_retained", "mean_FD", "density",
                    "kappa_final", "connected_at_requested_kappa"]].to_string(index=False))
    print(f"cache stats: L1 hits={stats.layer1_hits}/{stats.layer1_hits + stats.layer1_misses}, "
          f"L2 hits={stats.layer2_hits}/{stats.layer2_hits + stats.layer2_misses}, "
          f"L3 hits={stats.layer3_hits}/{stats.layer3_hits + stats.layer3_misses}")

    print("\n" + "=" * 60); print("3) qc_report"); print("=" * 60)
    flags = bg.qc_report(manifest, cfg.results_dir)
    print(pformat(flags))

    print("\n" + "=" * 60); print("4) sanity verdict"); print("=" * 60)
    broken = flags.get("density_age_flag") or flags.get("fd_age_flag")
    if flags["n_included"] == 0:
        print("LOOKS BROKEN — every subject excluded; check FD threshold / data integrity.")
    elif broken:
        print(f"LOOKS SUSPICIOUS — flagged: density_age_r={flags.get('density_age_r')}, "
              f"fd_age_r={flags.get('fd_age_r')}. (5 subjects is too small to trust; "
              f"re-check after primary run.)")
    else:
        print("LOOKS SANE — no QC flags raised on this 5-subject smoke.")

    print("\n" + "=" * 60); print("5) cache disk usage"); print("=" * 60)
    for layer in (1, 2, 3):
        size = _du_bytes(cfg.cache_dir / f"layer{layer}")
        print(f"  layer{layer}: {_human(size)}")
    return 0 if flags["n_included"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
