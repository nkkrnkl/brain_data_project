# Developmental fMRI graph-metric pipeline

Final project for ECE 5260 / ORIE 5735 (Cornell Tech). Builds undirected
weighted functional-connectivity graphs from the Richardson et al. 2018
developmental fMRI cohort (`gs://results_050626/`) and runs the locked
sensitivity sweeps over edge density (κ) and FD scrubbing threshold.

## Pipeline

```
raw 4D BOLD ──parcellate (Schaefer-100, MNI)──▶ ROI timeseries
            ──FD-scrub + Friston-24 + WM + CSF + DCT (OLS)──▶ clean TS  [Layer 1]
            ──Pearson r → Fisher z → zero negs/diag──────────▶ z-matrix [Layer 2]
            ──top-κ density + per-subject connectedness bump─▶ graphml  [Layer 3]
```

Cache layers are keyed independently. The κ sweep only re-touches Layer 3;
the FD sweep invalidates Layers 1+2 but reuses Layer 3 across κ within an FD.

## Files

- `build_graphs.py` — Config, cache helpers, all pipeline stages, QC report.
- `smoke_test.py` — 5-subject sanity check at primary values.
- `sweep_driver.py` — `primary | kappa | fd | all` modes, with cache-hit
  assertions enforcing the layering invariants.
- `visualize_graph.py` — 3-panel single-subject graph viz (matrix, spring,
  anatomical-MNI).
- `gcs_loader.py` — bucket helper (patched for ADC auth + `.nii.gz` gunzip).

## Cohort

- 155 subjects in `participants.tsv`, but only **131 have a preprocessed BOLD
  file in the bucket** — the usable cohort. The 24 missing-BOLD IDs are a
  data-availability exclusion separate from FD-retention.

## Results

### Primary (FD=0.5, κ=0.10)

- **108 included / 131** (18 at FD-retention floor, 5 at κ_max-disconnect).
- 78 of 108 needed κ-bump for connectedness (mean κ_final ≈ 0.20).
- QC: density-vs-age **r=+0.006** ✓, FD-vs-age **r=+0.020** ✓ (both well
  below the |r|>0.2 flag).

### κ sweep — n_included stable at 108 across all κ; cache layering proven

| κ    | density-vs-age r | FD-vs-age r | n_kappa_bumped | Layer 2 hits/miss |
|------|------------------|-------------|----------------|-------------------|
| 0.05 | +0.006           | +0.020      | 108 (all)      | 113 / 0           |
| 0.10 | +0.006           | +0.020      | 78             | 113 / 0           |
| 0.15 | -0.018           | +0.020      | 41             | 113 / 0           |
| 0.20 | -0.034           | +0.020      | 24             | 113 / 0           |
| 0.25 | +0.004           | +0.020      | 16             | 113 / 0           |

No flags fire across the entire κ grid. **κ=0.05 is too sparse** (universal
κ-bumping defeats the purpose); the analysis is robust at κ ∈ {0.10–0.25}.

### FD sweep — headline finding for the writeup

| FD  | n_included | density-vs-age r | FD-vs-age r | flags                |
|-----|------------|------------------|-------------|----------------------|
| 0.3 | 81         | -0.063           | **+0.286**  | ⚠ FD-vs-age flagged  |
| 0.5 | 108        | +0.006           | +0.020      | clean                |
| 0.9 | 119        | +0.006           | -0.165      | clean                |

**FD=0.3 fails the residual-motion-confound check.** At the strict threshold,
young children are disproportionately scrubbed; the included set has more
low-motion adults than low-motion children, so mean retained FD correlates
with age. **This justifies FD=0.5 (Power criterion) as the spec's primary
threshold**; FD=0.9 is a defensible lenient alternative.

### Cache-layering correctness

- **κ sweep:** zero L1/L2 misses across all 5 κ values — Layer 2 cache fully
  reused as designed.
- **FD sweep:** primary FD=0.5 was 100% L1+L2 hits (cached); FD=0.3 and
  FD=0.9 had fresh L1+L2 misses for new keys, which is the correct
  invalidation.
- All cache-hit assertions in `sweep_driver.py` passed.


## Outputs

- 7 manifest CSVs at `results/manifest_FD{0.3,0.5,0.9}_kappa{0.05..0.25}.csv`.
- 7 QC PNGs at `results/qc_FD*_kappa*/qc_FD*_kappa*.png`.
- 108 connected graphs at `cache/layer3/sub-pixar*_FD0.5_kappa0.1_*.graphml`.
- Example single-subject viz at `results/viz/graph_sub-pixar002.png`.

## Next steps

1. **Graph metrics** — compute degree, clustering, characteristic path length,
   global/local efficiency, modularity, and small-worldness σ on each cached
   graph. Small-worldness is where `maslov_sneppen_null` (already in
   `build_graphs.py`) finally gets called: σ = (C/C_rand) / (L/L_rand) over
   the 100 rewired nulls. Add a Layer 4 cache keyed on
   `(subject, FD, κ, null_hash)` so the κ sweep stays cheap.
2. **Age regression + sensitivity table** — per metric: linear regression on
   age controlling for sex and mean retained FD, repeated across the κ and
   FD grid, with FDR correction across metrics. One summary table per sweep
   showing whether the age-effect direction and significance are stable.
3. **Writeup figures** — for 2–3 metrics with stable age effects, plot age vs
   metric at primary settings, with the κ-grid forming a sensitivity envelope.
   Methods and cohort sections are already documented above.

Open methodological question for #1: metrics are computed on each subject's
*connected* graph (per the spec's per-subject κ-bump rule), so 78 of 108
subjects effectively report at `kappa_final`, not `kappa_requested`. This is
locked by the spec but worth flagging in the writeup.

## Caveats for the writeup

1. **Cohort = 131 BOLD-having, not 155.** Report as a separate exclusion stage
   from FD scrubbing.
2. **κ-bumping is pervasive at κ=0.10** (78 of 108) — likely driven by 4mm
   BOLD voxel size + Schaefer-100 atlas resampling leaving some parcels
   low-SNR. Per-subject κ_final is recorded in every manifest.
3. **5 subjects hit κ_max=0.5** even with bumping; tracked as
   `exclusion_reason=kappa_max_disconnected`.
4. **BOLD header TR=1.0s ≠ spec TR=2.0s.** Pipeline trusts the spec; warning
   fires per subject and is documented in `Config`.
5. **Sensitivity findings:** primary results are robust to κ ∈ {0.10–0.25}
   and to FD ∈ {0.5, 0.9}; **avoid FD=0.3** because it introduces residual
   motion-age coupling in the included set.

## How to run

```bash
gcloud auth application-default login           # one-time ADC
pip install -r requirements.txt                 # gcsfs, nibabel, nilearn, google-cloud-storage
pip install networkx scipy pandas matplotlib    # not yet in requirements.txt

python3 smoke_test.py                           # 5 subjects, ~10s
python3 sweep_driver.py primary                 # FD=0.5, κ=0.10
python3 sweep_driver.py kappa                   # κ ∈ {0.05..0.25}
python3 sweep_driver.py fd                      # FD ∈ {0.3, 0.5, 0.9}
python3 sweep_driver.py all                     # everything, idempotent
python3 visualize_graph.py sub-pixar002         # single-subject graph PNG
```
