# Developmental fMRI graph-metric pipeline

Final project for ECE 5260 / ORIE 5735 (Cornell Tech). Builds undirected
weighted functional-connectivity graphs from the Richardson et al. 2018
developmental fMRI cohort (`gs://results_050626/`), computes the locked
graph-metric panel, and tests age-related changes against the
explore-to-exploit hypothesis.

## Hypothesis

The Gopnik explore→exploit framing predicts that across development the
brain transitions from a broadly integrated, exploratory architecture to a
specialized, segregated, modular one. That gives a directional prediction
on each metric in the panel:

| Metric                          | Predicted sign with age | Why                                              |
|---------------------------------|------------------------|--------------------------------------------------|
| Clustering coefficient (Onnela) | +                      | More local segregation                           |
| Modularity Q                    | +                      | Sharper community structure                     |
| Characteristic path length L    | +                      | Less global integration                         |
| Small-worldness σ               | +                      | More efficient small-world organization         |
| Mean betweenness centrality     | −                      | Fewer cross-network hub-mediated shortcuts      |

The analysis tests these predictions across an age range of 3–39 years.

## Pipeline

```
raw 4D BOLD ──parcellate (Schaefer-100, MNI)──────────────────▶ ROI timeseries
            ──FD-scrub + Friston-24 + WM + CSF + DCT (OLS)────▶ clean TS    [Layer 1]
            ──Pearson r → Fisher z → zero negs/diag──────────-▶ z-matrix    [Layer 2]
            ──top-κ density + per-subject connectedness bump──▶ graphml     [Layer 3]
            ──C, Q, L, σ (Maslov–Sneppen ×100), betweenness───▶ metrics CSV [Layer 4]
            ──OLS metric ~ age + sex + mean_FD; FDR──────────-▶ regression tables
```

Cache layers are keyed independently. The κ sweep only re-touches Layers
3+4; the FD sweep invalidates Layers 1+2+4 but reuses Layer 3 across κ
within an FD.

## Files

Graph construction:

- `build_graphs.py` — Config, cache helpers, all pipeline stages, QC report,
  Methods-table dump.
- `smoke_test.py` — 5-subject graph-construction sanity check at primary values.
- `sweep_driver.py` — graph-build sweeps; `primary | kappa | fd | all`.
- `visualize_graph.py` — 3-panel single-subject graph viz (matrix, spring,
  anatomical-MNI).
- `gcs_loader.py` — bucket helper (patched for ADC auth + `.nii.gz` gunzip).

Metrics & analysis:

- `compute_metrics.py` — Layer 4 metric panel (C, Q, L, σ, betweenness),
  with `--smoke` mode and `primary | kappa | fd | all` runners.
- `analyze_age.py` — OLS age regression + Mann–Whitney child/adult tests
  per (FD, κ) condition; FDR across the metric panel; conditional
  permutation test.
- `make_tables.py` — writes the 6 writeup-ready tables (exclusion,
  connectedness, κ + FD sensitivity, prediction-vs-observed, limitations).
- `make_figures.py` — writes the 5 writeup-ready figures (cohort flow,
  group-avg z-matrix, age-regression panels, κ sensitivity, connectedness).
- `make_pipeline_figure.py` — Methods block-diagram figure + methods table.
- `run_finish.sh` — chains FD metrics → all stats → tables → figures.

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


## Results — hypothesis test

### Primary (FD=0.5, κ=0.10, n=108)

OLS regression `metric ~ age + sex + mean_retained_FD`, FDR across the 5-metric panel:

| Metric              | β_age      | p          | FDR-q      | predicted sign | observed sign | theory consistent |
|---------------------|------------|------------|------------|----------------|---------------|-------------------|
| Clustering (Onnela) | **+0.0016**| **4.7e-5** | **2.3e-4** | +              | +             | ✓                 |
| Modularity Q        | +0.0011    | 0.32       | 0.79       | +              | +             | ✓ (n.s.)          |
| Path length L       | −0.0004    | 0.93       | 0.93       | +              | −             | ✗                 |
| Small-worldness σ   | +0.0043    | 0.50       | 0.83       | +              | +             | ✓ (n.s.)          |
| Mean betweenness    | −4.4e-6    | 0.93       | 0.93       | −              | −             | ✓ (n.s.)          |

**4 of 5 metrics are sign-consistent with the explore→exploit prediction;
clustering coefficient is significant after FDR correction.** Group test
(child n=77 vs adult n=31) confirms: clustering_w child mean 0.226 < adult
0.260, Mann–Whitney q=2.1e-5, rank-biserial r=+0.57.

### κ sensitivity (FD=0.5)

`results/figures/fig6_kappa_sensitivity.png` shows β_age vs κ per metric.

| Metric              | κ=0.05 | κ=0.10 | κ=0.15 | κ=0.20  | κ=0.25  |
|---------------------|--------|--------|--------|---------|---------|
| Clustering (FDR-q)  | 6e-4   | 2e-4   | 2e-4   | 3e-4    | 4e-4    |
| Modularity Q        | 0.92   | 0.79   | 0.18   | **0.029** | **0.019** |
| Small-worldness σ   | 0.94   | 0.83   | 0.19   | **0.047** | **0.040** |

**Clustering is robust across the entire κ grid.** Modularity and
small-worldness emerge as significant only at κ ∈ {0.20, 0.25} — the
primary κ=0.10 is too sparse to detect them. Path length and betweenness
never reach significance.

### FD sensitivity (κ=0.10)

| Metric              | FD=0.3 (n=81)  | FD=0.5 (n=108) | FD=0.9 (n=119) | sign consistent |
|---------------------|----------------|----------------|----------------|-----------------|
| Clustering          | β=+0.0015 q=0.003 | β=+0.0016 q=2e-4 | β=+0.0014 q=7e-4 | ✓ all sig.       |
| Modularity Q        | +0.0019 (n.s.)  | +0.0011 (n.s.)  | +0.0012 (n.s.)  | ✓                |
| Small-worldness σ   | +0.0095 (n.s.)  | +0.0043 (n.s.)  | +0.0059 (n.s.)  | ✓                |

**Clustering effect is robust across all three FD thresholds. FD=0.3 has
a residual FD-vs-age confound (r=+0.286), FD=0.5 and FD=0.9 are clean.**

## Writeup data sources

For each section of the report, cite/embed the listed files directly. No
further analysis is needed.

| Writeup section | File(s) to use |
|---|---|
| **Methods — parameter table** | `results/methods_table.csv` |
| **Methods — pipeline figure** | `results/figures/fig0_pipeline.png` |
| **Methods — cohort & exclusions** | `results/exclusion_table.csv` + `results/figures/fig1_cohort_flow.png` |
| **Methods — connectedness footnote** | `results/connectedness_report.csv`, `results/connectedness_summary_by_group.csv`, `results/figures/fig_connectedness.png` |
| **Results — group-average matrices** | `results/figures/fig2_group_avg_zmatrix.png` |
| **Results — primary regression** | `results/age_regression_FD0.5_kappa0.1.csv` + `results/group_comparison_FD0.5_kappa0.1.csv` + `results/figures/fig3_age_regression_panels.png` |
| **Results — κ sensitivity** | `results/kappa_sensitivity_table.csv` + `results/figures/fig6_kappa_sensitivity.png` |
| **Results — FD sensitivity** | `results/fd_sensitivity_table.csv` |
| **Discussion — theory comparison** | `results/prediction_vs_observed.csv` |
| **Discussion — limitations (numbers)** | `results/limitations_numbers.csv` |
| **Appendix — per-(FD, κ) regression detail** | `results/age_regression_FD*_kappa*.csv` (7 files) + `results/group_comparison_FD*_kappa*.csv` (7 files) |

## Outputs

- 17 writeup-ready files under `results/` (5 figures + 6 tables + 14
  per-condition CSVs).
- 7 build manifests at `results/manifest_FD*_kappa*.csv`.
- 7 QC PNGs at `results/qc_FD*_kappa*/qc_FD*_kappa*.png`.
- 108 connected graphs at `cache/layer3/*.graphml` and per-subject metric
  CSVs at `cache/layer4/metrics_FD*_kappa*.csv`.
- Example single-subject graph viz at `results/viz/graph_sub-pixar002.png`.

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
6. **κ-density choice matters for which predictions you can detect.** At
   primary κ=0.10 only clustering reaches significance, but modularity Q
   and small-worldness σ become significant at κ ∈ {0.20, 0.25}. The
   primary κ=0.10 *under-detects* segregation effects; report this in the
   Discussion.
7. **Connected-graph caveat.** Per-subject κ-bumping means 78 of 108
   primary-included subjects effectively report metrics at `kappa_final`
   (mean ≈ 0.17), not the requested 0.10. Each manifest records both.

## How to run

```bash
gcloud auth application-default login                                       # one-time ADC
pip install -r requirements.txt                                             # gcsfs, nibabel, nilearn, google-cloud-storage
pip install networkx scipy pandas matplotlib statsmodels                    # not yet in requirements.txt

# graph construction
python3 smoke_test.py                                                       # 5 subjects, ~10s
python3 sweep_driver.py primary                                             # FD=0.5, κ=0.10
python3 sweep_driver.py kappa                                               # κ ∈ {0.05..0.25}
python3 sweep_driver.py fd                                                  # FD ∈ {0.3, 0.5, 0.9}

# Phase-0 methods artifacts (cheap, idempotent)
python3 make_pipeline_figure.py                                             # fig0 + methods_table.csv

# graph metrics (Layer 4); ~12 min for primary, ~50 min for full sweep
python3 compute_metrics.py --smoke                                          # 5 subjects, ~30s
python3 compute_metrics.py primary                                          # FD=0.5, κ=0.10
python3 compute_metrics.py kappa                                            # κ sweep
python3 compute_metrics.py fd                                               # FD sweep
# OR: ./run_finish.sh   # FD-metrics → all stats → tables → figures, autopilot

# stats + tables + figures (cheap, idempotent)
python3 analyze_age.py all                                                  # writes group_comparison_*.csv + age_regression_*.csv
python3 make_tables.py                                                      # writes the 6 writeup-ready tables
python3 make_figures.py                                                     # writes the 5 writeup-ready figures
python3 visualize_graph.py sub-pixar002                                     # optional: single-subject graph PNG
```
