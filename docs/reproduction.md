# Reproduction

## What this release supports

1. **Read results:** all report figures and aggregate result tables are included.
2. **Rebuild a summary:** `python scripts/rebuild_summary.py` recreates the forecast chart from the saved aggregate metrics. It does not refit models.
3. **Audit raw data:** obtain the source movement file separately, then run `python scripts/audit_raw.py data/wbjc.zip`. This reproduces the selected-SKU counts and descriptive statistics.
4. **Refit the response block:** `scripts/refit_response.py` implements the exact structured C1/C2 response objective on prepared, aligned temporal inputs. It does not construct those inputs or train B/m/E.

**The full historical feature/nuisance-training workflow is not yet packaged as a standalone raw-to-results command.** The original experiments depend on a sequence of intermediate feature contracts, baseline risk filters and saved temporal predictions. This release includes the portable response estimator, exact feature/fold contract, aggregate results and provenance; it does not claim that running the quick start recreates the published numerical results from raw data.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python scripts/rebuild_summary.py
python -m pytest
```

Open `notebooks/01_historical_results.ipynb` for a short tour. All paths are repository-relative. `generated/` is ignored by Git.

## Prepared response inputs

- Context CSV: one row per store-week, including `store`, `week`, `source_train_max`, `week_sin`, `week_cos` and the own-SKU columns listed in `response.py`.
- NPZ `actions`: observed log-price changes, shape `(contexts, 11)`.
- NPZ `residuals`: C1 uses `Y - B_past`; C2 uses `Y - m_past`.
- NPZ `mask`: observed-outcome mask with the same shape.
- C2 NPZ `e_plus`, `e_minus`: past-only predicted conditional means of the split actions. Splitting residualized actions is incorrect.
- Product order: `configs/model_contract.json`.
- Fit transforms only on the training contexts. Keep evaluation rows separate.

```bash
python scripts/refit_response.py data/context.csv data/c1_arrays.npz \
  --method C1 --output models/c1.joblib
python scripts/refit_response.py data/context.csv data/c2_arrays.npz \
  --method C2 --max-log-action 0.35667494393873245 \
  --output models/c2_moderate.joblib
```

These inputs are not bundled. The temporal assertion checks declared dates; it cannot prove that externally prepared features are leakage-free.

## Historical evaluation

- Price reference: previous week. Outcome-label delay: two weeks.
- Primary comparisons: Selection (244–295), Development (296–347), Later evaluation (348–399 planned; 348–375 observed complete subset).
- Main models: original asymmetric Direct/C1/C2 fits, not the moderate-only refits.
- Saved `final_holdout` names are legacy labels for a repeatedly inspected later period.
- Unit forecasts use historical conversion factors; raw log predictions and unit forecasts are different evaluations.
- Seeds, features and CatBoost parameters are recorded in the contract. Detailed scores are in `results/`.

## Verification performed for packaging

- Portable response predictions compared numerically with the original saved C1/C2 predictions on the cached later-period contexts.
- Synthetic tests check the anchored zero response and directional coefficient fitting.
- Published forecast chart rebuilt from included CSV.
- Raw audit run against the locally held source file; counts checked against the published EDA.
- Relative document/image links and tracked files checked; raw records and model binaries excluded.

## Scope of publication

No credentials, raw source files, row-level extracts or fitted binaries are included. Data access remains subject to the source terms. Aggregate figures/tables retain attribution. No code license is assigned in this draft; the author can choose one before broader reuse. Nothing has been pushed to a remote.

## Direct and optimization code

Install `pip install -e '.[train]'` to use `scripts/train_direct.py` on a prepared decision-time feature CSV. Columns and parameters follow the saved contract; the script trains on weeks 105–346 by default. It does not build lag features, reconstruct the baseline filter, or train nuisance models. Row ordering is explicitly chronological; exact historical bitwise reproduction is not promised.

`retail_pricing.optimization.optimize_one` exposes the previous-week-centered sanity check with a quantity prediction callback. Actual target prices are absent from its API. It uses the same log grid, cent rounding, cost floor and tie rule. Tests compare it with a known analytic profit optimum.
