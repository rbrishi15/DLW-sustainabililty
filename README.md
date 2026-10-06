# Smart Campus Energy Forecasting — NTU DLW Datathon 2026 (Track 1, 2nd Place)

Forecasting hourly energy use for 12 campus buildings with an interpretable linear model that handles missing data robustly.

**2nd place in Track 1** at one of NTU's largest datathons (140+ teams overall) · public leaderboard **RMSE 3.29**

## The problem

Predict `energy_usage` for a building-hour from the building, time (hour, day of week, month), weather
(temperature, humidity), occupancy and the previous period's usage. The metric is RMSE.

| | Train | Test |
|---|---|---|
| Rows | 8,000 (labelled) | 3,000 |
| Buildings / types | 12 / 8 | same |
| Missing values per numeric column | ~1.5% | **~7%** (4–5× more) |

Submissions were run by the organizers in an offline sandbox: a prediction notebook loads the trained model and
predicts test rows that arrive shuffled and without IDs.

## Approach

The data turned out to be close to **additive and linear**, so the final model is a ridge regression on a
carefully designed feature matrix, with model-based handling of missing inputs.

1. **Input cleaning:** types are coerced, unknown buildings map to a known building of the same type, and numeric
   inputs are clipped to a widened training range so the linear model cannot extrapolate wildly.
2. **Missing-value imputation:** each missing input is replaced by its expected value given the rest of the row.
   32 small ridge models do this, one per (missing column × combination of other columns present), trained on
   train and test feature rows (inputs only, never labels).
3. **Additive design (294 features):** an hourly usage profile per building type; base load and weekend shift
   per building; day-of-week and seasonal terms; per-building occupancy and humidity slopes; per-type
   temperature slopes with an extra cooling term above 28 °C; and `previous_usage`.
4. **Ridge regression** (α = 2), trained with weighted least squares.
5. **Missing-not-at-random offsets:** small constants added when an input was missing (e.g. +1.6 when
   `previous_usage` is missing), because values tend to go missing in unusual, higher-load situations.
6. **Clipping** predictions to a plausible range.

**Validation that mirrors the test:** every cross-validation fold was scored twice, as-is and after re-masking the
validation rows with the **test set's own missing-value patterns**. That test-like score (3.298) predicted the
public leaderboard (3.292) within 0.01.

## Results

5-fold cross-validated RMSE (lower is better). Each ablation changes one decision of the final model.

| Configuration | Complete rows | All train rows | Test-like missingness |
|---|---|---|---|
| **Final model (submitted)** | **3.022** | **3.066** | **3.298** |
| LightGBM on raw features (baseline) | 3.216 | 3.314 | 3.817 |
| without `previous_usage` | 3.985 | 4.009 | 4.110 |
| building × hour profiles instead of type × hour | 3.038 | 3.081 | 3.311 |
| without cooling term above 28 °C | 3.030 | 3.072 | 3.302 |
| without missing-value offsets | 3.022 | 3.068 | 3.254 |
| without weighted least squares | 3.019 | 3.061 | 3.295 |
| LightGBM imputers instead of ridge imputers | 3.021 | 3.066 | 3.287 |
| linear model fit on complete rows only | 3.022 | 3.063 | 3.296 |
| +15% LightGBM blend | 3.021 | 3.065 | 3.290 |

Takeaways:

- **The linear model beats gradient boosting**: by 6% on complete rows and **14% under test-like missingness**
  (3.82 → 3.30), where boosting learns little about missing inputs from train's 1.5% missing rate.
- **`previous_usage` is the key input** (removing it costs ~1 RMSE). Shared **type × hour** profiles generalise
  better than per-building profiles, and the cooling term adds a small but consistent gain.
- **The model sits at the noise floor**: residuals on complete rows are almost perfectly Gaussian
  (sd 3.02, skew 0.01, excess kurtosis 0.16).
- **Most remaining error comes from missing inputs**: test-like RMSE is 3.04 on complete rows, but 4.93 when
  `previous_usage` is missing and 4.51 when occupancy is missing.

### What the model learned

| Input | Effect (from `model.pkl`) |
|---|---|
| `previous_usage` | ≈ 0.34 per unit: about a third of last period's usage carries over |
| Occupancy | +4.9 (residences) to +13.3 (lecture hall) per 100 people |
| Temperature | +0.8 to +2.2 per °C depending on building type, plus +0.39 per °C above 28 °C |
| Weekend | −1.4 to −2.6 for business, lecture and admin buildings; +0.3 to +1.2 for residences, library, labs and sports |
| Hour of day | small per-type profile (1.5–5 units): occupancy already explains most of the daily cycle |

## Repository

| File | Contents |
|---|---|
| `track1_energy_notebook.ipynb` | The submitted notebook: EDA, validation and training (run only when `train.csv` is present), then prediction. Its code is unchanged; its explanatory notes were updated after the datathon to match the submitted configuration |
| `energy_model.py` | The model code (same as the notebook's model cell), for easier reading |
| `model.pkl` | The submitted trained model (`(model, feature_names)`, loadable with `joblib.load`) |
| `submission.csv` | The submitted test predictions |
| `analysis/ablations.py` | Reproduces every number in the Results section |
| `analysis/results.md` | Output of `analysis/ablations.py` |
| `requirements.txt` | Library versions of the evaluation platform (Python 3.12) |

## Reproducing

The datathon data is not redistributed here. Place the organizers' `train.csv` and `test.csv` in the repository root.

```bash
pip install -r requirements.txt
python analysis/ablations.py            # cross-validation and ablations (a few minutes)
jupyter nbconvert --to notebook --execute track1_energy_notebook.ipynb   # retrains model.pkl and predicts test.csv
```

Running the notebook with `train.csv` present retrains and overwrites `model.pkl`. Without `train.csv` it only
loads the model and writes `predictions.csv`, which is how the evaluation platform runs it.

## Notes and limitations

- The `previous_usage` offset (+1.6) is larger than the out-of-fold estimate from training rows where it is
  naturally missing (+0.95 ± 0.34); the other three offsets are within noise. The offsets are not visible in the
  test-like CV, because the masking there is random.
- Weighted least squares and the LightGBM components are neutral in cross-validation. The LightGBM models are
  still stored in `model.pkl` but have zero weight in the final predictions.
- The notebook's explanatory notes were revised after the datathon so they describe the submitted configuration
  (the `FINAL` and `EXTRA` settings in its validation cell, also used by `analysis/ablations.py`). No code changed:
  the notebook still reproduces `submission.csv` exactly.

## Acknowledgements

Thanks to the organizers of the NTU DLW Datathon 2026 for the problem, data and evaluation platform.
