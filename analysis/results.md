## Cross-validated RMSE (5-fold)

| Configuration | complete rows | all train rows | test-like masked |
|---|---|---|---|
| Final model (submitted) | 3.022 | 3.066 | 3.298 |
| without missing-value offsets | 3.022 | 3.068 | 3.254 |
| without weighted least squares | 3.019 | 3.061 | 3.295 |
| LightGBM imputers instead of ridge imputers | 3.021 | 3.066 | 3.287 |
| linear model fit on complete rows only | 3.022 | 3.063 | 3.296 |
| +15% LightGBM blend | 3.021 | 3.065 | 3.290 |
| building x hour profiles instead of type x hour | 3.038 | 3.081 | 3.311 |
| without cooling term above 28C | 3.030 | 3.072 | 3.302 |
| without previous_usage | 3.985 | 4.009 | 4.110 |
| LightGBM on raw features (baseline) | 3.216 | 3.314 | 3.817 |

## Are values missing at random? Out-of-fold mean(actual - predicted) on rows naturally missing each input

- temperature: n=119, -0.43 ± 0.32 (SE); offset used: -0.25
- humidity: n=119, +0.26 ± 0.27 (SE); offset used: +0.25
- occupancy: n=122, +0.09 ± 0.42 (SE); offset used: +0.30
- previous_usage: n=114, +0.95 ± 0.34 (SE); offset used: +1.60

## Residuals of the final model (complete rows)

sd 3.022, skew 0.01, excess kurtosis 0.16

## Final model, test-like error by missing input

- temperature missing: 4.18 (n=628)
- humidity missing: 3.60 (n=606)
- occupancy missing: 4.51 (n=633)
- previous_usage missing: 4.93 (n=603)
- nothing missing: 3.04 (n=6230)
