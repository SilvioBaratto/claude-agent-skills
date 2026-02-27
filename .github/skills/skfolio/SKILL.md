---
name: skfolio
description: |
  Load proactively whenever the user works with skfolio or portfolio optimization. Do not wait to be asked; apply this skill automatically. Covers optimization models (MeanRisk, HRP, HERC, NCO, RiskBudgeting, StackingOptimization), prior estimators (BlackLitterman, FactorModel, EntropyPooling, OpinionPooling, SyntheticData), covariance/expected return estimation, clustering, pre-selection, model selection (WalkForward, CombinatorialPurgedCV), hyperparameter tuning, metadata routing, and data preparation using the scikit-learn API.
---

# skfolio Portfolio Optimization Skill

Expert guidance for using skfolio — a portfolio optimization and risk management framework built on top of scikit-learn. All estimators follow the scikit-learn API (`fit`, `predict`, `get_params`, pipelines, cross-validation).

## Documentation Reference

When in doubt, **always consult** the official documentation:

| Topic | URL |
|-------|-----|
| API Reference | https://skfolio.org/api.html |
| Optimization | https://skfolio.org/user_guide/optimization.html |
| Portfolio | https://skfolio.org/user_guide/portfolio.html |
| Population | https://skfolio.org/user_guide/population.html |
| Prior Estimators | https://skfolio.org/user_guide/prior.html |
| Expected Returns | https://skfolio.org/user_guide/expected_returns.html |
| Covariance | https://skfolio.org/user_guide/covariance.html |
| Distance | https://skfolio.org/user_guide/distance.html |
| Clustering | https://skfolio.org/user_guide/cluster.html |
| Uncertainty Sets | https://skfolio.org/user_guide/uncertainty_set.html |
| Pre-Selection | https://skfolio.org/user_guide/pre_selection.html |
| Model Selection | https://skfolio.org/user_guide/model_selection.html |
| Hyperparameter Tuning | https://skfolio.org/user_guide/hyper_parameters_tuning.html |
| Metadata Routing | https://skfolio.org/user_guide/metadata_routing.html |
| Data Preparation | https://skfolio.org/user_guide/data_preparation.html |
| Examples Gallery | https://skfolio.org/auto_examples/index.html |

## Architecture Overview

```
skfolio/
├── optimization/        # Portfolio optimization models
├── prior/               # Prior return distribution estimators
├── moments/             # Expected returns & covariance estimators
├── distance/            # Codependence & distance estimators
├── cluster/             # Hierarchical clustering
├── uncertainty_set/     # Mu & covariance uncertainty sets
├── pre_selection/       # Asset filtering transformers
├── model_selection/     # Cross-validation & backtesting
├── metrics/             # Scoring functions
├── preprocessing/       # Data transformation (prices_to_returns)
├── distribution/        # Copulas & univariate distributions
├── datasets/            # Sample datasets
├── portfolio/           # Portfolio & MultiPeriodPortfolio
├── population/          # Population of portfolios
└── measures/            # Risk/performance measure enums
```

## Complete API Reference

### skfolio.optimization

```python
from skfolio.optimization import (
    # Naive
    EqualWeighted,
    InverseVolatility,
    Random,
    # Convex
    MeanRisk,
    BenchmarkTracker,
    RiskBudgeting,
    MaximumDiversification,
    DistributionallyRobustCVaR,
    # Clustering
    HierarchicalRiskParity,
    HierarchicalEqualRiskContribution,
    NestedClustersOptimization,
    # Ensemble
    StackingOptimization,
    # Enums
    ObjectiveFunction,
)
```

### skfolio.prior

```python
from skfolio.prior import (
    EmpiricalPrior,
    BlackLitterman,
    FactorModel,
    SyntheticData,
    EntropyPooling,
    OpinionPooling,
    LoadingMatrixRegression,
)
```

### skfolio.moments

```python
from skfolio.moments import (
    # Expected returns
    EmpiricalMu,
    EWMu,
    ShrunkMu,
    EquilibriumMu,
    # Covariance
    EmpiricalCovariance,
    EWCovariance,
    LedoitWolf,
    OAS,
    ShrunkCovariance,
    DenoiseCovariance,
    DetoneCovariance,
    GerberCovariance,
    GraphicalLassoCV,
    ImpliedCovariance,
)
```

### skfolio.distance

```python
from skfolio.distance import (
    PearsonDistance,
    KendallDistance,
    SpearmanDistance,
    CovarianceDistance,
    DistanceCorrelation,
    MutualInformation,
)
```

### skfolio.cluster

```python
from skfolio.cluster import HierarchicalClustering, LinkageMethod
```

### skfolio.uncertainty_set

```python
from skfolio.uncertainty_set import (
    EmpiricalMuUncertaintySet,
    EmpiricalCovarianceUncertaintySet,
    BootstrapMuUncertaintySet,
    BootstrapCovarianceUncertaintySet,
)
```

### skfolio.pre_selection

```python
from skfolio.pre_selection import (
    DropCorrelated,
    DropZeroVariance,
    SelectKExtremes,
    SelectNonDominated,
    SelectComplete,
    SelectNonExpiring,
)
```

### skfolio.model_selection

```python
from skfolio.model_selection import (
    WalkForward,
    CombinatorialPurgedCV,
    MultipleRandomizedCV,
    cross_val_predict,
    optimal_folds_number,
)
```

### skfolio.metrics

```python
from skfolio.metrics import make_scorer
```

### skfolio.preprocessing

```python
from skfolio.preprocessing import prices_to_returns
```

### skfolio.distribution

```python
from skfolio.distribution import (
    # Copulas
    VineCopula,
    GaussianCopula,
    StudentTCopula,
    ClaytonCopula,
    GumbelCopula,
    JoeCopula,
    IndependentCopula,
    # Univariate
    Gaussian,
    StudentT,
    JohnsonSU,
    NormalInverseGaussian,
)
```

### skfolio.datasets

```python
from skfolio.datasets import (
    load_sp500_dataset,        # 20 S&P 500 assets
    load_sp500_index,          # S&P 500 benchmark
    load_factors_dataset,      # 5 factor ETFs
    load_ftse100_dataset,      # 64 FTSE 100 assets
    load_nasdaq_dataset,       # 1,455 NASDAQ assets
    load_sp500_implied_vol_dataset,  # Implied volatility surface
)
```

### skfolio.measures (enums)

```python
from skfolio import (
    RiskMeasure,
    ExtraRiskMeasure,
    PerfMeasure,
    RatioMeasure,
    ObjectiveFunction,
)
```

---

## Optimization Models

### MeanRisk

The primary convex optimization model. Solves four objective functions over any convex risk measure.

```python
model = MeanRisk(
    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,  # max Sharpe
    risk_measure=RiskMeasure.CVAR,
    min_weights=0.0,           # long-only
    max_weights=0.15,          # max 15% per asset
    budget=1.0,                # fully invested
    prior_estimator=EmpiricalPrior(),
    mu_uncertainty_set_estimator=None,
    covariance_uncertainty_set_estimator=None,
    l1_coef=0.0,               # L1 regularization
    l2_coef=0.0,               # L2 regularization
    transaction_costs=0.0,
    management_fees=0.0,
    groups=None,               # asset groups dict
    linear_constraints=None,   # group constraints
    left_inequality=None,      # Aw <= b
    right_inequality=None,
)
```

**ObjectiveFunction enum:**

| Value | Description |
|-------|-------------|
| `MINIMIZE_RISK` | Minimize the risk measure |
| `MAXIMIZE_RETURN` | Maximize expected return |
| `MAXIMIZE_UTILITY` | Maximize return - risk_aversion * risk |
| `MAXIMIZE_RATIO` | Maximize return/risk (e.g., Sharpe ratio) |

**RiskMeasure enum (convex, for MeanRisk):**

| Value | Description |
|-------|-------------|
| `VARIANCE` | Portfolio variance |
| `SEMI_VARIANCE` | Downside variance |
| `STANDARD_DEVIATION` | Portfolio volatility |
| `SEMI_DEVIATION` | Downside deviation |
| `MEAN_ABSOLUTE_DEVIATION` | Mean absolute deviation |
| `FIRST_LOWER_PARTIAL_MOMENT` | First lower partial moment |
| `CVAR` | Conditional Value at Risk |
| `EVAR` | Entropic Value at Risk |
| `WORST_REALIZATION` | Worst-case scenario |
| `CDAR` | Conditional Drawdown at Risk |
| `MAXIMUM_DRAWDOWN` | Maximum drawdown |
| `AVERAGE_DRAWDOWN` | Average drawdown |
| `EDAR` | Entropic Drawdown at Risk |
| `ULCER_INDEX` | Ulcer index |
| `GINI_MEAN_DIFFERENCE` | Gini mean difference |

**ExtraRiskMeasure enum (non-convex, for scoring only):**

| Value | Description |
|-------|-------------|
| `VALUE_AT_RISK` | VaR |
| `DRAWDOWN_AT_RISK` | Drawdown VaR |
| `ENTROPIC_RISK_MEASURE` | Entropic risk |
| `FOURTH_CENTRAL_MOMENT` | Kurtosis proxy |
| `FOURTH_LOWER_PARTIAL_MOMENT` | Downside kurtosis |
| `SKEW` | Portfolio skewness |
| `KURTOSIS` | Portfolio kurtosis |

**RatioMeasure enum:**

| Value | Description |
|-------|-------------|
| `SHARPE_RATIO` | Return / StdDev |
| `SORTINO_RATIO` | Return / Downside deviation |
| `CALMAR_RATIO` | Return / Max drawdown |
| `CVAR_RATIO` | Return / CVaR |

**PerfMeasure enum:**

| Value | Description |
|-------|-------------|
| `MEAN` | Mean return |
| `ANNUALIZED_MEAN` | Annualized mean |

### RiskBudgeting

Allocates risk budget across assets.

```python
model = RiskBudgeting(
    risk_measure=RiskMeasure.CVAR,
    risk_budget=None,          # None = equal risk contribution
    prior_estimator=EmpiricalPrior(),
    min_weights=0.0,
    max_weights=1.0,
)
```

### MaximumDiversification

Maximizes the diversification ratio.

```python
model = MaximumDiversification(
    prior_estimator=EmpiricalPrior(),
    min_weights=0.0,
    max_weights=1.0,
)
```

### DistributionallyRobustCVaR

Minimizes worst-case CVaR within a Wasserstein ball.

```python
model = DistributionallyRobustCVaR(
    risk_aversion=1.0,
    wasserstein_ball_radius=0.02,
    prior_estimator=EmpiricalPrior(),
)
```

### HierarchicalRiskParity (HRP)

Uses hierarchical clustering with recursive bisection for risk allocation.

```python
model = HierarchicalRiskParity(
    risk_measure=RiskMeasure.CVAR,
    prior_estimator=EmpiricalPrior(),
    distance_estimator=PearsonDistance(),
    hierarchical_clustering_estimator=HierarchicalClustering(),
)
```

### HierarchicalEqualRiskContribution (HERC)

Top-down recursive division of the dendrogram for equal risk contribution.

```python
model = HierarchicalEqualRiskContribution(
    risk_measure=RiskMeasure.CDAR,
    prior_estimator=EmpiricalPrior(),
    distance_estimator=PearsonDistance(),
    hierarchical_clustering_estimator=HierarchicalClustering(),
)
```

### NestedClustersOptimization (NCO)

Combines inner and outer optimization via clustering.

```python
model = NestedClustersOptimization(
    inner_estimator=MeanRisk(),
    outer_estimator=MeanRisk(),
    distance_estimator=PearsonDistance(),
    hierarchical_clustering_estimator=HierarchicalClustering(),
)
```

### StackingOptimization

Ensemble of multiple optimizers combined via a final allocator.

```python
model = StackingOptimization(
    estimators=[
        ("hrp", HierarchicalRiskParity()),
        ("meanrisk", MeanRisk()),
    ],
    final_estimator=MeanRisk(),
)
```

### BenchmarkTracker

Minimizes tracking error against benchmark returns.

```python
model = BenchmarkTracker(
    tracking_error_target=0.01,  # 1% tracking error
)
# fit(X, y=benchmark_returns)
```

### Naive Models

```python
model = EqualWeighted()       # 1/N allocation
model = InverseVolatility()   # inverse-vol weighting
model = Random(n_portfolios=100)  # random portfolios
```

---

## Prior Estimators

All priors produce a `ReturnDistribution` containing `mu`, `covariance`, `returns`, `sample_weight`, and `cholesky`.

### EmpiricalPrior

Estimates distribution from historical data using pluggable mu/covariance estimators.

```python
prior = EmpiricalPrior(
    mu_estimator=ShrunkMu(),
    covariance_estimator=LedoitWolf(),
    is_log_normal=False,
    investment_horizon=None,
)
```

### BlackLitterman

Bayesian model combining market equilibrium prior with analyst views.

```python
prior = BlackLitterman(
    views=[
        "AAPL == 0.10",           # absolute: AAPL returns 10%
        "MSFT - GOOG == 0.03",    # relative: MSFT outperforms GOOG by 3%
    ],
    tau=0.05,
    prior_estimator=EmpiricalPrior(mu_estimator=EquilibriumMu()),
)
```

**View syntax:**
- Absolute: `"TICKER == value"` or `"TICKER >= value"`
- Relative: `"TICKER1 - TICKER2 == value"`

### FactorModel

Reduces dimensionality by explaining asset returns through common factors.

```python
prior = FactorModel(
    loading_matrix_estimator=LoadingMatrixRegression(),
    factor_prior_estimator=EmpiricalPrior(),
)
# fit(X, y=factor_returns) — X = asset returns, y = factor returns
```

Chainable: `FactorModel(factor_prior_estimator=BlackLitterman(views=[...]))` for a Black-Litterman Factor Model.

### SyntheticData

Generates synthetic scenarios from a fitted distribution (e.g., vine copula).

```python
prior = SyntheticData(
    distribution_estimator=VineCopula(),
    n_samples=10_000,
    sample_args=None,  # pass dict(conditioning={...}) for stress tests
)
```

### EntropyPooling

Adjusts baseline probabilities to incorporate views by minimizing KL divergence.

```python
prior = EntropyPooling(
    mean_views=["AAPL == 0.0005", "JPM >= BAC"],
    variance_views=["BAC == prior(BAC) * 4"],
    correlation_views=["(BAC,JPM) == 0.80"],
    skew_views=["BAC == -0.05"],
    cvar_views=["GE == 0.08"],
    cvar_beta=0.95,
    groups={"Financials": ["BAC", "JPM"], "Tech": ["AAPL", "MSFT"]},
    prior_estimator=EmpiricalPrior(),
)
```

**View types:**

| Type | Syntax | Example |
|------|--------|---------|
| Mean | `"TICKER == value"` | `"JPM == -0.002"` |
| Mean relative | `"TICKER1 >= TICKER2"` | `"PG >= LLY"` |
| Mean vs prior | `"TICKER >= prior(TICKER) * factor"` | `"BAC >= prior(BAC) * 1.2"` |
| Variance | `"TICKER == prior(TICKER) * factor"` | `"BAC == prior(BAC) * 4"` |
| Correlation | `"(T1,T2) == value"` | `"(BAC,JPM) == 0.80"` |
| Correlation vs prior | `"(T1,T2) <= prior(T1,T2) * factor"` | `"(BAC,JNJ) <= prior(BAC,JNJ) * 0.5"` |
| Skew | `"TICKER == value"` | `"BAC == -0.05"` |
| CVaR | `"TICKER == value"` | `"GE == 0.08"` |
| Group mean | `"Group1 == factor * Group2"` | `"Financials == 2 * Growth"` |

### OpinionPooling

Combines multiple expert distributions into consensus.

```python
prior = OpinionPooling(
    estimators=[
        ("expert_1", EntropyPooling(mean_views=["AAPL == 0.001"])),
        ("expert_2", EntropyPooling(mean_views=["AAPL == -0.001"])),
    ],
    opinion_probabilities=[0.4, 0.5],  # remaining 0.1 → base prior
    prior_estimator=EmpiricalPrior(),
)
```

---

## Moment Estimators

### Expected Returns

| Estimator | Description | Key Parameters |
|-----------|-------------|----------------|
| `EmpiricalMu` | Historical mean returns | — |
| `EWMu` | Exponentially weighted mean | `alpha` (decay factor) |
| `ShrunkMu` | Shrinkage toward grand mean | `shrinkage_method` |
| `EquilibriumMu` | Market equilibrium (CAPM) returns | `risk_aversion` |

All store results in `mu_` attribute after `fit(X)`.

### Covariance

| Estimator | Description | Key Parameters |
|-----------|-------------|----------------|
| `EmpiricalCovariance` | Sample covariance matrix | — |
| `EWCovariance` | Exponentially weighted | `alpha` |
| `LedoitWolf` | Shrinkage toward structured target | — |
| `OAS` | Oracle Approximating Shrinkage | — |
| `ShrunkCovariance` | Parametric shrinkage | `shrinkage` |
| `DenoiseCovariance` | Random Matrix Theory denoising | `n_components` |
| `DetoneCovariance` | Remove market factor | `n_components` |
| `GerberCovariance` | Gerber statistic-based | `threshold` |
| `GraphicalLassoCV` | Sparse precision matrix | `alphas` |
| `ImpliedCovariance` | From options implied vol | requires metadata routing for `implied_vol` |

All store results in `covariance_` attribute after `fit(X)`.

---

## Distance & Clustering

### Distance Estimators

All follow `fit(X)` and store `codependence_` and `distance_` attributes.

| Estimator | Measures |
|-----------|----------|
| `PearsonDistance` | Linear correlation |
| `KendallDistance` | Rank correlation (Kendall tau) |
| `SpearmanDistance` | Rank correlation (Spearman rho) |
| `CovarianceDistance` | Covariance-based |
| `DistanceCorrelation` | Non-linear dependence |
| `MutualInformation` | Information-theoretic |

### HierarchicalClustering

```python
from skfolio.cluster import HierarchicalClustering, LinkageMethod

clustering = HierarchicalClustering(
    linkage_method=LinkageMethod.WARD,
    max_clusters=None,  # or int for fixed cluster count
)
```

---

## Uncertainty Sets

Used with `MeanRisk` for robust optimization.

### Mu Uncertainty Sets

```python
from skfolio.uncertainty_set import (
    EmpiricalMuUncertaintySet,
    BootstrapMuUncertaintySet,
)

model = MeanRisk(
    mu_uncertainty_set_estimator=EmpiricalMuUncertaintySet(
        confidence_level=0.95,
    ),
)
```

### Covariance Uncertainty Sets

```python
from skfolio.uncertainty_set import (
    EmpiricalCovarianceUncertaintySet,
    BootstrapCovarianceUncertaintySet,
)

model = MeanRisk(
    covariance_uncertainty_set_estimator=BootstrapCovarianceUncertaintySet(
        confidence_level=0.95,
    ),
)
```

---

## Pre-Selection Transformers

All follow scikit-learn transformer API (`fit_transform`). Use in `Pipeline`.

| Transformer | Purpose | Key Parameter |
|-------------|---------|---------------|
| `DropCorrelated` | Remove highly correlated assets | `threshold=0.95` |
| `DropZeroVariance` | Remove near-zero variance | — |
| `SelectKExtremes` | Select k best/worst performers | `k`, `highest=True` |
| `SelectNonDominated` | Keep Pareto-optimal assets | — |
| `SelectComplete` | Keep assets with full history | — |
| `SelectNonExpiring` | Exclude soon-expiring assets | `expiration_lookahead` |

```python
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("pre", DropCorrelated(threshold=0.90)),
    ("opt", MeanRisk()),
])
pipe.fit(X)
```

---

## Model Selection

### WalkForward

Rolling/expanding window cross-validation for time series.

```python
from skfolio.model_selection import WalkForward, cross_val_predict

cv = WalkForward(
    test_size=60,    # number of test observations
    train_size=252,  # number of training observations (None = expanding)
)

pred = cross_val_predict(MeanRisk(), X, cv=cv)
print(pred.sharpe_ratio)  # MultiPeriodPortfolio
```

### CombinatorialPurgedCV

Generates multiple testing paths with purging and embargoing.

```python
from skfolio.model_selection import CombinatorialPurgedCV

cv = CombinatorialPurgedCV(
    n_folds=10,
    n_test_folds=8,
    purge_size=0,
    embargo_size=0,
)

pred = cross_val_predict(MeanRisk(), X, cv=cv)
# Returns Population of MultiPeriodPortfolio
print(pred.summary())
```

### MultipleRandomizedCV

Monte Carlo evaluation with asset subsampling and temporal windows.

```python
from skfolio.model_selection import MultipleRandomizedCV

cv = MultipleRandomizedCV(
    walk_forward=WalkForward(test_size=60, train_size=252),
    n_subsamples=10,
    asset_subset_size=10,
    window_size=None,
)
```

### cross_val_predict

Fits estimator on train splits, predicts on test splits. Returns:
- `MultiPeriodPortfolio` for single-path CV (KFold, WalkForward)
- `Population` for multi-path CV (CombinatorialPurgedCV, MultipleRandomizedCV)

```python
from skfolio.model_selection import cross_val_predict

pred = cross_val_predict(model, X, cv=cv)
```

---

## Hyperparameter Tuning

Uses scikit-learn's `GridSearchCV` and `RandomizedSearchCV`.

### Nested Parameter Syntax

Use double-underscore `__` to reach nested estimator parameters:

```python
# model.prior_estimator.mu_estimator.alpha
param_grid = {
    "prior_estimator__mu_estimator__alpha": [0.001, 0.01, 0.1],
    "risk_measure": [RiskMeasure.SEMI_VARIANCE, RiskMeasure.CVAR],
}
```

Discover available params: `model.get_params()`

### make_scorer

```python
from skfolio.metrics import make_scorer
from skfolio import RatioMeasure

scoring = make_scorer(RatioMeasure.SORTINO_RATIO)

# Or custom function:
def custom_score(pred):
    return pred.mean - 2 * pred.variance

scoring = make_scorer(custom_score)
```

### GridSearchCV

```python
from sklearn.model_selection import GridSearchCV, KFold

grid = GridSearchCV(
    estimator=MeanRisk(),
    param_grid=param_grid,
    cv=KFold(n_splits=5, shuffle=False),  # shuffle=False!
    scoring=scoring,
    n_jobs=-1,
)
grid.fit(X)
best = grid.best_estimator_
```

### RandomizedSearchCV

```python
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as stats

rd = RandomizedSearchCV(
    estimator=MeanRisk(),
    param_distributions={"l2_coef": stats.loguniform(0.01, 1)},
    n_iter=50,
    cv=KFold(n_splits=5, shuffle=False),
    n_jobs=-1,
)
```

---

## Metadata Routing

Passes additional data (e.g., implied volatility) through nested estimators.

### Setup Pattern

```python
from sklearn import set_config
set_config(enable_metadata_routing=True)

from skfolio.moments import ImpliedCovariance
from skfolio.prior import EmpiricalPrior
from skfolio.optimization import MeanRisk

model = MeanRisk(
    prior_estimator=EmpiricalPrior(
        covariance_estimator=ImpliedCovariance()
            .set_fit_request(implied_vol=True)
    )
)

model.fit(X, implied_vol=implied_vol)
```

Key steps:
1. `set_config(enable_metadata_routing=True)` — enable globally
2. `.set_fit_request(implied_vol=True)` — declare metadata on the consumer
3. `model.fit(X, implied_vol=implied_vol)` — pass metadata at fit time

---

## Data Preparation

### prices_to_returns

```python
from skfolio.preprocessing import prices_to_returns

X = prices_to_returns(prices)  # linear returns by default
```

### Critical Rules

1. **Always use linear returns** as input `X`: `(P_t / P_{t-1}) - 1`
2. Linear returns aggregate across assets (portfolio return = weighted sum)
3. Log returns aggregate across time but NOT across assets — never use log returns as `X`
4. For investment horizons > 1 year, use `EmpiricalPrior(is_log_normal=True, investment_horizon=...)` instead of square-root scaling
5. Input `X` should be a pandas DataFrame with asset tickers as columns and dates as index

---

## Distribution Module

### VineCopula

Models multivariate dependence structure for synthetic data generation.

```python
from skfolio.distribution import VineCopula, StudentT

copula = VineCopula(
    copulas=[GaussianCopula, StudentTCopula, ClaytonCopula],
    univariate_distributions=[StudentT, JohnsonSU],
)
copula.fit(X)
samples = copula.sample(n_samples=10_000)
```

**Available copulas:** `GaussianCopula`, `StudentTCopula`, `ClaytonCopula`, `GumbelCopula`, `JoeCopula`, `IndependentCopula`

**Univariate distributions:** `Gaussian`, `StudentT`, `JohnsonSU`, `NormalInverseGaussian`

### Stress Testing

```python
prior = SyntheticData(
    distribution_estimator=VineCopula(),
    n_samples=10_000,
    sample_args=dict(
        conditioning={"AAPL": -0.10}  # stress: AAPL drops 10%
    ),
)
```

---

## Portfolio & Population

### Portfolio

Returned by `model.predict(X)`. Key properties:

| Property | Description |
|----------|-------------|
| `returns` | Portfolio return series |
| `cumulative_returns` | Cumulative return series |
| `mean` | Mean return |
| `annualized_mean` | Annualized mean return |
| `variance` | Portfolio variance |
| `standard_deviation` | Portfolio volatility |
| `sharpe_ratio` | Sharpe ratio |
| `sortino_ratio` | Sortino ratio |
| `cvar` | Conditional Value at Risk |
| `max_drawdown` | Maximum drawdown |
| `calmar_ratio` | Calmar ratio |
| `weights` | Asset weights |
| `composition` | DataFrame of weights |

Key methods: `summary()`, `plot_cumulative_returns()`, `plot_composition()`

### MultiPeriodPortfolio

Sequence of portfolios across rebalancing periods. Same properties as Portfolio.

### Population

Collection of portfolios. Key methods:

| Method | Description |
|--------|-------------|
| `summary()` | Summary statistics for all portfolios |
| `plot_cumulative_returns()` | Overlay cumulative returns |
| `plot_composition()` | Compare compositions |
| `plot_frontier()` | Efficient frontier |
| `filter()` | Filter by criteria |
| `sort()` | Sort by measure |

---

## Key Constraints & Gotchas

1. **Always use linear returns** for input `X`, never log returns
2. **Set `shuffle=False`** in any `KFold` or `train_test_split` to prevent temporal data leakage
3. **Enable metadata routing explicitly** with `set_config(enable_metadata_routing=True)` before using `.set_fit_request()`
4. **FactorModel uses `fit(X, y)`** where `X` = asset returns and `y` = factor returns
5. **BenchmarkTracker uses `fit(X, y)`** where `y` = benchmark returns
6. **Group constraints** require a `groups` dict mapping group names to asset lists
7. **Nested parameter tuning** uses double-underscore syntax: `"prior_estimator__mu_estimator__alpha"`
8. **CombinatorialPurgedCV** returns `Population`, not `MultiPeriodPortfolio`
9. Use `optimal_folds_number()` to calibrate `CombinatorialPurgedCV` parameters
10. **Pipeline integration** works with scikit-learn `Pipeline`: pre-selection + optimization

## Implementation Patterns

See @.claude/skills/skfolio/PATTERNS.md for detailed code patterns.
