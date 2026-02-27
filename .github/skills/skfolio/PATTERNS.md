# skfolio Implementation Patterns

## 1. Basic Mean-Variance Optimization

```python
from skfolio.datasets import load_sp500_dataset
from skfolio.optimization import MeanRisk, ObjectiveFunction
from skfolio.preprocessing import prices_to_returns
from skfolio import RiskMeasure

prices = load_sp500_dataset()
X = prices_to_returns(prices)
X_train, X_test = X.iloc[:252], X.iloc[252:]

# Maximize Sharpe ratio (return / volatility)
model = MeanRisk(
    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
    risk_measure=RiskMeasure.STANDARD_DEVIATION,
    min_weights=0.0,
    max_weights=0.15,
)
model.fit(X_train)
portfolio = model.predict(X_test)

print(portfolio.sharpe_ratio)
print(portfolio.composition)
portfolio.plot_cumulative_returns()
```

### Minimize CVaR

```python
model = MeanRisk(
    objective_function=ObjectiveFunction.MINIMIZE_RISK,
    risk_measure=RiskMeasure.CVAR,
)
model.fit(X_train)
```

### Maximize Utility

```python
model = MeanRisk(
    objective_function=ObjectiveFunction.MAXIMIZE_UTILITY,
    risk_measure=RiskMeasure.VARIANCE,
    risk_aversion=2.0,
)
model.fit(X_train)
```

---

## 2. Black-Litterman with Views

```python
from skfolio.optimization import MeanRisk, ObjectiveFunction
from skfolio.prior import BlackLitterman, EmpiricalPrior
from skfolio.moments import EquilibriumMu, LedoitWolf
from skfolio import RiskMeasure

# Equilibrium prior with Ledoit-Wolf covariance
equilibrium_prior = EmpiricalPrior(
    mu_estimator=EquilibriumMu(risk_aversion=2.5),
    covariance_estimator=LedoitWolf(),
)

# Black-Litterman with analyst views
bl_prior = BlackLitterman(
    views=[
        "AAPL == 0.10",           # AAPL returns 10% annually
        "MSFT - GOOG == 0.03",    # MSFT outperforms GOOG by 3%
        "JPM == 0.08",            # JPM returns 8%
    ],
    tau=0.05,
    prior_estimator=equilibrium_prior,
)

model = MeanRisk(
    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
    risk_measure=RiskMeasure.STANDARD_DEVIATION,
    prior_estimator=bl_prior,
    min_weights=0.0,
    max_weights=0.15,
)
model.fit(X_train)
portfolio = model.predict(X_test)
```

---

## 3. Factor Model Pipeline

```python
from skfolio.datasets import load_sp500_dataset, load_factors_dataset
from skfolio.optimization import MeanRisk, ObjectiveFunction
from skfolio.preprocessing import prices_to_returns
from skfolio.prior import FactorModel, EmpiricalPrior
from skfolio.moments import ShrunkMu, DenoiseCovariance
from skfolio import RiskMeasure

prices = load_sp500_dataset()
factor_prices = load_factors_dataset()

X = prices_to_returns(prices)
y = prices_to_returns(factor_prices)

X_train, X_test = X.iloc[:252], X.iloc[252:]
y_train, y_test = y.iloc[:252], y.iloc[252:]

# Factor model with shrunk estimates
factor_prior = FactorModel(
    factor_prior_estimator=EmpiricalPrior(
        mu_estimator=ShrunkMu(),
        covariance_estimator=DenoiseCovariance(),
    ),
)

model = MeanRisk(
    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
    risk_measure=RiskMeasure.CVAR,
    prior_estimator=factor_prior,
    min_weights=0.0,
)

# X = asset returns, y = factor returns
model.fit(X_train, y=y_train)
portfolio = model.predict(X_test)
print(portfolio.sharpe_ratio)
```

### Black-Litterman Factor Model

```python
from skfolio.prior import BlackLitterman

bl_factor = FactorModel(
    factor_prior_estimator=BlackLitterman(
        views=[
            "MTUM == 0.12",       # Momentum factor returns 12%
            "QUAL - VLUE == 0.05", # Quality outperforms Value by 5%
        ],
        tau=0.05,
    ),
)

model = MeanRisk(
    prior_estimator=bl_factor,
    risk_measure=RiskMeasure.CVAR,
)
model.fit(X_train, y=y_train)
```

---

## 4. Entropy Pooling Views

```python
from skfolio.optimization import MeanRisk
from skfolio.prior import EntropyPooling, EmpiricalPrior
from skfolio import RiskMeasure

# Multiple view types
prior = EntropyPooling(
    # Mean views
    mean_views=[
        "JPM == -0.002",           # JPM daily return = -0.2%
        "PG >= LLY",               # P&G >= Eli Lilly
        "BAC >= prior(BAC) * 1.2", # BAC 20% above prior mean
    ],
    # Variance views
    variance_views=[
        "BAC == prior(BAC) * 4",   # BAC variance 4x prior
    ],
    # Correlation views
    correlation_views=[
        "(BAC,JPM) == 0.80",                       # correlation = 0.80
        "(BAC,JNJ) <= prior(BAC,JNJ) * 0.5",       # halve correlation
    ],
    # Skewness views
    skew_views=[
        "BAC == -0.05",            # negative skew for BAC
    ],
    # CVaR views
    cvar_views=[
        "GE == 0.08",             # 95% CVaR of 8%
    ],
    cvar_beta=0.95,
    # Group aggregate views
    groups={"Financials": ["BAC", "JPM"], "Healthcare": ["JNJ", "LLY"]},
    prior_estimator=EmpiricalPrior(),
)

model = MeanRisk(
    risk_measure=RiskMeasure.CVAR,
    prior_estimator=prior,
)
model.fit(X_train)
```

---

## 5. Hierarchical Risk Parity

```python
from skfolio.optimization import HierarchicalRiskParity
from skfolio.prior import EmpiricalPrior
from skfolio.moments import LedoitWolf
from skfolio.distance import KendallDistance
from skfolio.cluster import HierarchicalClustering, LinkageMethod
from skfolio import RiskMeasure

model = HierarchicalRiskParity(
    risk_measure=RiskMeasure.CVAR,
    prior_estimator=EmpiricalPrior(
        covariance_estimator=LedoitWolf(),
    ),
    distance_estimator=KendallDistance(),
    hierarchical_clustering_estimator=HierarchicalClustering(
        linkage_method=LinkageMethod.WARD,
    ),
)
model.fit(X_train)
portfolio = model.predict(X_test)

print(f"Sharpe: {portfolio.sharpe_ratio:.3f}")
print(f"Max DD: {portfolio.max_drawdown:.3%}")
```

### HERC with CDaR

```python
from skfolio.optimization import HierarchicalEqualRiskContribution

model = HierarchicalEqualRiskContribution(
    risk_measure=RiskMeasure.CDAR,
    distance_estimator=KendallDistance(),
    hierarchical_clustering_estimator=HierarchicalClustering(
        linkage_method=LinkageMethod.WARD,
    ),
)
model.fit(X_train)
```

---

## 6. Risk Budgeting

### Equal Risk Contribution

```python
from skfolio.optimization import RiskBudgeting
from skfolio import RiskMeasure

# Equal risk contribution (ERC) — default when risk_budget=None
model = RiskBudgeting(
    risk_measure=RiskMeasure.CVAR,
    min_weights=0.0,
    max_weights=0.20,
)
model.fit(X_train)
portfolio = model.predict(X_test)
```

### Custom Risk Budgets

```python
import numpy as np

n_assets = X_train.shape[1]

# Custom budget: double weight for first 5 assets
budget = np.ones(n_assets)
budget[:5] = 2.0
budget = budget / budget.sum()

model = RiskBudgeting(
    risk_measure=RiskMeasure.VARIANCE,
    risk_budget=budget,
)
model.fit(X_train)
```

---

## 7. Stacking Optimization

```python
from skfolio.optimization import (
    StackingOptimization,
    MeanRisk,
    HierarchicalRiskParity,
    RiskBudgeting,
    ObjectiveFunction,
)
from skfolio import RiskMeasure

# Ensemble of multiple strategies
model = StackingOptimization(
    estimators=[
        ("hrp", HierarchicalRiskParity(risk_measure=RiskMeasure.CVAR)),
        ("erc", RiskBudgeting(risk_measure=RiskMeasure.VARIANCE)),
        ("mv", MeanRisk(
            objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
            risk_measure=RiskMeasure.STANDARD_DEVIATION,
        )),
    ],
    final_estimator=MeanRisk(
        objective_function=ObjectiveFunction.MINIMIZE_RISK,
        risk_measure=RiskMeasure.CVAR,
    ),
)
model.fit(X_train)
portfolio = model.predict(X_test)
```

---

## 8. Pipeline with Pre-Selection

```python
from sklearn.pipeline import Pipeline
from skfolio.pre_selection import DropCorrelated, SelectKExtremes
from skfolio.optimization import MeanRisk, ObjectiveFunction
from skfolio import RiskMeasure

# Drop correlated assets, then select top performers, then optimize
pipe = Pipeline([
    ("drop_corr", DropCorrelated(threshold=0.90)),
    ("select_top", SelectKExtremes(k=10, highest=True)),
    ("optimization", MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
        risk_measure=RiskMeasure.CVAR,
        min_weights=0.0,
    )),
])

pipe.fit(X_train)
portfolio = pipe.predict(X_test)
print(portfolio.sharpe_ratio)
```

### Pipeline with Factor Model

```python
pipe = Pipeline([
    ("drop_corr", DropCorrelated(threshold=0.95)),
    ("optimization", MeanRisk(
        prior_estimator=FactorModel(),
        risk_measure=RiskMeasure.CVAR,
    )),
])

# Factor returns passed as y
pipe.fit(X_train, optimization__y=y_train)
```

---

## 9. Walk-Forward Backtesting

```python
from skfolio.model_selection import WalkForward, cross_val_predict
from skfolio.optimization import MeanRisk, ObjectiveFunction
from skfolio import RiskMeasure

# Walk-forward: 252-day train, 60-day test, rolling
cv = WalkForward(
    test_size=60,
    train_size=252,
)

model = MeanRisk(
    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
    risk_measure=RiskMeasure.CVAR,
    min_weights=0.0,
)

# Returns MultiPeriodPortfolio
pred = cross_val_predict(model, X, cv=cv)

print(f"Sharpe: {pred.sharpe_ratio:.3f}")
print(f"Max DD: {pred.max_drawdown:.3%}")
print(f"CVaR:   {pred.cvar:.3%}")

pred.plot_cumulative_returns()
```

### Expanding Window

```python
cv = WalkForward(
    test_size=60,
    train_size=None,  # expanding window
)
```

### Combinatorial Purged CV

```python
from skfolio.model_selection import CombinatorialPurgedCV

cv = CombinatorialPurgedCV(
    n_folds=10,
    n_test_folds=8,
    purge_size=5,
    embargo_size=5,
)

# Returns Population of MultiPeriodPortfolio
pred = cross_val_predict(model, X, cv=cv)
print(pred.summary())
```

---

## 10. Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV, KFold
from skfolio.optimization import MeanRisk, ObjectiveFunction
from skfolio.metrics import make_scorer
from skfolio import RiskMeasure, RatioMeasure

# Define parameter grid
param_grid = [
    {
        "risk_measure": [RiskMeasure.CVAR, RiskMeasure.SEMI_VARIANCE],
        "l2_coef": [0.001, 0.01, 0.1],
    },
    {
        "risk_measure": [RiskMeasure.CDAR],
        "l1_coef": [0.001, 0.01],
        "l2_coef": [0.01, 0.1],
    },
]

# Custom scorer
scoring = make_scorer(RatioMeasure.SORTINO_RATIO)

grid = GridSearchCV(
    estimator=MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
        min_weights=0.0,
    ),
    param_grid=param_grid,
    cv=KFold(n_splits=5, shuffle=False),  # NEVER shuffle financial data
    scoring=scoring,
    n_jobs=-1,
)

grid.fit(X_train)
print(f"Best params: {grid.best_params_}")
print(f"Best score: {grid.best_score_:.4f}")

# Use best model
portfolio = grid.best_estimator_.predict(X_test)
```

### Nested Parameter Tuning

```python
from skfolio.prior import EmpiricalPrior
from skfolio.moments import EWMu, LedoitWolf

model = MeanRisk(
    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
    risk_measure=RiskMeasure.CVAR,
    prior_estimator=EmpiricalPrior(
        mu_estimator=EWMu(alpha=0.2),
        covariance_estimator=LedoitWolf(),
    ),
)

# Tune nested estimator parameters
param_grid = {
    "prior_estimator__mu_estimator__alpha": [0.01, 0.05, 0.1, 0.2, 0.5],
}

grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=KFold(n_splits=5, shuffle=False),
    n_jobs=-1,
)
grid.fit(X_train)
```

---

## 11. Robust Optimization

```python
from skfolio.optimization import MeanRisk, ObjectiveFunction
from skfolio.uncertainty_set import (
    BootstrapMuUncertaintySet,
    BootstrapCovarianceUncertaintySet,
)
from skfolio import RiskMeasure

# Worst-case optimization over uncertainty sets
model = MeanRisk(
    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
    risk_measure=RiskMeasure.CVAR,
    mu_uncertainty_set_estimator=BootstrapMuUncertaintySet(
        confidence_level=0.95,
    ),
    covariance_uncertainty_set_estimator=BootstrapCovarianceUncertaintySet(
        confidence_level=0.95,
    ),
    min_weights=0.0,
)

model.fit(X_train)
portfolio = model.predict(X_test)
print(f"Robust Sharpe: {portfolio.sharpe_ratio:.3f}")
```

### Empirical Uncertainty Sets

```python
from skfolio.uncertainty_set import (
    EmpiricalMuUncertaintySet,
    EmpiricalCovarianceUncertaintySet,
)

model = MeanRisk(
    mu_uncertainty_set_estimator=EmpiricalMuUncertaintySet(
        confidence_level=0.90,
    ),
    covariance_uncertainty_set_estimator=EmpiricalCovarianceUncertaintySet(
        confidence_level=0.90,
    ),
)
```

---

## 12. Synthetic Data & Stress Testing

```python
from skfolio.distribution import VineCopula, StudentT, JohnsonSU
from skfolio.prior import SyntheticData
from skfolio.optimization import MeanRisk
from skfolio import RiskMeasure

# Fit vine copula and generate synthetic scenarios
copula = VineCopula(
    copulas="all",  # try all copula types
    univariate_distributions=[StudentT, JohnsonSU],
)

prior = SyntheticData(
    distribution_estimator=copula,
    n_samples=10_000,
)

model = MeanRisk(
    risk_measure=RiskMeasure.CVAR,
    prior_estimator=prior,
    min_weights=0.0,
)
model.fit(X_train)
```

### Stressed Factor Model

```python
from skfolio.prior import FactorModel, SyntheticData

# Stress test: factor drops 10%
factor_prior = SyntheticData(
    distribution_estimator=VineCopula(),
    n_samples=10_000,
    sample_args=dict(
        conditioning={"MTUM": -0.10}  # momentum crashes
    ),
)

prior = FactorModel(
    factor_prior_estimator=factor_prior,
)

model = MeanRisk(
    risk_measure=RiskMeasure.CVAR,
    prior_estimator=prior,
)
model.fit(X_train, y=y_train)
```

---

## 13. Opinion Pooling

```python
from skfolio.prior import OpinionPooling, EntropyPooling, EmpiricalPrior

# Expert 1: bullish on tech
expert_1 = EntropyPooling(
    mean_views=[
        "AAPL == 0.001",
        "MSFT == 0.0008",
    ],
)

# Expert 2: bearish on tech, bullish on healthcare
expert_2 = EntropyPooling(
    mean_views=[
        "AAPL == -0.0005",
        "JNJ == 0.001",
    ],
)

# Combine experts with weighted opinions
prior = OpinionPooling(
    estimators=[
        ("tech_bull", expert_1),
        ("healthcare_bull", expert_2),
    ],
    opinion_probabilities=[0.4, 0.5],  # 0.1 → base prior
    prior_estimator=EmpiricalPrior(),
)

model = MeanRisk(
    risk_measure=RiskMeasure.CVAR,
    prior_estimator=prior,
)
model.fit(X_train)
```

---

## 14. Custom Scoring

```python
from skfolio.metrics import make_scorer
from skfolio import RatioMeasure

# Built-in ratio measure
scoring = make_scorer(RatioMeasure.CALMAR_RATIO)

# Custom scoring function — receives a Portfolio object
def risk_adjusted_score(portfolio):
    """Penalize high drawdowns and reward Sharpe."""
    return portfolio.sharpe_ratio - 2.0 * abs(portfolio.max_drawdown)

scoring = make_scorer(risk_adjusted_score)

# Another custom scorer: mean-variance-skew utility
def mvsv_utility(portfolio):
    return (
        portfolio.annualized_mean
        - 0.5 * portfolio.variance
        - (1 / 6) * portfolio.cvar
    )

scoring = make_scorer(mvsv_utility)

# Use in GridSearchCV
from sklearn.model_selection import GridSearchCV, KFold

grid = GridSearchCV(
    estimator=MeanRisk(),
    param_grid={"risk_measure": [RiskMeasure.CVAR, RiskMeasure.VARIANCE]},
    cv=KFold(n_splits=5, shuffle=False),
    scoring=scoring,
)
grid.fit(X_train)
```

---

## 15. Metadata Routing

```python
from sklearn import set_config
set_config(enable_metadata_routing=True)

from skfolio.datasets import load_sp500_dataset, load_sp500_implied_vol_dataset
from skfolio.moments import ImpliedCovariance
from skfolio.optimization import MeanRisk, ObjectiveFunction
from skfolio.preprocessing import prices_to_returns
from skfolio.prior import EmpiricalPrior
from skfolio import RiskMeasure

# Load data
prices = load_sp500_dataset()
implied_vol = load_sp500_implied_vol_dataset()

X = prices_to_returns(prices)
X_train, X_test = X.iloc[:252], X.iloc[252:]
implied_vol_train = implied_vol.iloc[:252]

# Build model with metadata-aware covariance estimator
model = MeanRisk(
    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
    risk_measure=RiskMeasure.CVAR,
    prior_estimator=EmpiricalPrior(
        covariance_estimator=ImpliedCovariance()
            .set_fit_request(implied_vol=True)
    ),
)

# Pass implied_vol through the entire estimator chain
model.fit(X_train, implied_vol=implied_vol_train)
portfolio = model.predict(X_test)
```

### Metadata in GridSearchCV

```python
from sklearn.model_selection import GridSearchCV, KFold

grid = GridSearchCV(
    estimator=model,
    param_grid={"risk_measure": [RiskMeasure.CVAR, RiskMeasure.VARIANCE]},
    cv=KFold(n_splits=5, shuffle=False),
)

# Metadata is automatically routed through CV splits
grid.fit(X_train, implied_vol=implied_vol_train)
```

---

## 16. Full Production Pipeline

```python
"""
End-to-end portfolio optimization combining:
- Pre-selection filtering
- Factor model with Black-Litterman views
- Ledoit-Wolf covariance shrinkage
- Walk-forward backtesting
- Hyperparameter tuning
"""
from sklearn import set_config
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline

from skfolio import RiskMeasure, RatioMeasure
from skfolio.datasets import load_sp500_dataset, load_factors_dataset
from skfolio.metrics import make_scorer
from skfolio.model_selection import WalkForward, cross_val_predict
from skfolio.moments import ShrunkMu, LedoitWolf
from skfolio.optimization import MeanRisk, ObjectiveFunction
from skfolio.pre_selection import DropCorrelated, SelectComplete
from skfolio.preprocessing import prices_to_returns
from skfolio.prior import FactorModel, BlackLitterman, EmpiricalPrior

# ── Data ─────────────────────────────────────────────────────────
prices = load_sp500_dataset()
factor_prices = load_factors_dataset()

X = prices_to_returns(prices)           # linear returns
y = prices_to_returns(factor_prices)    # factor returns

# ── Factor Model with BL Views ──────────────────────────────────
bl_factor_prior = FactorModel(
    factor_prior_estimator=BlackLitterman(
        views=[
            "MTUM == 0.10",            # momentum factor +10%
            "QUAL - VLUE == 0.04",     # quality outperforms value
        ],
        tau=0.05,
        prior_estimator=EmpiricalPrior(
            mu_estimator=ShrunkMu(),
            covariance_estimator=LedoitWolf(),
        ),
    ),
)

# ── Pipeline: Pre-Selection + Optimization ───────────────────────
pipe = Pipeline([
    ("complete", SelectComplete()),
    ("drop_corr", DropCorrelated(threshold=0.95)),
    ("optimization", MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
        risk_measure=RiskMeasure.CVAR,
        prior_estimator=bl_factor_prior,
        min_weights=0.0,
        max_weights=0.15,
        l2_coef=0.01,
    )),
])

# ── Hyperparameter Tuning ───────────────────────────────────────
param_grid = {
    "drop_corr__threshold": [0.90, 0.95],
    "optimization__risk_measure": [RiskMeasure.CVAR, RiskMeasure.SEMI_VARIANCE],
    "optimization__l2_coef": [0.001, 0.01, 0.1],
}

scoring = make_scorer(RatioMeasure.SORTINO_RATIO)

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=KFold(n_splits=5, shuffle=False),
    scoring=scoring,
    n_jobs=-1,
)

# Factor returns passed to optimization step
grid.fit(X, optimization__y=y)
best_pipe = grid.best_estimator_

print(f"Best params: {grid.best_params_}")

# ── Walk-Forward Backtest ────────────────────────────────────────
cv = WalkForward(test_size=60, train_size=252)

pred = cross_val_predict(
    best_pipe,
    X,
    cv=cv,
    params=dict(optimization__y=y),
)

print(f"\nWalk-Forward Results:")
print(f"  Sharpe Ratio:  {pred.sharpe_ratio:.3f}")
print(f"  Sortino Ratio: {pred.sortino_ratio:.3f}")
print(f"  Max Drawdown:  {pred.max_drawdown:.3%}")
print(f"  CVaR (95%):    {pred.cvar:.3%}")

pred.plot_cumulative_returns()
```
